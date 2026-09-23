# -*- coding: utf-8 -*-
"""E7 点预测训练（四段协议，与 uncalib_v1/SCRIPTS/train_r3.py 同口径）。

与 train_r3.py 的差异（均为溯源与扩展需要，不改变训练语义）：
  1. N 从 desc.json 的 shape 读取，不再硬编码数据集字典 → 支持 PEMS08
  2. checkpoint 写入 revision_v1/checkpoints/，**不写入 uncalib_v1 原始实验目录**
  3. train_meta 记录 data.dat 的 sha256、本脚本 sha256、每 epoch 墙钟、峰值显存
  4. --pilot E：只跑 E 个 epoch 测时长与显存并外推 100 epoch 总耗时，不产出可用权重
  5. 训练与验证只使用 train / selection 两段，**从不读取 calibration 与 test**
     （因此 PEMS08 的 pilot 与训练不构成"打开 PEMS08 test"）

用法:
  python e7_train.py --dataset PEMS08 --seed 42 --pilot 3      # 计时
  python e7_train.py --dataset PEMS08 --seed 42 --epochs 100    # 正式训练
"""
import os
import sys
import json
import time
import random
import hashlib
import argparse

_PKG = os.path.dirname(os.path.abspath(__file__))
PROJ = os.environ.get('UC_PROJ') or os.path.abspath(
    os.path.join(_PKG, os.pardir, os.pardir))   # parent of revision_v1; override: UC_PROJ
BASICTS = os.path.join(PROJ, 'BasicTS058')
ROOT = os.path.join(PROJ, 'revision_v1')
sys.path.insert(0, BASICTS)
sys.path.insert(0, os.path.join(BASICTS, 'baselines', 'STAEformer'))
os.chdir(BASICTS)

import numpy as np                                  # noqa: E402
import torch                                        # noqa: E402

IL, OL = 12, 12
CKPT_ROOT = os.path.join(ROOT, 'checkpoints')


def sha256(path, limit=400 * 1024 * 1024):
    h = hashlib.sha256()
    n = os.path.getsize(path)
    with open(path, 'rb') as f:
        if n > limit:
            h.update(f.read(1 << 20))
            return 'partial1MB:' + h.hexdigest()
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def main(dataset, max_epochs, seed, pilot):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    from baselines.STAEformer.arch.staeformer_arch import STAEformer

    dpath = os.path.join(BASICTS, 'datasets', dataset, 'data.dat')
    desc = json.load(open(os.path.join(BASICTS, 'datasets', dataset, 'desc.json')))
    shape = tuple(desc['shape'])
    T, N = shape[0], shape[1]
    train_end, select_end, calib_end = int(T * .6), int(T * .7), int(T * .8)
    mode = f'PILOT({pilot}ep)' if pilot else f'FULL({max_epochs}ep)'
    print(f'[E7-train] {dataset} seed{seed} {mode} T={T} N={N}', flush=True)
    print(f'  train[0,{train_end}) select[{train_end},{select_end}) '
          f'| calib/test 不读取', flush=True)

    full = np.array(np.memmap(dpath, dtype='float32', mode='r', shape=shape))
    mean = full[:train_end, :, 0].mean(axis=0)
    std = full[:train_end, :, 0].std(axis=0); std[std == 0] = 1.0
    mean_t = torch.tensor(mean, dtype=torch.float32, device='cuda')
    std_t = torch.tensor(std, dtype=torch.float32, device='cuda')

    class RawWindowDataset(torch.utils.data.Dataset):
        def __init__(self, g0, g1):
            self.gs = np.arange(g0, g1)
        def __len__(self):
            return len(self.gs)
        def __getitem__(self, i):
            g = self.gs[i]
            return {'x': torch.from_numpy(np.array(full[g:g + IL]).astype(np.float32)),
                    'y': torch.from_numpy(np.array(full[g + IL:g + IL + OL]).astype(np.float32))}

    ds_train = RawWindowDataset(0, train_end - IL - OL + 1)
    ds_select = RawWindowDataset(train_end, select_end - IL - OL + 1)

    def collate(b):
        return torch.stack([x['x'] for x in b]), torch.stack([x['y'] for x in b])

    ld_tr = torch.utils.data.DataLoader(ds_train, batch_size=16, shuffle=True, num_workers=0,
                                        pin_memory=False, collate_fn=collate)
    ld_se = torch.utils.data.DataLoader(ds_select, batch_size=64, shuffle=False, num_workers=0,
                                        pin_memory=False, collate_fn=collate)
    print(f'  windows: train={len(ds_train)} select={len(ds_select)}', flush=True)

    model = STAEformer(num_nodes=N, in_steps=12, out_steps=12, steps_per_day=288,
                       input_dim=3, output_dim=1, input_embedding_dim=24, tod_embedding_dim=24,
                       dow_embedding_dim=24, spatial_embedding_dim=0, adaptive_embedding_dim=80,
                       feed_forward_dim=256, num_heads=4, num_layers=3, dropout=0.1,
                       use_mixed_proj=True).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=3e-4)
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[20, 25], gamma=0.1)
    USE_AMP = False
    nparam = sum(p.numel() for p in model.parameters())

    def masked_mae(pred_norm, y_raw):
        out = pred_norm.squeeze(-1) * std_t + mean_t
        t = y_raw[..., 0]; m = t > 0
        return (torch.abs(out - t) * m).sum() / m.sum().clamp(min=1), out

    @torch.no_grad()
    def val():
        model.eval(); es, c = 0.0, 0
        for xr, yr in ld_se:
            xr = xr.cuda(); xn = xr.clone()
            xn[..., 0] = (xn[..., 0] - mean_t) / std_t
            f = torch.zeros(xn.shape[0], OL, N, 3, device='cuda')
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                o = model(history_data=xn[:, :, :, [0, 1, 2]], future_data=f,
                          batch_seen=0, epoch=0, train=False)
            if isinstance(o, dict):
                o = o['prediction']
            t = yr.cuda()[..., 0]; mk = t > 0
            es += float((torch.abs((o.float().squeeze(-1) * std_t + mean_t) - t) * mk).sum())
            c += int(mk.sum())
        model.train()
        return es / max(c, 1)

    ck = os.path.join(CKPT_ROOT, f'{dataset}_e7_4split_seed{seed}')
    os.makedirs(ck, exist_ok=True)
    best, best_ep, pat = float('inf'), 0, 0
    MIN_EP, PATIENCE, MIN_DELTA = 20, 3, 0.01
    ep_times, t0 = [], time.time()
    torch.cuda.reset_peak_memory_stats()
    epochs = pilot if pilot else max_epochs
    for ep in range(1, epochs + 1):
        te = time.time(); model.train(); ls, nb = 0.0, 0
        for bi, (xr, yr) in enumerate(ld_tr):
            xr = xr.cuda(); xn = xr.clone()
            xn[..., 0] = (xn[..., 0] - mean_t) / std_t
            f = torch.zeros(xn.shape[0], OL, N, 3, device='cuda')
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                o = model(history_data=xn[:, :, :, [0, 1, 2]], future_data=f,
                          batch_seen=bi, epoch=ep, train=True)
            if isinstance(o, dict):
                o = o['prediction']
            loss, _ = masked_mae(o.float(), yr.cuda())
            opt.zero_grad(); loss.backward(); opt.step()
            ls += float(loss.item()); nb += 1
        sch.step()
        ep_times.append(time.time() - te)
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        line = (f'ep {ep}/{epochs} loss={ls/nb:.4f} ep_time={ep_times[-1]:.1f}s '
                f'peak_vram={peak:.2f}GB elapsed={(time.time()-t0)/60:.1f}min')
        if not pilot and ((ep == 1) or (ep < 30 and ep % 5 == 0) or (ep >= 30 and ep % 10 == 0)):
            v = val()
            if v < best - MIN_DELTA:
                best, best_ep, pat = v, ep, 0
                torch.save({'model_state_dict': model.state_dict(), 'best_mae': v, 'best_epoch': ep,
                            'seed': seed, 'protocol': 'E7 R3-equivalent 4-split canonical',
                            'dataset_sha256': sha256(dpath)}, os.path.join(ck, 'best.pt'))
            else:
                pat += 1
            line += f' val_rawMAE={v:.4f} best={best:.4f}@ep{best_ep} pat={pat}/{PATIENCE}'
            print(line, flush=True)
            if ep >= MIN_EP and pat >= PATIENCE:
                print(f'Early stop @ep{ep}', flush=True)
                break
        else:
            print(line, flush=True)

    meta = {'dataset': dataset, 'seed': seed, 'mode': 'pilot' if pilot else 'full',
            'T': T, 'N': N, 'n_params': nparam,
            'boundaries': {'train_end': train_end, 'select_end': select_end,
                           'calib_end': calib_end, 'test_end': T},
            'n_train_windows': len(ds_train), 'n_select_windows': len(ds_select),
            'batch_size': 16, 'amp': USE_AMP,
            'epochs_run': len(ep_times),
            'sec_per_epoch_mean': float(np.mean(ep_times)), 'sec_per_epoch_last3':
                float(np.mean(ep_times[-3:])),
            'peak_vram_gb': float(torch.cuda.max_memory_allocated() / 1024 ** 3),
            'gpu': torch.cuda.get_device_name(0),
            'extrapolated_100ep_hours': float(np.mean(ep_times[-3:]) * 100 / 3600),
            'data_sha256': sha256(dpath),
            'script_sha256': sha256(os.path.join(ROOT, 'scripts', 'e7_train.py')),
            'best_mae': best if not pilot else None, 'best_epoch': best_ep if not pilot else None,
            'note': 'train/select only; calibration and test never read'}
    fn = os.path.join(ck, ('pilot_meta.json' if pilot else 'train_meta.json'))
    json.dump(meta, open(fn, 'w'), indent=2)
    print(f'\n[E7] {dataset} seed{seed}: {len(ep_times)} epochs, '
          f'{np.mean(ep_times[-3:]):.1f}s/epoch(last3), '
          f'peak VRAM {meta["peak_vram_gb"]:.2f} GB, '
          f'外推 100ep ≈ {meta["extrapolated_100ep_hours"]:.2f} h', flush=True)
    print('saved ->', fn)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--pilot', type=int, default=0, help='只跑 N 个 epoch 计时后退出')
    a = ap.parse_args()
    main(a.dataset, a.epochs, a.seed, a.pilot)
