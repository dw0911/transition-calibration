# -*- coding: utf-8 -*-
"""E7 STID 训练（四段协议，与 e7_train.py / train_r3.py 同口径）。

为什么需要这个脚本：`audit/stid_compliance.md` 判定既有 PEMS04 STID checkpoint
**NOT COMPLIANT**（窗口编号切分、无 purge、calib 目标与 test 输入重叠、全局单标量归一化、
norm 空间掩码与选择指标）。第二骨干验证必须在冻结协议下重训 → PEMS08 STID ×3 seed。

与 legacy `train_stid_fast.py` 的差异（全部朝冻结协议靠拢）：
  1. 数据源：BasicTS058/datasets/<ds>/data.dat 原始时间轴（不再用 legacy 60/20/20 npz）
  2. 边界：train[0,int(T·.6)) / select[int(T·.6),int(T·.7)) ；窗口起点半开区间 [start, end−23)
  3. 归一化：**per-node mean/std，仅 train 段估计**，作用于输入 flow 通道（ch0）与输出的逆变换；
     ch1/ch2（tod/dow 分数）保持原值。legacy 输入形态同样是"归一化后的三通道窗口"，
     但用的是全局单标量 scaler（实测 PEMS04 mean=207.4577/std=156.0088）→ 此处改为 per-node，
     与 e7_train.py（STAEformer）在同一条冻结协议下完全对齐。
  4. 损失与选择指标：**raw 空间 masked MAE，掩码 = raw y > 0**（legacy 为 norm 空间 != 0，实测不排除任何点）
  5. tod/dow：由 data.dat 的 ch1/ch2 还原为整数索引（ch1·288、ch2·7），per-window 取输入首步
  6. checkpoint 写入 revision_v1/checkpoints/，train_meta 记录数据与脚本 sha256、每 epoch 墙钟、峰值显存
保留 legacy 的 STID 专有配置：batch 128、Adam lr=2e-3 wd=0、MultiStepLR[20,25] γ=0.1、
  早停 MIN_EP=20 / PATIENCE=3 / MIN_DELTA=0.01、验证频次同 R3。
只读 train 与 select 两段，**从不读取 calib 与 test**。

用法:
  python e7_train_stid.py --dataset PEMS08 --seed 42 --pilot 3
  python e7_train_stid.py --dataset PEMS08 --seed 42 --epochs 100
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
sys.path.insert(0, os.path.join(PROJ, 'baselines', 'common'))
sys.path.insert(0, os.path.join(PROJ, 'baselines', 'STID'))
os.chdir(BASICTS)

import numpy as np                                  # noqa: E402
import torch                                        # noqa: E402
from model import STID                              # noqa: E402

IL, OL, PURGE = 12, 12, 23
CKPT_ROOT = os.path.join(ROOT, 'checkpoints')


def sha256(path, limit=400 * 1024 * 1024):
    h = hashlib.sha256(); n = os.path.getsize(path)
    with open(path, 'rb') as f:
        if n > limit:
            h.update(f.read(1 << 20)); return 'partial1MB:' + h.hexdigest()
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def main(dataset, max_epochs, seed, pilot):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    dpath = os.path.join(BASICTS, 'datasets', dataset, 'data.dat')
    desc = json.load(open(os.path.join(BASICTS, 'datasets', dataset, 'desc.json')))
    shape = tuple(desc['shape'])
    T, N = shape[0], shape[1]
    train_end, select_end = int(T * .6), int(T * .7)
    mode = f'PILOT({pilot}ep)' if pilot else f'FULL({max_epochs}ep)'
    print(f'[E7-STID] {dataset} seed{seed} {mode} T={T} N={N}', flush=True)
    print(f'  train[0,{train_end}) select[{train_end},{select_end}) | calib/test 不读取', flush=True)

    full = np.array(np.memmap(dpath, dtype='float32', mode='r', shape=shape))
    mean = full[:train_end, :, 0].mean(axis=0)
    std = full[:train_end, :, 0].std(axis=0); std[std == 0] = 1.0
    mean_t = torch.tensor(mean, dtype=torch.float32, device='cuda')
    std_t = torch.tensor(std, dtype=torch.float32, device='cuda')
    # tod/dow 整数索引：ch1 ∈ {k/288}、ch2 ∈ {k/7}，取窗口输入首步（对所有节点相同）
    tod_all = np.rint(full[:, 0, 1] * 288).astype(np.int64).clip(0, 287)
    dow_all = np.rint(full[:, 0, 2] * 7).astype(np.int64).clip(0, 6)

    g_tr = np.arange(0, train_end - PURGE)
    g_se = np.arange(train_end, select_end - PURGE)
    print(f'  windows: train={len(g_tr)} select={len(g_se)}', flush=True)

    def batch(gs, idx):
        """输入 ch0 做 per-node 归一化（train-only 统计量），ch1/ch2 原值；y 保持 raw 用于 raw 空间损失。"""
        g = gs[idx]
        x = full[g[:, None] + np.arange(IL)].copy()                  # (B,12,N,3) raw
        y = full[g[:, None] + np.arange(IL, IL + OL)].copy()          # (B,12,N,3) raw
        xt = torch.from_numpy(x).cuda()
        xt[..., 0] = (xt[..., 0] - mean_t) / std_t                    # per-node norm on flow only
        return xt, torch.from_numpy(y[..., 0]).cuda(), \
            torch.from_numpy(tod_all[g]).cuda(), torch.from_numpy(dow_all[g]).cuda()

    model = STID(num_nodes=N, input_dim=3, his_steps=IL, pred_steps=OL).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=0)
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[20, 25], gamma=0.1)
    BS_TR, BS_VA = 128, 256
    nparam = sum(p.numel() for p in model.parameters())

    def to_raw(out):
        o = out.unsqueeze(-1) if out.dim() == 3 else out              # (B,12,N,1)
        return o.squeeze(-1) * std_t + mean_t                         # (B,12,N)

    def masked_mae(out, y_raw):
        o = to_raw(out)
        m = y_raw > 0
        return (torch.abs(o - y_raw) * m).sum() / m.sum().clamp(min=1)

    @torch.no_grad()
    def val():
        model.eval(); es, c = 0.0, 0
        for bi in range(0, len(g_se), BS_VA):
            idx = np.arange(bi, min(bi + BS_VA, len(g_se)))
            xt, yt, tod, dow = batch(g_se, idx)
            o = to_raw(model(xt, tod, dow))
            m = yt > 0
            es += float((torch.abs(o - yt) * m).sum()); c += int(m.sum())
        model.train()
        return es / max(c, 1)

    ck = os.path.join(CKPT_ROOT, f'{dataset}_stid_e7_4split_seed{seed}')
    os.makedirs(ck, exist_ok=True)
    best, best_ep, pat = float('inf'), 0, 0
    MIN_EP, PATIENCE, MIN_DELTA = 20, 3, 0.01
    ep_times, t0 = [], time.time()
    torch.cuda.reset_peak_memory_stats()
    epochs = pilot if pilot else max_epochs
    for ep in range(1, epochs + 1):
        te = time.time(); model.train()
        perm = np.random.permutation(len(g_tr))
        ls, nb = 0.0, 0
        for bi in range(0, len(perm), BS_TR):
            idx = perm[bi:bi + BS_TR]
            xt, yt, tod, dow = batch(g_tr, idx)
            loss = masked_mae(model(xt, tod, dow), yt)
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
                            'seed': seed, 'model': 'STID',
                            'protocol': 'E7 4-split canonical (per-node norm, raw masked MAE, purge 23)',
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

    meta = {'dataset': dataset, 'model': 'STID', 'seed': seed,
            'mode': 'pilot' if pilot else 'full', 'T': T, 'N': N, 'n_params': nparam,
            'boundaries': {'train_end': train_end, 'select_end': select_end},
            'purge': PURGE, 'n_train_windows': len(g_tr), 'n_select_windows': len(g_se),
            'batch_size_train': BS_TR, 'batch_size_val': BS_VA, 'amp': False,
            'normalization': 'per-node, train-only', 'loss': 'raw masked MAE (mask: raw y > 0)',
            'selection_metric': 'raw masked MAE on selection segment',
            'epochs_run': len(ep_times), 'sec_per_epoch_mean': float(np.mean(ep_times)),
            'sec_per_epoch_last3': float(np.mean(ep_times[-3:])),
            'peak_vram_gb': float(torch.cuda.max_memory_allocated() / 1024 ** 3),
            'gpu': torch.cuda.get_device_name(0),
            'extrapolated_100ep_hours': float(np.mean(ep_times[-3:]) * 100 / 3600),
            'data_sha256': sha256(dpath),
            'script_sha256': sha256(os.path.join(ROOT, 'scripts', 'e7_train_stid.py')),
            'best_mae': best if not pilot else None, 'best_epoch': best_ep if not pilot else None,
            'note': 'train/select only; calibration and test never read'}
    fn = os.path.join(ck, ('pilot_meta.json' if pilot else 'train_meta.json'))
    json.dump(meta, open(fn, 'w'), indent=2)
    print(f'\n[E7-STID] {dataset} seed{seed}: {len(ep_times)} epochs, '
          f'{np.mean(ep_times[-3:]):.1f}s/epoch(last3), peak VRAM {meta["peak_vram_gb"]:.2f} GB, '
          f'外推 100ep ≈ {meta["extrapolated_100ep_hours"]:.2f} h', flush=True)
    print('saved ->', fn)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--pilot', type=int, default=0)
    a = ap.parse_args()
    main(a.dataset, a.epochs, a.seed, a.pilot)
