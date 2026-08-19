# -*- coding: utf-8 -*-
"""R3 training: STAEformer scratch four-split (Round1 audit P0-1 fix).

Fixes vs the old train_ucg3b_fast.py:
  P0-1  transform correctly wired: BasicTS dataset returns raw; this script explicitly applies
        x_norm = (x_raw - mean_n)/std_n (per-node, train-only, same source as ZScoreScaler),
        the model consumes norm and outputs out_norm -> out_raw = out_norm*std + mean,
        loss = raw-space masked MAE (mask: y_raw > 0, same convention as the official runner)
  boundary  select window strictly <= select_end-24 (does not cross into calib); four-split
        boundary int(T*r) unified with eval
  P1-6  calib eval split starts from select_end+23 (purges 23 anchors, same convention as r3_eval)
  fast_val changed to raw masked MAE (comparable to official val; under the old raw-input the
        "norm MAE" was actually the raw MAE, mislabeled)

Usage: python train_r3.py --dataset PEMS04 --epochs 100 --seed 42
"""
import os
import sys
import json
import time
import argparse
import random

REPRO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASICTS = os.environ.get('BASICTS_ROOT', '')
UC = REPRO_ROOT
sys.path.insert(0, BASICTS)
sys.path.insert(0, os.path.join(BASICTS, 'baselines', 'STAEformer'))
os.chdir(BASICTS)

import numpy as np
import torch


def main(dataset, max_epochs, seed=42, smoke=False):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    print(f'[R3-train] dataset={dataset} max_epochs={max_epochs} seed={seed} smoke={smoke}', flush=True)

    from baselines.STAEformer.arch.staeformer_arch import STAEformer

    desc = json.load(open(os.path.join(BASICTS, 'datasets', dataset, 'desc.json')))
    shape = tuple(desc['shape'])
    T = shape[0]
    N = {'PEMS03': 358, 'PEMS04': 307, 'PEMS07': 883}[dataset]
    # four-split boundary (unified with eval: int(T*r))
    train_end = int(T * 0.6)
    select_end = int(T * 0.7)
    calib_end = int(T * 0.8)
    print(f'  T={T} N={N} train[0,{train_end}) select[{train_end},{select_end}) '
          f'calib[{select_end},{calib_end}) test[{calib_end},{T})', flush=True)

    # ---- data (raw; Plan A: fully preload into memory to avoid repeated memmap reads) and per-node scaler ----
    full = np.array(np.memmap(os.path.join(BASICTS, 'datasets', dataset, 'data.dat'),
                              dtype='float32', mode='r', shape=shape))
    mean = full[:train_end, :, 0].mean(axis=0)                 # (N,)
    std = full[:train_end, :, 0].std(axis=0)
    std[std == 0] = 1.0
    mean_t = torch.tensor(mean, dtype=torch.float32, device='cuda')
    std_t = torch.tensor(std, dtype=torch.float32, device='cuda')

    IL, OL = 12, 12

    class RawWindowDataset(torch.utils.data.Dataset):
        """Raw sliding window (start g), with its own index built within the split boundaries (select does not cross into calib)."""
        def __init__(self, g_start, g_end_excl):
            self.gs = np.arange(g_start, g_end_excl)
        def __len__(self):
            return len(self.gs)
        def __getitem__(self, i):
            g = self.gs[i]
            x = np.array(full[g:g + IL])                        # (12,N,3) raw
            y = np.array(full[g + IL:g + IL + OL])              # (12,N,3) raw
            return {'x': torch.from_numpy(x.astype(np.float32)),
                    'y': torch.from_numpy(y.astype(np.float32))}

    ds_train = RawWindowDataset(0, train_end - IL - OL + 1)
    ds_select = RawWindowDataset(train_end, select_end - IL - OL + 1)   # last window output end <= select_end-1
    print(f'  windows: train={len(ds_train)} select={len(ds_select)}', flush=True)

    def collate_norm(batch):
        # Plan A: collate only does a CPU stack (no CPU<->GPU round-trip); norm is applied uniformly on the GPU side in the training loop
        x = torch.stack([b['x'] for b in batch])                # (B,12,N,3) raw CPU
        y = torch.stack([b['y'] for b in batch])
        return x, y

    loader_train = torch.utils.data.DataLoader(ds_train, batch_size=16, shuffle=True,
                                               num_workers=0, pin_memory=False,
                                               collate_fn=collate_norm)
    loader_select = torch.utils.data.DataLoader(ds_select, batch_size=64, shuffle=False,
                                                num_workers=0, pin_memory=False,
                                                collate_fn=collate_norm)

    model = STAEformer(num_nodes=N, in_steps=12, out_steps=12, steps_per_day=288,
                       input_dim=3, output_dim=1, input_embedding_dim=24, tod_embedding_dim=24,
                       dow_embedding_dim=24, spatial_embedding_dim=0, adaptive_embedding_dim=80,
                       feed_forward_dim=256, num_heads=4, num_layers=3, dropout=0.1,
                       use_mixed_proj=True).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 25], gamma=0.1)
    USE_AMP = False

    def raw_masked_mae(pred_norm, y_raw):
        out_raw = pred_norm.squeeze(-1) * std_t + mean_t        # (B,12,N)
        t = y_raw[..., 0]
        m = (t > 0)
        return (torch.abs(out_raw - t) * m).sum() / m.sum().clamp(min=1), out_raw

    @torch.no_grad()
    def fast_val():
        model.eval()
        err_sum, cnt = 0.0, 0
        for x_raw, y_raw in loader_select:
            x_raw = x_raw.cuda()
            xn = x_raw.clone()
            xn[..., 0] = (xn[..., 0] - mean_t) / std_t           # GPU-side per-node norm
            h = xn[:, :, :, [0, 1, 2]]
            f = torch.zeros(xn.shape[0], OL, N, 3, device='cuda')
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                out = model(history_data=h, future_data=f, batch_seen=0, epoch=0, train=False)
            if isinstance(out, dict):
                out = out['prediction']
            t = y_raw.cuda()[..., 0]
            mk = (t > 0)
            err_sum += float((torch.abs((out.float().squeeze(-1) * std_t + mean_t) - t) * mk).sum())
            cnt += int(mk.sum())
        model.train()
        return err_sum / max(cnt, 1)

    ckpt_dir = os.path.join(UC, 'EXPERIMENTS', 'UC-G3', 'checkpoints',
                            f'{dataset}_r3_4split_seed{seed}')
    os.makedirs(ckpt_dir, exist_ok=True)
    best_mae, best_ep, patience = float('inf'), 0, 0
    MIN_EP, PATIENCE, MIN_DELTA = 20, 3, 0.01
    t0 = time.time()
    for ep in range(1, max_epochs + 1):
        model.train()
        ep_loss, nb = 0.0, 0
        for bi, (x_raw, y_raw) in enumerate(loader_train):
            x_raw = x_raw.cuda()
            xn = x_raw.clone()
            xn[..., 0] = (xn[..., 0] - mean_t) / std_t           # GPU-side per-node norm
            h = xn[:, :, :, [0, 1, 2]]
            f = torch.zeros(xn.shape[0], OL, N, 3, device='cuda')
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                out = model(history_data=h, future_data=f, batch_seen=bi, epoch=ep, train=True)
            if isinstance(out, dict):
                out = out['prediction']
            loss, _ = raw_masked_mae(out.float(), y_raw.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += float(loss.item()); nb += 1
            if smoke and bi >= 1:
                break
        scheduler.step()
        if smoke:
            print(f'[SMOKE OK] loss={loss.item():.4f}', flush=True)
            return
        do_val = (ep == 1) or (ep < 30 and ep % 5 == 0) or (ep >= 30 and ep % 10 == 0)
        if do_val:
            v = fast_val()
            if v < best_mae - MIN_DELTA:
                best_mae, best_ep, patience = v, ep, 0
                torch.save({'model_state_dict': model.state_dict(), 'best_mae': v,
                            'best_epoch': ep, 'seed': seed, 'protocol': 'R3 4-split canonical'},
                           os.path.join(ckpt_dir, 'best.pt'))
            else:
                patience += 1
            print(f'ep {ep}/{max_epochs} loss={ep_loss/nb:.4f} val_rawMAE={v:.4f} '
                  f'best={best_mae:.4f}@ep{best_ep} pat={patience}/{PATIENCE} '
                  f'elapsed={(time.time()-t0)/60:.1f}min', flush=True)
            if ep >= MIN_EP and patience >= PATIENCE:
                print(f'Early stop @ep{ep}', flush=True)
                break
        else:
            print(f'ep {ep}/{max_epochs} loss={ep_loss/nb:.4f} (skip val) '
                  f'elapsed={(time.time()-t0)/60:.1f}min', flush=True)

    with open(os.path.join(ckpt_dir, 'train_meta.json'), 'w') as f:
        json.dump({'dataset': dataset, 'seed': seed, 'best_mae': best_mae,
                   'best_epoch': best_ep, 'max_epochs': max_epochs,
                   'protocol': 'R3: raw window + per-node norm in-model + raw masked MAE',
                   'boundaries': {'train_end': train_end, 'select_end': select_end,
                                  'calib_end': calib_end, 'test_end': T},
                   'purge': 'calib starts at select_end+23 (P1-6)'}, f, indent=2)
    print(f'Done. best={best_mae:.4f}@ep{best_ep} total={(time.time()-t0)/60:.1f}min', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, required=True, choices=['PEMS03', 'PEMS04', 'PEMS07'])
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--smoke', action='store_true', help='exit after a 2-batch forward+loss check')
    a = ap.parse_args()
    main(a.dataset, a.epochs, a.seed, a.smoke)
