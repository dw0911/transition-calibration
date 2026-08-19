# -*- coding: utf-8 -*-
"""UC-G3b: STID x PEMS04 training (legacy framework, four-split, fast_val + early stop).

STID is a pure MLP without attention, so training is fast (~30s/epoch). The legacy npz is
pre-split 60/20/20; the four-split uses the second half of val as calib (training-time select
uses the first half of val).

Usage:
  python train_stid_fast.py --seed 42
  python train_stid_fast.py --seed 123
  python train_stid_fast.py --seed 456
"""
import os, sys, json, time, argparse
import numpy as np
import torch

REPRO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINES = os.environ.get('BASELINES_ROOT', '')
PROCDIR = os.environ.get('REPRO_PROCDIR',
                         os.path.join(REPRO_ROOT, 'data_preprocessing', 'processed'))
UC = REPRO_ROOT
sys.path.insert(0, os.path.join(BASELINES, 'common'))
sys.path.insert(0, os.path.join(BASELINES, 'STID'))

import pickle
from model import STID


def main(seed=42, max_epochs=100):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    print(f'[STID-fast] seed={seed}, max_epochs={max_epochs}', flush=True)

    d = np.load(os.path.join(PROCDIR, 'PEMS04_samples.npz'))
    m = pickle.load(open(os.path.join(PROCDIR, 'PEMS04_meta.pkl'), 'rb'))
    t = np.load(os.path.join(PROCDIR, 'PEMS04_timeidx.npz'))
    mean, std = float(m['scaler_mean'].reshape(-1)[0]), float(m['scaler_std'].reshape(-1)[0])
    N = d['x_train'].shape[2]

    # four-split: val second half as calib (training-time select uses the val first half)
    n_val = d['x_val'].shape[0]
    n_select = n_val // 2
    x_select, y_select = d['x_val'][:n_select], d['y_val'][:n_select]
    tod_select, dow_select = t['val_tod'][:n_select], t['val_dow'][:n_select]

    model = STID(num_nodes=N, input_dim=3, his_steps=12, pred_steps=12).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=0)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[20, 25], gamma=0.1)

    # fast_val: norm space masked MAE on select set
    @torch.no_grad()
    def fast_val():
        model.eval()
        err_sum, cnt = 0.0, 0
        for bi in range(0, len(x_select), 128):
            xt = torch.tensor(x_select[bi:bi+128], dtype=torch.float32, device='cuda')
            tod = torch.tensor(tod_select[bi:bi+128], dtype=torch.long, device='cuda')
            dow = torch.tensor(dow_select[bi:bi+128], dtype=torch.long, device='cuda')
            out = model(xt, tod, dow)
            if out.dim() == 3: out = out.unsqueeze(-1)
            t_norm = torch.tensor(y_select[bi:bi+128], dtype=torch.float32, device='cuda')[..., 0:1]
            msk = (t_norm != 0)
            err_sum += float((torch.abs(out - t_norm) * msk).sum().item())
            cnt += int(msk.sum().item())
        model.train()
        return err_sum / max(cnt, 1)

    ckpt_dir = os.path.join(UC, 'EXPERIMENTS', 'UC-G3', 'checkpoints', f'PEMS04_stid_fast_4split_seed{seed}')
    os.makedirs(ckpt_dir, exist_ok=True)
    best_mae, best_ep, patience = float('inf'), 0, 0
    MIN_EP, PATIENCE, MIN_DELTA = 20, 3, 0.01
    t0 = time.time()

    for ep in range(1, max_epochs + 1):
        model.train()
        # shuffle train
        perm = np.random.permutation(len(d['x_train']))
        ep_loss, ep_cnt = 0.0, 0
        for bi in range(0, len(perm), 128):
            idx = perm[bi:bi+128]
            xt = torch.tensor(d['x_train'][idx], dtype=torch.float32, device='cuda')
            yt = torch.tensor(d['y_train'][idx], dtype=torch.float32, device='cuda')
            tod = torch.tensor(t['train_tod'][idx], dtype=torch.long, device='cuda')
            dow = torch.tensor(t['train_dow'][idx], dtype=torch.long, device='cuda')
            out = model(xt, tod, dow)
            if out.dim() == 3: out = out.unsqueeze(-1)
            t_norm = yt[..., 0:1]
            msk = (t_norm != 0)
            loss = (torch.abs(out - t_norm) * msk).sum() / msk.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss.item()); ep_cnt += 1
        sched.step()

        do_val = (ep < 30 and ep % 5 == 0) or (ep >= 30 and ep % 10 == 0) or (ep == 1)
        if do_val:
            val_mae = fast_val()
            if val_mae < best_mae - MIN_DELTA:
                best_mae, best_ep, patience = val_mae, ep, 0
                torch.save({'model_state_dict': model.state_dict(), 'best_mae': best_mae,
                            'best_epoch': best_ep, 'seed': seed},
                           os.path.join(ckpt_dir, 'best.pt'))
            else:
                patience += 1
            el = time.time() - t0
            print(f'ep {ep}/{max_epochs} loss={ep_loss/ep_cnt:.4f} fast_val={val_mae:.4f} '
                  f'best={best_mae:.4f}@ep{best_ep} patience={patience}/{PATIENCE} '
                  f'elapsed={el/60:.1f}min', flush=True)
            if ep >= MIN_EP and patience >= PATIENCE:
                print(f'Early stop ep{ep}', flush=True); break
        else:
            el = time.time() - t0
            print(f'ep {ep}/{max_epochs} loss={ep_loss/ep_cnt:.4f} (skip val) elapsed={el/60:.1f}min', flush=True)

    print(f'\nDone. best_mae={best_mae:.4f}@ep{best_ep}, total={time.time()-t0:.0f}s', flush=True)
    with open(os.path.join(ckpt_dir, 'train_meta.json'), 'w') as f:
        json.dump({'dataset': 'PEMS04', 'model': 'STID', 'best_mae': best_mae,
                   'best_epoch': best_ep, 'seed': seed,
                   'protocol': '4-split fast_val norm-space MAE, legacy framework'}, f, indent=2)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=100)
    a = ap.parse_args()
    main(a.seed, a.epochs)
