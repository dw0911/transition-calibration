# -*- coding: utf-8 -*-
"""R3 equivalence acceptance (Gate): the train_r3 transform pipeline vs the official protocol
(ucg1/c2_sweep inference chain).

Principle: the Base-0 official pretrained ckpt is trained in norm space. Using train_r3's
transform (per-node (x-mean)/std -> model -> out*std+mean), infer the first 128 test windows
and compare window-by-window with ucg1.backbone_infer('staeformer') (official protocol; the
a_base0 era already had official-log diff <= 0.0003). If max|diff| < 1e-4 -> the transform
pipeline is correct and R3 retraining is cleared.

Usage: python r3_equiv.py
"""
import os
import sys
import json

REPRO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COURT = os.environ.get('COURT_ROOT', '')
BASICTS = os.environ.get('BASICTS_ROOT', '')
UC = REPRO_ROOT

if COURT:
    sys.path.insert(0, os.path.join(COURT, 'EXPERIMENTS', 'C_headroom'))
if BASICTS:
    sys.path.insert(0, BASICTS)
    sys.path.insert(0, os.path.join(BASICTS, 'baselines', 'STAEformer'))
    os.chdir(BASICTS)

import numpy as np                                  # noqa: E402
import torch                                        # noqa: E402
import ucg1                                         # noqa: E402

PRETRAINED = os.path.join(BASICTS, 'checkpoints', 'STAEformer', 'PEMS04_100_12_12',
                          'd5e02b2b1247a62beb34f5f18cc70769', 'STAEformer_best_val_MAE.pt')


def main():
    nb = 128
    print('[equiv] official protocol pred (ucg1/c2_sweep)...', flush=True)
    pt_off, tt_off = ucg1.backbone_infer('staeformer', 'test')
    pt_off = pt_off[:nb]
    print(f'  official: {pt_off.shape}, raw MAE(first128)='
          f'{float(np.abs(pt_off - tt_off[:nb]).mean()):.4f}', flush=True)

    # train_r3 same-style transform inference
    desc = json.load(open(os.path.join(BASICTS, 'datasets', 'PEMS04', 'desc.json')))
    shape = tuple(desc['shape'])
    full = np.memmap(os.path.join(BASICTS, 'datasets', 'PEMS04', 'data.dat'),
                     dtype='float32', mode='r', shape=shape)
    # same-source stats as the official protocol (c2_sweep convention: train_len = T - 2*int(0.2T)) + autocast
    train_end = shape[0] - 2 * int(shape[0] * 0.2)
    mean = full[:train_end, :, 0].mean(axis=0)
    std = full[:train_end, :, 0].std(axis=0)
    std[std == 0] = 1.0
    mean_t = torch.tensor(mean, dtype=torch.float32, device='cuda')
    std_t = torch.tensor(std, dtype=torch.float32, device='cuda')

    from baselines.STAEformer.arch.staeformer_arch import STAEformer
    model = STAEformer(num_nodes=307, in_steps=12, out_steps=12, steps_per_day=288,
                       input_dim=3, output_dim=1, input_embedding_dim=24, tod_embedding_dim=24,
                       dow_embedding_dim=24, spatial_embedding_dim=0, adaptive_embedding_dim=80,
                       feed_forward_dim=256, num_heads=4, num_layers=3, dropout=0.1,
                       use_mixed_proj=True).cuda()
    sd = torch.load(PRETRAINED, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    model.load_state_dict(sd)
    model.eval()

    # test-split window start: window 0 start of official infer_split
    import c2_sweep as base
    TRAIN_LEN, VAL_END = base.get_split_boundaries(base.T)
    g0 = VAL_END
    ar = np.arange(12)
    gs = g0 + np.arange(nb)
    x = full[gs[:, None] + ar].copy()                          # (nb,12,N,3) raw
    x[..., 0] = (x[..., 0] - mean) / std                       # train_r3 same-style per-node norm
    xt = torch.tensor(x, dtype=torch.float32, device='cuda')
    fut = torch.zeros(nb, 12, 307, 3, device='cuda')
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):
        out = model(history_data=xt, future_data=fut, batch_seen=0, epoch=1, train=False)
    if isinstance(out, dict):
        out = out['prediction']
    pt_mine = (out.squeeze(-1).float() * std_t + mean_t).cpu().numpy()

    diff = float(np.abs(pt_mine - pt_off).max())
    rel = diff / float(np.abs(pt_off).mean())
    print(f'  mine:     {pt_mine.shape}', flush=True)
    print(f'\nmax|diff| = {diff:.3e}   rel ~ {rel:.3e}', flush=True)
    ok = diff < 1e-4
    print(f'GATE: {"PASS - R3 retraining cleared" if ok else "FAIL - transform pipeline still has issues"}', flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
