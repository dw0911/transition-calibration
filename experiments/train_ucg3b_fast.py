# -*- coding: utf-8 -*-
"""UC-G3b: STAEformer x PEMS07 training (Fast Validation + four-split + early stopping).

Protocol upgrade (Plan A'):
  - four-split: train 60% / select 10% / calib 10% / test 20% (chronological; fixes the val dual-use issue)
  - Fast Validation: only compute norm-space masked MAE (used for checkpoint selection), skips RMSE/MAPE/inverse_transform
  - val frequency: every 5 epochs (epoch<30) / every 10 epochs (epoch>=30)
  - early stopping: min_epochs=20, patience=3 (x5ep=15ep), min_delta=0.01, max=100
  - batch 16 (official PEMS07.py), AMP autocast, seed 42

Usage:
  python train_ucg3b_fast.py --dataset PEMS07 --epochs 100
"""
import os, sys, json, time, argparse

REPRO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASICTS = os.environ.get('BASICTS_ROOT', '')
UC = REPRO_ROOT
sys.path.insert(0, BASICTS)
sys.path.insert(0, os.path.join(BASICTS, 'baselines', 'STAEformer'))
os.chdir(BASICTS)

import random
import numpy as np
import torch


def main(dataset, max_epochs, seed=42, gpu='0'):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    print(f'[UC-G3b-fast] dataset={dataset}, max_epochs={max_epochs}, seed={seed}', flush=True)

    from baselines.STAEformer.arch.staeformer_arch import STAEformer
    from basicts.data import TimeSeriesForecastingDataset
    from basicts.scaler import ZScoreScaler

    # data (four-split: use the original 60/20/20 train+val; first half of val as select, second half as calib)
    # BasicTS defaults to 60/20/20; we manually take the first 50% of val as select, leaving the last 50% as calib
    ds_train = TimeSeriesForecastingDataset(dataset_name=dataset, train_val_test_ratio=[0.6,0.2,0.2],
                                            input_len=12, output_len=12, mode='train')
    ds_val = TimeSeriesForecastingDataset(dataset_name=dataset, train_val_test_ratio=[0.6,0.2,0.2],
                                          input_len=12, output_len=12, mode='valid')
    # four-split: split val in half (select + calib)
    n_val = len(ds_val)
    n_select = n_val // 2
    ds_select = torch.utils.data.Subset(ds_val, range(n_select))
    # calib/test conformal evaluation is done separately after training (ucg3b_eval.py, four-split version)

    loader_train = torch.utils.data.DataLoader(ds_train, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    loader_select = torch.utils.data.DataLoader(ds_select, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    scaler = ZScoreScaler(dataset, 0.6, True, True)
    N = {'PEMS03': 358, 'PEMS04': 307, 'PEMS07': 883}[dataset]

    model = STAEformer(num_nodes=N, in_steps=12, out_steps=12, steps_per_day=288,
                       input_dim=3, output_dim=1, input_embedding_dim=24, tod_embedding_dim=24,
                       dow_embedding_dim=24, spatial_embedding_dim=0, adaptive_embedding_dim=80,
                       feed_forward_dim=256, num_heads=4, num_layers=3, dropout=0.1, use_mixed_proj=True).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,25], gamma=0.1)
    USE_AMP = False  # the PEMS03/07 custom loop caused AMP nan, so it is disabled (N=358/883, batch16 fits in memory)

    # Fast Validation: norm space masked MAE
    @torch.no_grad()
    def fast_val():
        model.eval()
        err_sum, cnt = 0.0, 0
        for data in loader_select:
            history_data = data['inputs'].cuda()
            future_data = data['target'].cuda()
            h = history_data[:, :, :, [0,1,2]]
            f = future_data[:, :, :, [0,1,2]]
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                out = model(history_data=h, future_data=f, batch_seen=0, epoch=0, train=False)
            if isinstance(out, dict): out = out['prediction']
            out = out.float()
            t_norm = future_data[:,:,:,0:1]
            m = (t_norm != 0)
            err_sum += float((torch.abs(out - t_norm) * m).sum().item())
            cnt += int(m.sum().item())
        model.train()
        return err_sum / max(cnt, 1)

    ckpt_dir = os.path.join(UC, 'EXPERIMENTS', 'UC-G3', 'checkpoints',
                            f'{dataset}_fast_4split_seed{seed}')
    os.makedirs(ckpt_dir, exist_ok=True)
    best_mae, best_ep, patience_cnt = float('inf'), 0, 0
    MIN_EPOCHS, PATIENCE, MIN_DELTA = 20, 3, 0.01
    t0 = time.time()

    for ep in range(1, max_epochs + 1):
        model.train()
        ep_loss, ep_cnt = 0.0, 0
        for bi, data in enumerate(loader_train):
            history_data = data['inputs'].cuda()
            future_data = data['target'].cuda()
            h = history_data[:, :, :, [0,1,2]]
            f = future_data[:, :, :, [0,1,2]]
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                out = model(history_data=h, future_data=f, batch_seen=bi, epoch=ep, train=True)
            if isinstance(out, dict): out = out['prediction']
            out = out.float()
            t_norm = future_data[:,:,:,0:1]
            m = (t_norm != 0)
            loss = (torch.abs(out - t_norm) * m).sum() / m.sum().clamp(min=1)
            optimizer.zero_grad()
            if USE_AMP:
                amp_scaler.scale(loss).backward()
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                loss.backward()
                optimizer.step()
            ep_loss += float(loss.item()); ep_cnt += 1
        scheduler.step()

        # val frequency: ep<30 per5ep, ep>=30 per10ep
        do_val = (ep < 30 and ep % 5 == 0) or (ep >= 30 and ep % 10 == 0) or (ep == 1)
        if do_val:
            val_mae = fast_val()
            improved = val_mae < best_mae - MIN_DELTA
            if improved:
                best_mae, best_ep, patience_cnt = val_mae, ep, 0
                torch.save({'model_state_dict': model.state_dict(), 'best_mae': best_mae,
                            'best_epoch': best_ep, 'seed': seed, 'fast_val': True,
                            'protocol': '4-split train/select/calib/test'},
                           os.path.join(ckpt_dir, 'best.pt'))
            else:
                patience_cnt += 1
            el = time.time() - t0
            print(f'ep {ep}/{max_epochs} loss={ep_loss/ep_cnt:.4f} fast_val_mae={val_mae:.4f} '
                  f'best={best_mae:.4f}@ep{best_ep} patience={patience_cnt}/{PATIENCE} '
                  f'elapsed={el/60:.1f}min', flush=True)
            if ep >= MIN_EPOCHS and patience_cnt >= PATIENCE:
                print(f'Early stop: ep{ep} >= {MIN_EPOCHS} & patience {patience_cnt} >= {PATIENCE}', flush=True)
                break
        else:
            el = time.time() - t0
            print(f'ep {ep}/{max_epochs} loss={ep_loss/ep_cnt:.4f} (skip val) '
                  f'elapsed={el/60:.1f}min', flush=True)

    print(f'\nDone. best_mae={best_mae:.4f}@ep{best_ep}, total={time.time()-t0:.0f}s', flush=True)
    # save training metadata
    with open(os.path.join(ckpt_dir, 'train_meta.json'), 'w') as f:
        json.dump({'dataset': dataset, 'best_mae': best_mae, 'best_epoch': best_ep,
                   'max_epochs': max_epochs, 'seed': seed,
                   'protocol': '4-split: train60/select10/calib10/test20, fast_val norm-space MAE',
                   'early_stop': {'min_epochs': MIN_EPOCHS, 'patience': PATIENCE,
                                   'min_delta': MIN_DELTA}}, f, indent=2)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, required=True, choices=['PEMS03', 'PEMS04', 'PEMS07'])
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    main(a.dataset, a.epochs, a.seed)
