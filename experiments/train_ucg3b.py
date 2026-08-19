# -*- coding: utf-8 -*-
"""UC-G3b: STAEformer x PEMS03/07 training (scratch, verify dataset generalization). 

Protocol (aligned with the official STAEformer config, PEMS03.py/PEMS07.py):
  - scratch training (no PEMS03/07 pretrained ckpt)
  - Adam(1e-3, 3e-4) / MultiStepLR[20,25] / batch16 / 100ep / seed 42
  - SimpleTimeSeriesForecastingRunner (no AMP; the official PEMS03/07 config has no AMP)

Usage:
  python train_ucg3b.py --dataset PEMS03 --epochs 100
  python train_ucg3b.py --dataset PEMS07 --epochs 100
"""
import os, sys, argparse

REPRO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASICTS = os.environ.get('BASICTS_ROOT', '')
COURT = os.environ.get('COURT_ROOT', '')
UC = REPRO_ROOT
sys.path.insert(0, BASICTS)
sys.path.insert(0, os.path.join(BASICTS, 'baselines', 'STAEformer'))
os.chdir(BASICTS)

import random
import numpy as np
import torch


def main(dataset, epochs, seed=42, gpu='0'):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    print(f'[UC-G3b] dataset={dataset}, epochs={epochs}, seed={seed}', flush=True)

    cfg_mod = __import__(f'baselines.STAEformer.{dataset}', fromlist=['CFG'])
    cfg = cfg_mod.CFG
    cfg.TRAIN.NUM_EPOCHS = epochs
    cfg.TRAIN.CKPT_SAVE_DIR = os.path.join(
        UC, 'EXPERIMENTS', 'UC-G3', 'checkpoints',
        f'{dataset}_{epochs}_12_12_seed{seed}')
    # load acceleration (PEMS07 N=883, officialconfigno num_workers causes GPU idle wait)
    cfg.TRAIN.DATA.NUM_WORKERS = 4
    cfg.TRAIN.DATA.PIN_MEMORY = True
    cfg.VAL.DATA.NUM_WORKERS = 4
    cfg.VAL.DATA.PIN_MEMORY = True

    from easytorch import launch_training
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    launch_training(cfg, devices=gpu)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, required=True, choices=['PEMS03', 'PEMS07'])
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--gpu', type=str, default='0')
    a = ap.parse_args()
    main(a.dataset, a.epochs, a.seed, a.gpu)
