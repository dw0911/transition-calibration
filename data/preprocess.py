# -*- coding: utf-8 -*-
"""
PEMS dataset preprocessing module
=================================
Functions:
1. Load npz traffic-flow data + distance.csv distance data
2. Data cleaning (zero / missing-value handling)
3. Z-score normalization (using training-set statistics only)
4. Sliding-window segmentation (history 12 steps -> future 12 steps)
5. Dataset split (train:val:test = 6:2:2, in chronological order)
6. Adjacency matrix construction (thresholded Gaussian kernel, after Graph WaveNet)

Command-line usage:
    python preprocess.py --dataset PEMS04
    python preprocess.py --dataset PEMS08
    python preprocess.py --dataset all

Module usage:
    from preprocess import load_dataset, get_adjacency_matrix
    data = load_dataset('PEMS04', data_dir=...)
"""

import os
import argparse
import numpy as np
import pandas as pd
import pickle

# ---------------------------------------------------------------------------
# Dataset configuration (consistent with the measured datasets)
# ---------------------------------------------------------------------------
DATASET_CONFIG = {
    'PEMS04': {
        'npz': 'pems04.npz',
        'dist': 'distance.csv',
        'num_nodes': 307,
        'input_dim': 3,          # flow / occupancy / speed
        'target_dim': 0,         # evaluate flow only
    },
    'PEMS08': {
        'npz': 'pems08.npz',
        'dist': 'distance.csv',
        'num_nodes': 170,
        'input_dim': 3,
        'target_dim': 0,
    },
    'PEMS03': {
        'npz': 'PEMS03.npz',
        'dist': 'PEMS03.csv',
        'num_nodes': 358,
        'input_dim': 1,
        'target_dim': 0,
    },
    'PEMS07': {
        'npz': 'PEMS07.npz',
        'dist': 'PEMS07.csv',
        'num_nodes': 883,
        'input_dim': 1,
        'target_dim': 0,
    },
}

# Default hyperparameters (aligned with literature standards)
HIS_STEPS = 12        # history window (1 hour)
PRED_STEPS = 12       # prediction window (1 hour)
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
# test set = 1 - 0.6 - 0.2 = 0.2


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
def load_raw(npz_path, num_nodes, input_dim):
    """Load raw npz data, return a (T, N, C) array and validate its shape."""
    d = np.load(npz_path)
    key = d.files[0]
    data = d[key].astype(np.float32)
    T, N, C = data.shape
    assert N == num_nodes, f'node count mismatch: expected {num_nodes}, got {N}'
    assert C == input_dim, f'feature dim mismatch: expected {input_dim}, got {C}'
    print(f'  raw data: T={T} ({T/288:.0f} days), N={N}, C={C}')
    return data


def handle_missing(data):
    """
    Zero (missing) handling:
    Treat 0 as missing and fill it by linear interpolation using the
    non-zero mean of the same sensor within the same time window; if an
    entire segment is zero it stays zero (a fully-missing sensor).
    Returns (filled data, missing-ratio statistic).
    """
    flow = data[..., 0]
    missing_mask = (flow == 0)
    missing_ratio = missing_mask.mean()

    filled = data.copy()
    # Linear interpolation along the time axis (per node, per feature)
    for c in range(data.shape[2]):
        ch = data[..., c]
        df = pd.DataFrame(ch)
        df = df.replace(0, np.nan).interpolate(
            method='linear', limit_direction='both', axis=0)
        df = df.fillna(0)
        filled[..., c] = df.values.astype(np.float32)

    print(f'  missing rate (zeros): {missing_ratio*100:.2f}%  -> linearly interpolated')
    return filled, missing_ratio


# ---------------------------------------------------------------------------
# 2. Normalization (Z-score, training set only)
# ---------------------------------------------------------------------------
class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


def fit_scaler(train_data):
    """Compute per-feature mean/std on the training set (stats over time+nodes)."""
    mean = train_data.mean(axis=(0, 1), keepdims=True)   # (1,1,C)
    std = train_data.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-5, 1.0, std)                 # avoid division by zero
    return StandardScaler(mean, std)


# ---------------------------------------------------------------------------
# 3. Sliding-window segmentation
# ---------------------------------------------------------------------------
def sliding_window(data, his_steps, pred_steps):
    """
    Split (T, N, C) into samples:
        X: (num_samples, his_steps, N, C)
        Y: (num_samples, pred_steps, N, C)
    Uses sliding stride = 1.
    """
    T = data.shape[0]
    num_samples = T - his_steps - pred_steps + 1
    X, Y = [], []
    for i in range(num_samples):
        X.append(data[i: i + his_steps])
        Y.append(data[i + his_steps: i + his_steps + pred_steps])
    X = np.stack(X).astype(np.float32)
    Y = np.stack(Y).astype(np.float32)
    return X, Y


# ---------------------------------------------------------------------------
# 4. Adjacency matrix construction (thresholded Gaussian kernel, Graph WaveNet standard)
# ---------------------------------------------------------------------------
def build_adjacency(dist_path, num_nodes, sigma2=0.1, epsilon=0.5):
    """
    Build a symmetric adjacency matrix from distance.csv:
        A[i,j] = exp(-d(i,j)^2 / sigma^2)  if >= epsilon, else 0
    Return a (N, N) numpy array.

    Note: the from/to columns in distance.csv use the raw sensor ids,
    which must be mapped to contiguous 0..N-1 indices.
    """
    df = pd.read_csv(dist_path)
    # Accommodate varying distance column names across datasets:
    # PEMS04/07/08 use 'cost', PEMS03 uses 'distance'
    dist_col = 'cost' if 'cost' in df.columns else 'distance'

    sensor_ids = sorted(set(df['from'].unique()) | set(df['to'].unique()))
    assert len(sensor_ids) == num_nodes, \
        f'distance file sensor count ({len(sensor_ids)}) != data node count ({num_nodes})'
    id_to_idx = {sid: i for i, sid in enumerate(sensor_ids)}

    distances = df[dist_col].values
    std = distances.std()
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for _, row in df.iterrows():
        i = id_to_idx[row['from']]
        j = id_to_idx[row['to']]
        w = np.exp(-(row[dist_col] ** 2) / (std ** 2))
        if w >= epsilon:
            A[i, j] = w

    sparsity = (A > 0).mean()
    print(f'  adjacency matrix: {A.shape}, non-zero ratio={sparsity*100:.2f}%')
    return A


# ---------------------------------------------------------------------------
# 5. Main pipeline
# ---------------------------------------------------------------------------
def preprocess_dataset(name, raw_dir, out_dir,
                       his_steps=HIS_STEPS, pred_steps=PRED_STEPS,
                       train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO,
                       fill_missing=True, save=True):
    """
    Run the full preprocessing for a single dataset, return a dict and
    optionally save it as npz/pkl.
    """
    cfg = DATASET_CONFIG[name]
    print(f'\n{"="*60}\nPreprocessing {name}\n{"="*60}')

    npz_path = os.path.join(raw_dir, cfg['npz'])
    dist_path = os.path.join(raw_dir, cfg['dist'])

    # 1) load
    data = load_raw(npz_path, cfg['num_nodes'], cfg['input_dim'])

    # 2) missing-value handling
    if fill_missing:
        data, missing_ratio = handle_missing(data)
    else:
        missing_ratio = (data[..., 0] == 0).mean()

    # 3) dataset split (by time, no shuffle) -- split before normalize to avoid leakage
    T = data.shape[0]
    n_train = int(T * train_ratio)
    n_val = int(T * val_ratio)
    train_data = data[:n_train]
    val_data = data[n_train: n_train + n_val]
    test_data = data[n_train + n_val:]
    print(f'  split: train={n_train}, val={n_val}, test={T-n_train-n_val}')

    # 4) normalize (training-set statistics only)
    scaler = fit_scaler(train_data)
    train_n = scaler.transform(train_data)
    val_n = scaler.transform(val_data)
    test_n = scaler.transform(test_data)

    # 5) sliding window
    x_train, y_train = sliding_window(train_n, his_steps, pred_steps)
    x_val, y_val = sliding_window(val_n, his_steps, pred_steps)
    x_test, y_test = sliding_window(test_n, his_steps, pred_steps)
    print(f'  samples: train={x_train.shape[0]}, val={x_val.shape[0]}, test={x_test.shape[0]}')

    # 6) adjacency matrix
    adj = build_adjacency(dist_path, cfg['num_nodes'])

    result = {
        'name': name,
        'x_train': x_train, 'y_train': y_train,
        'x_val': x_val, 'y_val': y_val,
        'x_test': x_test, 'y_test': y_test,
        'adj': adj,
        'scaler_mean': scaler.mean,
        'scaler_std': scaler.std,
        'target_dim': cfg['target_dim'],
        'his_steps': his_steps,
        'pred_steps': pred_steps,
        'missing_ratio': missing_ratio,
    }

    if save:
        os.makedirs(out_dir, exist_ok=True)
        # sample data saved compressed as npz
        np.savez_compressed(
            os.path.join(out_dir, f'{name}_samples.npz'),
            x_train=x_train, y_train=y_train,
            x_val=x_val, y_val=y_val,
            x_test=x_test, y_test=y_test)
        # adjacency matrix + scaler saved as pkl
        with open(os.path.join(out_dir, f'{name}_meta.pkl'), 'wb') as f:
            pickle.dump({
                'adj': adj,
                'scaler_mean': scaler.mean,
                'scaler_std': scaler.std,
                'target_dim': cfg['target_dim'],
                'his_steps': his_steps,
                'pred_steps': pred_steps,
            }, f)
        print(f'  saved -> {out_dir}/{name}_samples.npz + {name}_meta.pkl')

    return result


def main():
    parser = argparse.ArgumentParser(description='PEMS data preprocessing')
    parser.add_argument('--dataset', type=str, default='PEMS04',
                        help='PEMS04 / PEMS08 / PEMS03 / PEMS07 / all')
    parser.add_argument('--raw_root', type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'PeMs'),
                        help='root directory of raw data')
    parser.add_argument('--out_dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data_preprocessing', 'processed'),
                        help='output directory')
    parser.add_argument('--his', type=int, default=HIS_STEPS)
    parser.add_argument('--pred', type=int, default=PRED_STEPS)
    args = parser.parse_args()

    if args.dataset.lower() == 'all':
        targets = ['PEMS04', 'PEMS08', 'PEMS03', 'PEMS07']
    else:
        targets = [args.dataset]

    summary = []
    for name in targets:
        raw_dir = os.path.join(args.raw_root, name)
        res = preprocess_dataset(name, raw_dir, args.out_dir,
                                 his_steps=args.his, pred_steps=args.pred)
        summary.append(res)

    # summary report
    print(f'\n{"="*60}\nPreprocessing summary\n{"="*60}')
    print(f'{"dataset":<10}{"train":>10}{"val":>10}{"test":>10}{"miss%":>10}')
    for r in summary:
        print(f'{r["name"]:<10}{r["x_train"].shape[0]:>10}'
              f'{r["x_val"].shape[0]:>10}{r["x_test"].shape[0]:>10}'
              f'{r["missing_ratio"]*100:>9.2f}%')


if __name__ == '__main__':
    main()
