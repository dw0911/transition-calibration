# -*- coding: utf-8 -*-
"""
Generate time indices (time_of_day / day_of_week) for the preprocessed samples.
========================================================================
Principle:
  - each sample x[i] corresponds to an absolute time step abs_t of its "prediction start"
  - train/val/test are sliced from the full series in chronological order, so abs_t can be recovered
  - abs_t -> time_of_day = abs_t % 288 (one day = 288 five-minute steps)
          -> day_of_week = (abs_t // 288) % 7
  - the origin date is unknown; we use the literature-common assumption that the series starts at
    "Monday 00:00" (virtual origin). The model only needs to learn the relative daily/weekly
    periodicity, so the virtual origin does not affect periodic modeling.

Output:
  processed/{name}_timeidx.npz containing train/val/test tod/dow arrays.
"""
import os
import argparse
import numpy as np

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'data_preprocessing', 'processed')

DATASET_CONFIG = {
    'PEMS04': {'T': 16992, 'his': 12, 'pred': 12},
    'PEMS08': {'T': 17856, 'his': 12, 'pred': 12},
    'PEMS03': {'T': 26208, 'his': 12, 'pred': 12},
    'PEMS07': {'T': 28224, 'his': 12, 'pred': 12},
}
STEPS_PER_DAY = 288
DAYS_PER_WEEK = 7


def compute_abs_starts(T, his, pred, train_ratio=0.6, val_ratio=0.2):
    """
    Return arrays of absolute time steps for the prediction starts of train/val/test samples.
    Sample i has its prediction start at i + his (sub-series local coordinates);
    adding the sub-series offset within the full series yields the absolute time step.
    """
    n_train = int(T * train_ratio)
    n_val = int(T * val_ratio)

    def local_starts(seg_len):
        num = seg_len - his - pred + 1
        # local prediction-start indices (sub-series coordinates): i + his
        return np.arange(his, his + num)

    train_abs = local_starts(n_train) + 0                 # train offset = 0
    val_abs = local_starts(n_val) + n_train               # val offset = n_train
    test_abs = local_starts(T - n_train - n_val) + n_train + n_val
    return train_abs, val_abs, test_abs


def abs_to_tod_dow(abs_steps):
    tod = abs_steps % STEPS_PER_DAY
    dow = (abs_steps // STEPS_PER_DAY) % DAYS_PER_WEEK
    return tod.astype(np.int64), dow.astype(np.int64)


def generate(name, out_dir=PROCESSED_DIR):
    cfg = DATASET_CONFIG[name]
    T, his, pred = cfg['T'], cfg['his'], cfg['pred']
    tr_abs, va_abs, te_abs = compute_abs_starts(T, his, pred)

    tr_tod, tr_dow = abs_to_tod_dow(tr_abs)
    va_tod, va_dow = abs_to_tod_dow(va_abs)
    te_tod, te_dow = abs_to_tod_dow(te_abs)

    out_path = os.path.join(out_dir, f'{name}_timeidx.npz')
    np.savez_compressed(out_path,
                        train_tod=tr_tod, train_dow=tr_dow,
                        val_tod=va_tod, val_dow=va_dow,
                        test_tod=te_tod, test_dow=te_dow)

    # alignment check against the existing samples
    d = np.load(os.path.join(out_dir, f'{name}_samples.npz'))
    assert len(tr_tod) == d['x_train'].shape[0], \
        f'{name} train time indices ({len(tr_tod)}) != samples ({d["x_train"].shape[0]})'
    assert len(va_tod) == d['x_val'].shape[0]
    assert len(te_tod) == d['x_test'].shape[0]

    print(f'{name}: time indices generated and aligned with the samples')
    print(f'  train tod range[{tr_tod.min()},{tr_tod.max()}] dow range[{tr_dow.min()},{tr_dow.max()}]')
    print(f'  sample counts train={len(tr_tod)} val={len(va_tod)} test={len(te_tod)}')
    print(f'  saved -> {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='PEMS04')
    args = parser.parse_args()
    targets = ['PEMS04', 'PEMS08', 'PEMS03', 'PEMS07'] if args.dataset.lower() == 'all' \
        else [args.dataset]
    for n in targets:
        generate(n)
