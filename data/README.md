# Data

Raw PeMS data are **not** redistributed in this repository. Download them from the official
Caltrans Performance Measurement System (PeMS) source:

- PEMS03: `https://pems.dot.ca.gov/`
- PEMS04: `https://pems.dot.ca.gov/`
- PEMS07: `https://pems.dot.ca.gov/`
- PEMS08: `https://pems.dot.ca.gov/`

## Preprocessing

Each dataset directory must contain the raw traffic-flow `*.npz` and the distance file
(`distance.csv` / `PEMS03.csv` / `PEMS07.csv`). Then run:

```bash
python data/preprocess.py --dataset all
python data/generate_time_index.py --dataset all
```

### Conventions (frozen)

- Sampling rate: 5 minutes (288 steps per day).
- History / prediction window: 12 / 12 steps (1 hour each).
- Split (chronological, no shuffle): train 60% / val 20% / test 20%.
- Missing handling: `flow == 0` is treated as missing and linearly interpolated along time;
  fully-missing sensors stay zero.
- Normalization: per-feature Z-score with statistics estimated **on the training split only**;
  the scaler is applied before sliding-window segmentation.
- Adjacency: thresholded Gaussian kernel on the distance matrix
  `A[i,j] = exp(-d(i,j)^2 / std^2)` if `>= epsilon`, else 0 (Graph WaveNet convention).
- Valid target mask: a target is valid iff `y > 0` and finite; coverage/WCE/GCD are computed
  only on valid targets.

### Outputs

- `data_preprocessing/processed/{name}_samples.npz` — sliding-window samples
  (`x_train/y_train/x_val/y_val/x_test/y_test`).
- `data_preprocessing/processed/{name}_meta.pkl` — adjacency matrix + scaler statistics.
- `data_preprocessing/processed/{name}_timeidx.npz` — time-of-day / day-of-week indices.
