# Model checkpoints

**No model checkpoints are distributed with this repository.** The trained point-forecast
weights are either produced by the training scripts in this repository or downloaded from the
official BasicTS release assets. The calibration method is a *post-hoc* wrapper: it never
retrains the point-forecast model, so any reasonable traffic-forecasting checkpoint can be
plugged in.

## How to obtain a checkpoint

### Option A — train with this repository (reproduces the exact paper numbers)

The main tables use the STAEformer checkpoints trained under the paper's four-split protocol
(`train/select/calib/test = 60/10/10/20`). Train them with the provided script (requires a GPU,
the [BasicTS](https://github.com/GestaltCogTeam/BasicTS) framework, and the PEMS data):

```bash
export BASICTS_ROOT=/path/to/BasicTS
export REPRO_CKPT_DIR=/path/to/reproducible_uc/checkpoints

# Table 4 (PEMS04): three seeds
python experiments/train_staeformer.py --dataset PEMS04 --epochs 100 --seed 42
python experiments/train_staeformer.py --dataset PEMS04 --epochs 100 --seed 123
python experiments/train_staeformer.py --dataset PEMS04 --epochs 100 --seed 456

# Table 4 (PEMS03): one seed
python experiments/train_staeformer.py --dataset PEMS03 --epochs 100 --seed 42
```

Each run writes `checkpoints/<dataset>_r3_4split_seed<seed>/best.pt` plus a `train_meta.json`
provenance record. Training is ~2-4 hours per run on a single modern GPU.

### Option B — use an official pretrained checkpoint

If you only want to sanity-check the calibration pipeline (not match the paper's exact
numbers), the official BasicTS pretrained STAEformer checkpoint for PEMS04 can be downloaded
from the BasicTS release assets and placed at:

```
checkpoints/PEMS04_r3_4split_seed42/best.pt
```

> Note: the official weights follow the standard 60/20/20 split, whereas the paper's tables use
> the four-split protocol. Coverage/WCE/GCD will be close but not bit-identical to the paper.

## Layout expected by the evaluation scripts

```
reproducible_uc/
└── checkpoints/
    ├── PEMS04_r3_4split_seed42/best.pt
    ├── PEMS04_r3_4split_seed123/best.pt
    ├── PEMS04_r3_4split_seed456/best.pt
    └── PEMS03_r3_4split_seed42/best.pt
```

`best.pt` may be a plain state dict or wrapped as `{"model_state_dict": ...}` (the scripts
handle both forms). Point `REPRO_CKPT_DIR` at this directory when running the reproduction
scripts.
