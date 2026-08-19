# Model checkpoints (companion data package)

The trained STAEformer checkpoints used by the paper's main tables are **not** committed to
GitHub (they are several hundred MB in total). They are distributed in the companion **data
package** so that a stranger can reproduce every table number without retraining.

## Layout

Place the checkpoint directory so that this repository looks like:

```
reproducible_uc/
├── checkpoints/
│   ├── PEMS04_r3_4split_seed42/best.pt       # Table 4 (PEMS04)
│   ├── PEMS04_r3_4split_seed123/best.pt
│   ├── PEMS04_r3_4split_seed456/best.pt
│   └── PEMS03_r3_4split_seed42/best.pt       # Table 4 (PEMS03)
```

Each `best.pt` is a PyTorch state dict that may be wrapped as `{"model_state_dict": ...}`
(the scripts handle both forms). The adjacent `train_meta.json` records the training protocol
(dataset, seed, best MAE/epoch) for provenance.

## Download

If the checkpoints are published as a GitHub Release asset or a shared link, either unpack the
archive into this directory, or run:

```bash
python scripts/download_checkpoints.py
```

Set `REPRO_CKPT_DIR` to point at the directory containing the `<dataset>_r3_4split_seed<seed>`
folders when you run the reproduction scripts:

```bash
export REPRO_CKPT_DIR=/path/to/reproducible_uc/checkpoints
```

## Checksums

After unpacking, verify integrity:

```bash
sha256sum checkpoints/PEMS04_r3_4split_seed42/best.pt
```

Expected hashes are listed in the data-package release notes.
