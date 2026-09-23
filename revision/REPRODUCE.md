# What this tier reproduces (and what it does not)

Added 2026-09-24 for the revised manuscript, *The Cost of Simultaneous Coverage in
Multi-Horizon Traffic Forecasting*.  Nothing here re-runs a neural network.

## Tier 1 - regenerate every published number and both figures

```bash
pip install numpy matplotlib PyMuPDF             # no torch, no data download
python revision/scripts/check_release_tables.py     # one command: tables + figures + manifest
```

`check_release_tables.py` is read-only and does three things: it imports
`revision/scripts/gen_tables_pr2.py`, rebuilds all 15 table files in memory from the
50 result records under `revision/results/` and compares them byte for byte with what
is shipped in `revision/paper/generated/`; it recomputes both figures' plotted values through
the bundled `revision/scripts/build_figures.py` and compares them with
`revision/paper/figure_data/` (CSVs byte for byte, coordinates to 1e-12), including the
12 source hashes recorded in `plotted_values.json`; and it re-hashes every path in
`RELEASE_MANIFEST.json`.  Any mismatch makes it exit non-zero, so the release carries its
own proof that the manuscript numbers and the shipped records agree.

What is claimed to be reproducible is the **data and the drawn geometry**, not file bytes:
regenerating the figures legitimately changes them, because pdfTeX stamps a build date into
each PDF and matplotlib embeds per-process object ids in the SVG sidecars.  The check above
therefore compares CSV files byte for byte, plotted values to 1e-12, and drawn coordinates to
1e-6, and it compares shipped tables to regenerated tables byte for byte.  The hashes in
`RELEASE_MANIFEST.json` answer a different question -- whether the copy in front of you is the
copy that was checked at release time -- so re-running the figure build will change them for
that file set and nothing in the numbers moves.

To regenerate rather than merely verify (same layout, writes inside this directory):

```bash
python revision/scripts/gen_tables_pr2.py --out revision/paper        # the 15 numeric tables
python revision/scripts/build_figures.py                           # figures + figure_data
```

## Tier 2 and tier 3 - the same code that produced the records

Everything the pipeline imports is in the repository, computed as a transitive closure rather
than hand-listed: ' + str(len(CLOSURE.get('scripts') or [0])) + ' scripts under `revision/scripts/` plus the calibration library it
actually imports, `' + ', '.join(CLOSURE.get('libs') or ['(computed at build)']) + '`, under `revision/uncalib_v1/SRC/`.  Note that this library is **not**
the same code as `calibration/*.py` at the repository root, which belongs to the submitted
version; `revision/LEGACY.md` says which is which.

```bash
export UC_PROJ=$(pwd)/revision      # the pipeline resolves its library and roots from this
python revision/scripts/e7_metrics_v1.py        # tier 2: statistics, gates, bootstrap
python revision/scripts/pr1_supplementary.py    # tier 2: fixed-interval supplementary analyses
```

Tier 2 additionally reads the prediction caches, and tier 3 (`e7_train*.py`, `e7_export.py`)
additionally needs the public PeMS arrays, BasicTS 0.5.8 and torch.  Those large inputs are not
redistributed; `revision/CACHES.md` lists every one of them with its size and full sha256, the
cache array schema, and the command order above.  After a tier-2 re-run, compare with the golden
records:

```bash
python revision/scripts/check_release_tables.py --tier2    # byte-level diff vs revision/results/
```

The comparison ignores environment-only fields (wall-clock timings, VRAM, GPU id, export
timestamps and the recorded training-provenance paths), because those legitimately differ on
any machine; every coverage, width, interval-score, bootstrap bound, gate flag and count is
compared exactly.  On this project's own re-run, 50 of 50 records came back with no
non-volatile difference and 49 byte-identical apart from `runtime_sec`.

The release is only claim-complete if that comparison reproduces the shipped records; if it
cannot be run, say so rather than describing this repository as a full reproduction of the
experiments.

## What the numbers come from

| Manuscript object | Regenerated from |
|---|---|
| Main Table 1 (population and complete-path shares) | `revision/results/phase3_PR1/pr1_supplementary.json` |
| Figure 1 (coverage-width operating points, 4 panels) | `revision/results/phase2_E7/v1/*.json` |
| Figure 2 (matched-model coverage reallocation) | same 12 configuration records |
| Main Table 2 (prospective cell ranges) | `revision/results/phase3_PR1/pr1_supplementary.json` |
| Supplement Tables S1-S13 | the same records via `revision/scripts/gen_tables_pr2.py` |

Aggregation rules are the declared ones: per-configuration ratios and differences are
formed first and then averaged unweighted over seeds 42/123/456; the effect tables print
raw statistics, while the bootstrap means are stored separately and never substituted for
them; intervals come from one frozen paired stationary block bootstrap (2,000 replicates,
geometric restart probability 1/288, indices shared between compared programs and seeds).

## Tier 2 and tier 3 are not in this repository

Tier 2 (re-estimate the statistics from forecasts) additionally needs the prediction
caches and half-width fields, which are several gigabytes of PeMS-derived arrays and are
not redistributed here.  Tier 3 (retrain the backbones) additionally needs the training
configurations and checkpoints; the training entry points in the parent repository
assume the external BasicTS framework.  The run registry (`plan/experiment_registry.json`) stays out of the public repository on
purpose: it embeds the authors' local working-directory paths for the audit bundles and is
process provenance rather than an input to any published number.  What it records -- run id,
script hash, artifact list, status for each of the 89 registered runs -- is summarised in
`RELEASE_MANIFEST.json`, which lists every file in this tier with its sha256.

The manuscript's Data availability statement
says exactly this, so nobody should read this repository as supporting an end-to-end
reproduction on its own.

## PeMS08 provenance

PeMS08 was not an unseen dataset: it had been used in an earlier, abandoned study in this
group, and that history is disclosed in the manuscript.  The test segment of PeMS08 was
opened only after an internal protocol freeze, under a signed seven-item checklist whose
record is shipped as `revision/plan/pems08_unseal_signoff.json`.  The freeze is an internal
project record, not a third-party registration, and the manuscript says so.
