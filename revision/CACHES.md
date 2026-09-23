# Large artifacts referenced by tiers 2 and 3 (not redistributed here)

Tier 1 needs nothing from this file. Tiers 2 and 3 read the arrays and weights below; they
are **not** in the repository, and this page exists so that a reviewer can tell exactly what
is missing, how big it is, and how to check whatever they obtain from elsewhere.

Total not redistributed: prediction caches 3.09 GB + checkpoints 28.7 MB.
Reason: size (GitHub limits), plus PeMS-derived arrays are redistributed from their own
source rather than second-hand here.

## Prediction caches - `revision/cache/*.npz` (tier 2 input)

Each file holds one (network, backbone, seed, segment) export with nine arrays:
`pred (n,12,N) f4`, `true (n,12,N) f4`, `flow_in (n,N) f4`, `pred_mag (n,N) f4`,
`r_in_max (n,N) f8`, `argpos (n,N) i1`, `lag (n,N) f4`, `r_last3 (n,N) f4`,
`r_recency (n,N) f4`, where n is the number of window origins in that segment and N the node
count. dtype choices are load-bearing: `r_in_max` is float64 because the severity z-score
feeds the learner target, the rest are float32 as exported.

| file | size | sha256 |
|---|---|---|
| `PEMS03_s123_cal.npz` | 116.3 MB | `a841c3d0578e0ae2b5cee59e1951a403e44f2e48e5e6b8771458fcf80b142069` |
| `PEMS03_s123_sel.npz` | 116.3 MB | `e4c1400192b7e00061a77ccccfd635dddfdf13c717890504a44f8bc700e3f460` |
| `PEMS03_s123_test.npz` | 235.4 MB | `34cb27e20c6e7d902a8bca9b99a0e83a2ae49098d97cdb933f9fb54f41abfd79` |
| `PEMS03_s42_cal.npz` | 116.3 MB | `b05cae38bb388847d694f564d9fe0ba9dcd0187ef7d6b97d4339c07a00448ddd` |
| `PEMS03_s42_sel.npz` | 116.3 MB | `69ef28eada0a44d387051e7435ad644335d575d00b8d8789b104d1a65302325e` |
| `PEMS03_s42_test.npz` | 235.4 MB | `b19b72881050385f8f05dd765afabc7cc10d045113c4f47ec59b773329856afe` |
| `PEMS03_s456_cal.npz` | 116.3 MB | `0baa4be3506408f71ad7a96e5868b03cc6f8aedbff97cb0fce58ef518c81bdda` |
| `PEMS03_s456_sel.npz` | 116.3 MB | `d718a0d13b21014d329b78e96c5adbdcafb0877b18f5dd7989bbad8bc3694618` |
| `PEMS03_s456_test.npz` | 235.4 MB | `a49da6e354a42f09008289b85d9be453b89f39942d0f7180fa3f51e296d190e3` |
| `PEMS04_s123_cal.npz` | 64.3 MB | `d7876ae1610a92ae4cd4b3c45483e5558d7733bbcdca31bd96e3db5b92bc7f78` |
| `PEMS04_s123_sel.npz` | 64.3 MB | `0ad1a7562ae2c89a1efaddf780b0206dad081b9ff7373d7c87a199cb093a83a8` |
| `PEMS04_s123_test.npz` | 130.6 MB | `5b82be91eaee6df86c1a3f0c08a1c18167736a492452d1038a8be934cb2ca113` |
| `PEMS04_s42_cal.npz` | 64.3 MB | `daf04afb24245bd79f603ffd472fb052bbd05ba493b6bda518405c648d2e7eba` |
| `PEMS04_s42_sel.npz` | 64.3 MB | `b3925b2b5dfe9a1429a279b229e2510466ff73f6e54699f49ae059a19bd97b8d` |
| `PEMS04_s42_test.npz` | 130.6 MB | `e15ebfcffa92e70ee4210c1c33dd1fefebbeb9fb5dfdd6f22ed8b46a9d220ca4` |
| `PEMS04_s456_cal.npz` | 64.3 MB | `f8a5051ff05d15b4b082f92adc603b9b927a53b3425c2d44d9234b1faf5e533e` |
| `PEMS04_s456_sel.npz` | 64.3 MB | `6675623664b1c6ff3212aa3b9834caa7af52addfc21b12981222b65e899d05cb` |
| `PEMS04_s456_test.npz` | 130.6 MB | `081a2feb79fdea378f2ae55d5b710b0404f5786c9b4a12205856611ec638a93b` |
| `PEMS08_s123_cal.npz` | 37.4 MB | `87a06b00debb77186e8e4e742600039012661d00270375aa470fb3e03361c5a6` |
| `PEMS08_s123_sel.npz` | 37.5 MB | `81ca1eeea8ee555986447b8765a91a166f534b54589a69c76159e76f1c385981` |
| `PEMS08_s123_test.npz` | 76.0 MB | `a4de2d64ee0d91ec2fbd7bff86d6677d6fe52edb6667aefbd1d2a0bd1606806c` |
| `PEMS08_s42_cal.npz` | 37.4 MB | `3d8c47ffb6571debc2cf6520a2d47833ee17a2438ae1689c48fceb5c37260cca` |
| `PEMS08_s42_sel.npz` | 37.5 MB | `eac017e112de1d882aa87a1144a969ca738718077ad62c23b15c235e5efc01f5` |
| `PEMS08_s42_test.npz` | 76.0 MB | `f366a553e24e28ca8b12c4fc83b08a2782c644013daa6ab1b0bfe0626d17e604` |
| `PEMS08_s456_cal.npz` | 37.4 MB | `24a277b040b660d16dde5e69d5ef3df877005d8406dbcc08e6ed690b8493317a` |
| `PEMS08_s456_sel.npz` | 37.5 MB | `c823440082b840fc554d64513c3f843248b8a5947a06755f22251b46fd720569` |
| `PEMS08_s456_test.npz` | 76.0 MB | `a23a1b0af88d6172558cbb9d14ed0bdd3ea497d31ea33113dcbd40aff488f181` |
| `PEMS08_stid_s123_cal.npz` | 37.4 MB | `dfc2ded37c14e509688635b35656a65d5f2b7044736278c92ebe88c600691988` |
| `PEMS08_stid_s123_sel.npz` | 37.5 MB | `1054df7da0ccf58135185b1514a94d4e527c257eb1c1d938e5aafb1652d6ef9d` |
| `PEMS08_stid_s123_test.npz` | 76.0 MB | `2f7532845d0c5344311c7d630fbebb646522355c51c6e46d8ea156ddc72d1ccd` |
| `PEMS08_stid_s42_cal.npz` | 37.4 MB | `0007e75aa4ef9b591653f45a7a9c6cc91facbc7e172d7b01e2df0cec0c1303c0` |
| `PEMS08_stid_s42_sel.npz` | 37.5 MB | `dc35df8df6156e5c69abfbff8e41163e16154130e07a838deffbf605817c074c` |
| `PEMS08_stid_s42_test.npz` | 76.0 MB | `fd12e3d80d2a22f31ca84416dfe474b3adfe5a8b07402e59df170e12ce8dbbc0` |
| `PEMS08_stid_s456_cal.npz` | 37.4 MB | `180c9771e7927a8c47acf80c9c0c283e909bdfd0bc16a2d4dfc657006fa50d92` |
| `PEMS08_stid_s456_sel.npz` | 37.5 MB | `4ce65129945ead5ee9764dd5bec58f13a9257746a56210f1cb73cd03960ff9c1` |
| `PEMS08_stid_s456_test.npz` | 76.0 MB | `3de9f1a36dfda1617ea5f8b5a97f866775d884d010975379811cae0ea188ec94` |

Verify any file you obtain by recomputing its digest (`sha256sum <file>`) and comparing with
the full value in the table; these are the same hashes recorded in the project's
`audit/artifact_inventory.json`, so the repository and the working records agree.

## Checkpoint / training metadata - `revision/checkpoints/` (tier 3 input)

| file | size | sha256 |
|---|---|---|
| `best.pt` | 5.7 MB | `13bef7b28850d5215229368d17854139017a45d512531d05ddd96ea3b4020e06` |
| `train_meta.json` | 0.0 MB | `e3a9b7d9a4eab905bff03f6f39c9a78aacfbe5a1ac5a8cb07ce88c6ea77dd7ac` |
| `pilot_meta.json` | 0.0 MB | `bc713e1d1b85e4892d4c43fbb6fb759f42c4ef7938e6f4574e1283bbf7a25576` |
| `best.pt` | 5.7 MB | `d8f2d138d1c1676481a3e9234174666f2abcb0ac93cc6705dbe15314355418fc` |
| `train_meta.json` | 0.0 MB | `d8ce439b8d63f899c677e1dc7677821d9069d6dc5337ca5338ea95cec3a3de93` |
| `best.pt` | 4.9 MB | `0d86440fc2d9a272f96dba72bb11676486d472d2b44d9b93b4473454f7a1e566` |
| `train_meta.json` | 0.0 MB | `3fd1033e21dabe1cc60be5c982a0811de70601ecf9752dfce7282cc2544b3261` |
| `best.pt` | 4.9 MB | `082072a9ed859a27e1537cf9e955572b14b52b9403858a3023f0227f5551be1d` |
| `pilot_meta.json` | 0.0 MB | `72afc230d2d0b14be0dc3f0ddbca490a355b789603f268f555c1592810e58f70` |
| `train_meta.json` | 0.0 MB | `eafc06ba32ef8c45db7acf66b2a9a96c4eee3c4e5562ceb6e9ab3ae3f5bdab35` |
| `best.pt` | 4.9 MB | `9670ac7b277e3b1462f6d32cbaf087e669123c97057adaefa755020586eee84a` |
| `train_meta.json` | 0.0 MB | `c01768f1f391f08c3ab20a1f6212dbf3ab118a2349a3d176a7d38e77728626df` |
| `best.pt` | 0.9 MB | `13e73584db2c5a756a60364814a28cddbf285fc9e0bf1031609fee25cb804b5a` |
| `train_meta.json` | 0.0 MB | `8b6edb4856bcb168e9c4587e02e524dfa6a3d80c380834bcb29e925af2c7a237` |
| `best.pt` | 0.9 MB | `0731b5b98f883b607491a0f4fe89c01cd2d4e43f061d091a4f86237c1807d232` |
| `pilot_meta.json` | 0.0 MB | `cf2390ce0bb23ca49419bd95dcbe257e96dff314c62a04063eed0d3afc60da90` |
| `train_meta.json` | 0.0 MB | `30752bc790d3ae4f478429a454c887c9ec114a269ee97641e9132cb054e57c12` |
| `best.pt` | 0.9 MB | `c1dc5239b3d5fe0c544409cf8451bd1d2c0a718462c8e523f42df5eace565562` |
| `train_meta.json` | 0.0 MB | `e8d9ada83eba0ec49b6cceb6ebaacf2b61e9fa9fef2c79e6ffd2128ae1c1a98b` |

Weights themselves are not redistributed (see the repository's checkpoint policy); the
metadata above pins the training runs whose exports produced the caches.

## How to regenerate the caches yourself

```bash
# 0. dependencies (tier 3 only), then point the pipeline at this checkout
export UC_PROJ=$(pwd)/revision

# 1. PeMS03/04/08 from the public PeMS source, through BasicTS 0.5.8, into $UC_PROJ/BasicTS058
# 2. train the four-segment backbones (12 configurations, seeds 42/123/456)
python revision/scripts/e7_train.py --dataset PEMS08 --backbone stid --seed 42
python revision/scripts/e7_train_stid.py --dataset PEMS08 --seed 42
# 3. export the per-segment prediction caches listed above
python revision/scripts/e7_export.py            # gate: refuses PEMS08 test before sign-off
# 4. recompute the statistics, gates and golden records, then compare against revision/results/
python revision/scripts/e7_metrics_v1.py
python revision/scripts/pr1_supplementary.py
python revision/scripts/check_release_tables.py --tier2   # byte-level compare against shipped records
```

Two properties of this pipeline are worth knowing before you start: the PEMS08 test export is
gated on the signed unseal checklist that ships as `plan/pems08_unseal_signoff.json`, and the
bootstrap is replayed from a fixed seed with indices shared across compared programs, so a
correct re-run reproduces the shipped result records rather than merely falling near them.
