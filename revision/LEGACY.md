# Relationship to the 2026-08 code in this repository

The files that used to sit at the repository root (`calibration/`, `evaluation/`,
`experiments/`, `tests/`) implemented the **submitted** version of the study.  They were removed
from the working tree when this release replaced it, and remain readable in history at commit
`b143fc2`, whose title claimed a
deployment-observable transition-aware calibration method.  That manuscript was
desk-rejected, and the revision changed the scientific question rather than only the
packaging.  Both remain here on purpose, but they are not the same evidence and must not
be quoted interchangeably:

| | submitted version (root) | revision (`revision/`) |
|---|---|---|
| Question | does a severity-conditioned calibrator improve reliability | what width does simultaneous path coverage cost, and how does that cost vary across environments |
| Split | train / calibration / test, three-way | training / selection / calibration / test, 60/10/10/20 on the original timeline with a 23-origin purge |
| Validity rule | elements with `true > 0` | complete 12-step node paths (V=1) as the primary population |
| Headline quantity | WCE / GCD method-effect deltas | achieved path coverage and paired width ratio against pointwise calibration, with one-sided bootstrap bounds and two pre-set gates |
| Severity | proposed mechanism | one added feature in a matched learner: width falls 1.1-1.5%, which fails the 5% efficiency gate; the reallocated coverage sits in retrospective TypeB/TypeC groups |

Two corrections found during the revision also apply to the older artifacts.  The
submitted appendix tables were computed with calibration data that overlapped the
evaluation windows; recomputed on the independent calibration segment, three of those
robustness tables keep their conclusions and one has to be reworded, and the contaminated
numbers were systematically mildly optimistic.  Separately, some stored contrasts in the
early probe artifacts were produced by an older bootstrap random stream, so point
estimates agree while resampled bounds differ in the last digits.  For that reason the
revision numbers quoted in the manuscript come only from the frozen generator outputs
under `revision/results/`, never from the older probe files.

Those historical files are left unmodified in history -- including the appendix numbers
that the revision later recomputed on an independent calibration segment -- so that what was
actually sent for review stays traceable rather than rewritten.
