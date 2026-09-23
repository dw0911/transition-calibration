# Plot data

`figure1_coverage_cost.csv` contains 24 natural operating points: six programs in each of four dataset–backbone groups. `path_coverage` is a proportion; `width_ratio` is the unweighted mean of the three within-configuration ratios to the pointwise program, on the V1 population.

`figure2_severity_increment.csv` contains four matched-model summaries. `Overall_change_pp`, `TypeB_change_pp` and `TypeC_change_pp` equal 100 times the mean within-configuration g1−g0 coverage difference. They are percentage points, not relative percent changes. `width_ratio_g1_g0` is retained as supporting numerical data, not a plotted axis.

`plotted_values.json` records the same values, source-file paths and SHA-256 hashes. These are derived presentation data from the twelve frozen result JSONs, not new observations or re-estimated intervals. No confidence interval is inferred from the spread of three seeds. The plots do not represent an equal-achieved-coverage frontier.

Rebuild with `python build_figures.py` at the archive root. Original result records are never overwritten.
