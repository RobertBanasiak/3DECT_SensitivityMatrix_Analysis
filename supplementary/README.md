# Supplementary Data S1

`Supplementary_Data_S1_pairwise_statistics.csv` contains the 630 paired
comparisons from the primary matched-reconstruction benchmark: 105 pairs of
configurations A-O for each of six metrics, with 40 paired target positions.
Five noise repetitions are first collapsed to a median for each target and
configuration. The file is unchanged from:

```text
reference_results/primary/run_20261013T173538_632316Z/tables/phantom_pairwise_comparisons.csv
```

Columns identify the metric, favourable direction, two configurations, paired
sample size, medians, median paired improvement of A over B, paired bootstrap
95% interval, paired win fraction, Wilcoxon p-value and Benjamini-Hochberg
adjusted q-value. Positive improvement favours A. Adjustment is performed
separately across the 105 comparisons for each metric. Bootstrap intervals use
2000 target-level resamples; Wilcoxon uses Pratt's zero convention and SciPy's
automatic method selection.

Use `pandas.read_csv(path, float_precision="round_trip")` when checking the
full-precision numerical export. This CSV can be uploaded separately to the
journal as Supplementary Data S1 alongside the supplementary PDF.
