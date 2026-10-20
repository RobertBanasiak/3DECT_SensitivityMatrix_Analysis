# Manuscript results and reference outputs

The revised manuscript reports common-basis comparisons of sensor and
measurement designs. Quadrature resolution is a method setting, documented in
`REPRODUCING.md` and assessed in the mapping-sensitivity study. Existing script,
manifest and output names retain `q512` so that commands and recorded source
hashes remain consistent with the completed analyses.

The paths below are relative to `reference_results/`. Exact run-directory
names and original archive checksums are listed in `reference_results/INDEX.json`.

| Manuscript content | Reference study | Principal source tables |
| --- | --- | --- |
| Mapping and grid sensitivity, Results 2.1-2.2 | `mapping/run_20261013T155031_270748Z/` | `mapping_scorecard.csv`, `mapping_changes.csv`, `mapping_rank_stability.csv`, `mapping_gram_spectra.csv` |
| Full-set sensor comparison, Table 1 | `primary/run_20261013T173538_632316Z/` | `tables/global_common_basis_scorecard.csv` |
| Reduced measurement sets, Supplementary Table S2 | Same primary run | `tables/subset_scorecard.csv`, `tables/subset_class_composition.csv`, `tables/selected_channels.csv` |
| Matched reconstruction, Table 2 and Supplementary Table S4 | Same primary run | `tables/phantom_summary.csv`, `tables/phantom_metrics_all_realisations.csv` |
| Paired statistical evidence and Supplementary Data S1 | Same primary run | `tables/phantom_pairwise_comparisons.csv` |
| Measurement-selection baselines | `baselines/run_20261013T201807_425028Z/` | `baseline_summary.csv`, `baseline_scorecard.csv`, `baseline_selected_channels.csv` |
| Selection stability | `stability/run_20261013T204022_406405Z/` | `stability_summary.csv`, `stability_noise_summary.csv`, `stability_overlap.csv`, `stability_scorecard.csv` |
| Computational cost | `benchmark/run_20261015T051003_416958Z/` | `runtime_by_stage.csv`, `runtime_summary.csv`, `benchmark_environment.json` |
| A-C cross-mesh reconstruction | `model_mismatch/run_20261015T093001_941649Z/` | `tables/mismatch_summary.csv`, `tables/mismatch_target_metrics.csv`, `tables/mismatch_paired_changes.csv` |

## Interpretation boundaries

- The primary basis has 2851 bins, shared spatial coordinates and 496 channels
  across configurations A-O. Volume-integrated mapping controls parameter
  correspondence and cell-volume scaling; it does not remove all native FEM
  discretisation error.
- Small descriptor changes from Q = 64 to 512 do not demonstrate full operator
  convergence. The maximum relative Frobenius change is 4.60% on the reference
  common support and 19.46% on the finer full-cylinder grid. Rankings can change
  under grid refinement. The supplied FEM fields are not refined by quadrature.
- The primary reconstruction benchmark uses matched linear forward and inverse
  models. The additional 36-case A-C study tests linear cross-mesh behaviour;
  it does not establish experimental or independent nonlinear validation.
- Selection baselines compare operator descriptors at the same subset size.
  Their differences do not establish improved reduced-subset image quality.
- The runtime benchmark measures computation on one reference computer.
  Channel-count fractions are potential acquisition reductions, conditional on
  an implementable measurement sequence; acquisition speed-up was not measured.
- Target-level paired statistics preserve target pairing. Adjustment covers 105
  configuration comparisons separately within each of six metrics. Imaging
  performance remains task dependent rather than defining one universal winner.

Numerical source files, reference-run manifests and original tables are
preserved. Documentation changes do not alter the recorded scientific results.
