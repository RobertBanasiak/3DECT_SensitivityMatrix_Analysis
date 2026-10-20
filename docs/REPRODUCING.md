# Reproducing the revised analyses

## Fixed publication settings

The primary analysis uses A-O, 496 channels, float64 sensitivity densities and
millimetre geometry. The grid has 10 radial, 16 angular and 20 axial divisions,
with radial boundaries uniform in squared radius. The frozen mask has 2851
bins and SHA-256 digest:

```text
9e39d980294e2de86bb10618e924d32100fe77a6483328d448646142b3db8338
```

The primary mapper uses Q = 512, a maximum block size of 4096 native cells and
a point budget of 262144. The budget limits effective blocks to at most 512
tetrahedra at Q = 512. Selection uses k = 128, unit independent noise, relative
alpha 0.001 and sweep values 0.0001, 0.001 and 0.01. The recorded primary
absolute alpha is 1254.9733365162294.

Reconstructions use spherical targets of radius 20 mm, 40 retained positions,
five paired noise realisations, 1% relative noise, relative Tikhonov parameter
0.01, a 50%-of-peak threshold, seed 123 and 2000 target-level bootstrap resamples.
The target position, after the median over noise repetitions, is the statistical
unit. Signed images are retained for AUC, NRMSE and weighted correlation.

## Entry points and output locations

| Study | Entry point | Output parent |
| --- | --- | --- |
| Primary analysis | `run_q512_revision.py` | `q512_revision_results/` |
| Mapping/grid sensitivity | `analyze_mapping_sensitivity.py` | `mapping_sensitivity_results/` |
| Selection baselines | `run_q512_selection_revision.py --selection-study baselines` | `q512_baseline_comparison_results/` |
| Selection stability | `run_q512_selection_revision.py --selection-study stability` | `q512_selection_stability_results/` |
| Runtime/memory | `run_q512_benchmark_revision.py` | `q512_benchmark_results/` |
| A-C model mismatch | `run_q512_model_mismatch.py` | `q512_model_mismatch_results/` |

Each execution creates a timestamped run directory. Dependent studies require a
compatible completed primary run. With a non-default primary output parent,
pass `--primary-run` to its completed run directory. Generate a new primary run
after installing this release: changing the version metadata intentionally
changes its source hash. Do not bypass the source/input guards.

## Mapping sensitivity

```text
python analyze_mapping_sensitivity.py --input-dir matrices --mapping-grids 10x16x20 20x32x40 --subdivision-levels 0 2 3
```

The helper reports centroid-common support and full-cylinder scopes separately.
The former can change physical coverage between grid resolutions; the latter
keeps geometric extent fixed. The publication study fixes centroid-reference
alpha density at 0.7714314553447922, with alpha 1341.9314099132944 on the reference
grid. These values intentionally differ from the recalibrated primary alpha.
This study does not perform reconstructions.

At the reference resolution, Q = 64 to 512 changes the aggregate descriptors
only slightly and preserves the regularised-information ranking. This supports
the primary quadrature setting at that resolution; it does not establish full
operator convergence. The maximum relative Frobenius operator change is 4.60%
on the reference common support and 19.46% on the finer full-cylinder grid.
Grid resolution and physical coverage must be assessed separately. Subdivision
quadrature keeps the supplied native FEM sensitivity fields unchanged.

## Selection studies

Baselines include greedy regularised D-optimal selection, column-pivoted QR,
highest-energy selection, a low-coherence heuristic and 100 uniform random
subsets per configuration. Deterministic selection timings use five repeats
and exclude shared preprocessing and subset scoring.

Stability defaults use k = 64, 96, 128, 160, 192 and 256; ten unequal-variance
profiles with standard-deviation factor two; and shared-electrode correlation
strengths 0.25 and 0.50. Alpha is not recalibrated for each imposed noise scenario.

## Runtime benchmark

The benchmark measures mapping including HDF5 loading, full-set Gram/eigenvalues,
selection from a precomputed Gram matrix, and reconstruction including data
generation and metrics. Preparation, pipeline and verification clocks are
reported separately. The reference Windows run reports 182.65 s pipeline time,
185.30 s including preparation/verification and 2.33 GB lifetime peak process
resident memory. They describe one execution with uncontrolled disk-cache
state, not acceptance targets for a different computer. Acquisition and
128-versus-496 reconstruction speed-ups are not measured.

## Model mismatch

All nine ordered A-C forward/inverse combinations are evaluated under four
noise scenarios. The study keeps the same primary targets and unweighted
Tikhonov inverse under correlated noise. Expected dimensions: 36 cases, 5760 raw
reconstructions, 1440 target records and 144 paired metric comparisons. Matched
independent-noise controls are checked against the primary reconstruction table.

This evaluates linear cross-mesh behaviour on the shared reconstruction basis.
It does not establish independent nonlinear or experimental validation, or
robustness of the F/G/H ordering to model mismatch. Synthetic self-tests verify
the implementation separately from the full native-input experiment.

## Statistical exports

Use `pandas.read_csv(path, float_precision="round_trip")` for the 17-digit CSVs.
Paired bootstrap intervals resample targets, preserving pairing. Wilcoxon tests
use Pratt's zero convention and SciPy's automatic method selection, rather than
universally exact p-values. Benjamini-Hochberg adjustment is applied separately
to the 105 pairs within each metric.

`supplementary/Supplementary_Data_S1_pairwise_statistics.csv` is a byte-identical
copy of the primary run's `tables/phantom_pairwise_comparisons.csv`: 630 rows,
six metrics, 105 configuration pairs per metric and 40 paired target positions.
It is the separately named CSV accompanying the supplementary PDF.

The release check of the stored Windows primary results on Linux reproduced
the summary statistics and paired improvements/bootstrap intervals exactly.
Some p-values and adjusted q-values differed only in their last floating-point
bits (maximum absolute differences 2.78e-17 and 5.55e-17, respectively). The
q < 0.05 decisions were unchanged. Cross-platform reference-table comparisons
should therefore allow floating-point tolerance, for example relative 1e-12
and absolute 1e-15. This is separate from the scripts' strict within-run CSV
round-trip check; that internal guard is retained.

## Original reference outputs

The six original completed runs are in `reference_results/`, with their original
run-directory names. Their manifests, figures and historical local paths are
preserved. They are prior author-run results, not newly computed release results.
Their source hashes include the old version string, which is explicitly
accounted for in `provenance/source_audit.json`.

For fresh computation, generate a new primary run. To plot Figure 1 directly
from the included reference outputs, use explicit paths:

```text
python plot_fig1_q512.py --primary-run reference_results/primary/run_20261013T173538_632316Z --mapping-run reference_results/mapping/run_20261013T155031_270748Z
```
