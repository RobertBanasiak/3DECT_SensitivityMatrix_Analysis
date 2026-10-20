# Common-basis analysis of three-dimensional ECT sensor and measurement designs

Version **1.1.0** contains the common-basis analysis and additional numerical studies
used in the revised manuscript:

**Common-basis computational analysis reveals task-dependent trade-offs in
three-dimensional electrical capacitance tomography sensor and measurement design**

Author: Robert Banasiak, Lodz University of Technology.

## Included analyses

- Primary volume-integrated analysis on the fixed 2851-bin domain, spectral and
  spatial analysis, 128-channel selection, 3000 matched reconstructions and
  630 paired statistical comparisons.
- Mapping/grid sensitivity with Q = 1, 64 and 512 on two cylindrical grids.
- Selection baselines and sensitivity to regularisation, subset size and noise.
- Runtime and peak-process-memory benchmark of the primary pipeline.
- A-C cross-mesh reconstruction: 36 cases and 5760 reconstructions.
- Original reference results, manifests, source provenance and file checksums.

These are computational comparisons of precomputed FEM models. The primary
reconstruction benchmark uses matched linear forward and inverse operators.
Subdivision quadrature approximates integration of native piecewise-constant
sensitivity; it is not an exact curved-bin intersection calculation. The A-C experiment is a linear
cross-mesh test, not experimental or independent nonlinear validation.

## Native inputs

The fifteen native matrices are archived separately at:

https://doi.org/10.5281/zenodo.21409630

Place `meshA.mat` through `meshO.mat` in `matrices/`, or pass their directory
with `--input-dir`. These large inputs are not duplicated in this distribution.
The supplied geometry uses millimetres and the sensitivities are densities.

## Environment

Use **Python 3.12.4** with the libraries pinned in `requirements.txt`.
For example on Windows, select a Python 3.12.4 installation:

```text
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements-dev.txt
```

On Linux/macOS use `python3.12 -m venv .venv` and
`source .venv/bin/activate` before installing the requirements. Run the commands
below from this release's root directory.

## Checks without native matrices

```text
python -m pytest -q
python run_q512_revision.py --self-test
python run_q512_selection_revision.py --self-test
python run_q512_benchmark_revision.py --self-test
python run_q512_model_mismatch.py --self-test
```

These are synthetic numerical and integration checks, not a replacement for
rerunning the full native-input experiments.

## Primary analysis

```text
python run_q512_revision.py --input-dir matrices
```

Use `run_q512_revision.py` for the revised primary analysis. `main.py` is retained
as the original centroid-mapping configuration and backend; running it alone
uses the older mapping. The primary entry point creates a fresh
`q512_revision_results/run_*` directory, uses the frozen original 2851-bin mask,
recalibrates the shared alpha once on the mapped operators, and verifies full-precision
CSV exports. Native inputs and previous runs are not overwritten.

## Additional studies

After the primary run completes:

```text
python run_q512_selection_revision.py --selection-study baselines --input-dir matrices
python run_q512_selection_revision.py --selection-study stability --input-dir matrices
python run_q512_benchmark_revision.py --input-dir matrices
python run_q512_model_mismatch.py --input-dir matrices
```

These entry points find the latest complete primary run under
`q512_revision_results/`. Use `--primary-run PATH` to select a particular run.
Keep the source files unchanged between the primary run and its dependent
studies: both source and native-input hashes are checked.

The publication mapping-sensitivity settings override the helper's defaults:

```text
python analyze_mapping_sensitivity.py --input-dir matrices --mapping-grids 10x16x20 20x32x40 --subdivision-levels 0 2 3
```

Levels 0, 2 and 3 correspond to Q = 1, 64 and 512. This study holds the
centroid-reference regularisation density fixed; it is separate from the
recalibrated primary analysis. Figure 1 can then be generated from
compatible completed runs with `python plot_fig1_q512.py`.

## Reference results and provenance

`reference_results/` contains the author's six completed numerical runs grouped
by study, with original tables, figures and manifests. `reference_results/INDEX.json`
records the original archive names and SHA-256 hashes. Historical local paths
and recorded environments in those manifests have been preserved.

The numerical source files match the hashes recorded in the reference runs.
Only the version string in `ect_analysis/__init__.py` has changed to 1.1.0;
this metadata-only difference is recorded in `provenance/source_audit.json`.
Generate a fresh primary run from this release before running its dependent
studies so their source hashes include the current release metadata. Reference
outputs are kept separate from automatically discovered new runs.

Read exported float64 tables using `pandas.read_csv(path, float_precision="round_trip")`.
`SHA256SUMS.txt` covers the distributed files, and local checks are recorded in
`provenance/release_validation.json`. Further settings and limitations are in
`docs/REPRODUCING.md`. A guide to the manuscript results and their source tables
is in `docs/MANUSCRIPT_ALIGNMENT.md`. The journal-ready paired-statistics CSV is
included as `supplementary/Supplementary_Data_S1_pairwise_statistics.csv`.

Citation metadata: `CITATION.cff`. Software publication instructions:
`docs/RELEASING.md`. The software and input dataset have separate citations.
License: BSD-3-Clause; see `LICENSE`.
