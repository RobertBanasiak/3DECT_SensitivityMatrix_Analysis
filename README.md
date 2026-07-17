# Common-basis analysis of 3D ECT sensitivity matrices

This repository contains the reproducible Python workflow used to compare
three-dimensional electrical capacitance tomography (ECT) sensitivity matrices
on a shared equal-volume cylindrical basis. It computes global spectral and
spatial criteria, electrode-pair group summaries, greedy measurement subsets,
and paired spherical-phantom image-reconstruction metrics.

## Running the analysis in PyCharm

1. Open this repository folder as a PyCharm project.
2. Select a Python 3.12 interpreter or create a new virtual environment.
3. Install the packages from `requirements.txt` using PyCharm's Python
   Interpreter window.
4. Put the MATLAB v7.3 files `meshA.mat` through `meshO.mat` in `matrices/`.
5. Open `main.py` and edit only the `PYCHARM_CONFIG` block if required.
6. Right-click `main.py` and select **Run 'main'**, or press the green Run
   button while `main.py` is the current file.

No command-line arguments are required for the publication workflow. Paths in
`PYCHARM_CONFIG` are resolved relative to `main.py`, not to PyCharm's working
directory.

## Installation self-test in PyCharm

Open `run_self_test.py` and press Run. A correct installation prints:

```text
SELF-TEST PASSED
```

The self-test does not require the `.mat` input files and does not generate the
publication results.

## Expected input format

Each MATLAB v7.3/HDF5 file must contain:

- `S`: sensitivity matrix, with one dimension equal to the tetrahedron count;
- `vtx`: mesh vertices, orientable as `N x 3`;
- `simp`: tetrahedral connectivity, orientable as `M x 4`.

The default publication configuration expects 32 electrodes and 496 unordered
measurement channels for each of the 15 configurations A--O.

## Output

The default run creates `SIMPAT_results/` with:

- `tables/`: scorecards, selected channels, phantom metrics, statistical
  comparisons, and automated quality checks;
- `figures/`: publication figures generated from the same run;
- `run_manifest.json`: parameters, software versions, input hashes, grid
  definition, and interpretation notes.

The random seed and matched noise draws are fixed in `PYCHARM_CONFIG`.

## Project structure

```text
main.py                       PyCharm entry point and editable configuration
run_self_test.py              PyCharm installation test
ect_analysis/
  cli.py                      argument validation and file discovery
  mesh.py                     MATLAB/HDF5 loading and common-grid mapping
  models.py                   shared data classes
  metrics.py                  spectral and spatial criteria
  channels.py                 electrode-pair classes and greedy selection
  phantoms.py                 phantom generation, inversion, and statistics
  plotting.py                 publication figures
  reporting.py                CSV, JSON, and checksum utilities
  pipeline.py                 explicit end-to-end analysis orchestration
tests/                        automated regression tests
```

## Reproducibility and archival release

Before submission, the exact code release and shareable sensitivity matrices
should be archived in Zenodo. The Zenodo DOI should be cited in the manuscript;
GitHub can remain the actively maintained development repository.

The software and data licences must be selected after confirming institutional
rights to distribute the code and matrices.
