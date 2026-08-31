# Common-basis analysis of three-dimensional ECT sensor and measurement designs

This repository contains the Python analysis workflow associated with the
manuscript:

**Common-basis analysis reveals task-dependent trade-offs in
three-dimensional electrical capacitance tomography sensor and measurement
design**

The workflow provides a reproducible computational framework for comparing
three-dimensional electrical capacitance tomography (ECT) sensitivity models
represented by heterogeneous finite-element discretisations.

It performs:

- volume integration and mapping of native FEM sensitivity matrices onto a
  shared equal-volume cylindrical basis;
- regularised spectral and spatial sensitivity analysis;
- electrode-pair classification and class-wise analysis;
- greedy regularised D-optimal measurement-channel selection;
- matched-model spherical-phantom reconstruction benchmarking;
- paired statistical comparisons;
- automated verification and provenance tracking.

## Software environment

The publication workflow was developed and executed using Python 3.12.3.

The main package versions used in the reported analysis are:

- NumPy 2.4.6
- SciPy 1.16.3
- pandas 2.3.3
- h5py 3.15.1
- Matplotlib 3.10.7

Exact runtime dependencies are listed in `requirements.txt` and
`pyproject.toml`.

## Installation

Create a Python 3.12 virtual environment and install the required packages:

```bash
pip install -r requirements.txt

## Archival release

The exact software version associated with the manuscript has been archived
as release `v1.0.0` in Zenodo:

https://doi.org/10.5281/zenodo.22209591

This archived release provides a permanent and immutable record of the
analysis code corresponding to the reported study. The GitHub repository
remains the actively maintained development repository.

## Citation

If you use this software, please cite the archived Zenodo release:

> Banasiak, R. (2026). *Common-basis analysis of three-dimensional ECT sensor
> and measurement designs* (Version 1.0.0). Zenodo.
> https://doi.org/10.5281/zenodo.22209591

Citation metadata are also provided in `CITATION.cff`.

## Licence

This software is released under the BSD 3-Clause License. See `LICENSE` for
details.
