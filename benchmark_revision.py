#!/usr/bin/env python3
"""Measure the four main stages of 3DECT_SensitivityMatrix_Analysis v1.0.0.

Place this file beside the project's main.py and run it in the same Python
environment. With no arguments it reuses main.py's PYCHARM_CONFIG. Explicit
arguments use the original command-line parser, for example:

    python benchmark_revision.py --input-dir "D:\\ECT\\matrices"

Results go to a new timestamped folder under revision_benchmark_results.
Original numerical routines and input files are unchanged. Mapping includes
HDF5 reading; the reconstruction timer includes synthetic-data generation,
linear inversion and image metrics. Full-set spectral timing covers whitening,
Gram construction and eigenvalues. Auxiliary reporting, plots, statistics,
subset scorecards and input hashing are included only in total pipeline time.

Peak RSS is the OS-reported lifetime peak of the entire Python process,
including native-library allocations. It is not an isolated peak per stage.
Times describe a single execution; disk-cache state is not controlled.
No additional Python dependencies are required beyond the original project.
"""

from __future__ import annotations

import csv
import functools
import hashlib
import inspect
import io
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path


PROJECT_DIRECTORY = Path(__file__).resolve().parent
STAGES = (
    "common_basis_mapping",
    "full_set_gram_and_eigenvalues",
    "greedy_selection",
    "matched_reconstruction_benchmark",
)
FIELDS = (
    "configuration", "stage", "elapsed_seconds", "status", "native_cells",
    "grid_bins_before_filter", "common_parameters", "measurement_channels",
    "selected_channels", "target_positions", "reconstruction_vectors",
    "successful_reconstructions",
)


def process_peak_rss() -> tuple[int | None, str]:
    """Return the OS process high-water mark in bytes, without sampling."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    *[(name, ctypes.c_size_t) for name in (
                        "PeakWorkingSetSize", "WorkingSetSize",
                        "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                        "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                        "PagefileUsage", "PeakPagefileUsage",
                    )],
                ]

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel.GetCurrentProcess.argtypes = []
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            query = psapi.GetProcessMemoryInfo
            query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            query.restype = wintypes.BOOL
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            if not query(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                raise ctypes.WinError(ctypes.get_last_error())
            return int(counters.PeakWorkingSetSize), "Windows PeakWorkingSetSize (bytes)"
        if sys.platform.startswith("linux") or sys.platform == "darwin":
            import resource
            value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            factor = 1 if sys.platform == "darwin" else 1024
            return int(value * factor), "getrusage(RUSAGE_SELF).ru_maxrss"
        return None, "unsupported platform"
    except Exception as exc:
        return None, f"unavailable: {type(exc).__name__}: {exc}"


def physical_ram_bytes() -> int | None:
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                    *[(name, ctypes.c_ulonglong) for name in (
                        "ullTotalPhys", "ullAvailPhys", "ullTotalPageFile",
                        "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual",
                        "ullAvailExtendedVirtual",
                    )],
                ]

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            query = kernel.GlobalMemoryStatusEx
            query.argtypes = [ctypes.POINTER(MemoryStatus)]
            query.restype = wintypes.BOOL
            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(status)
            if query(ctypes.byref(status)):
                return int(status.ullTotalPhys)
        elif sys.platform == "darwin":
            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
        else:
            return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def cpu_model() -> str:
    try:
        if sys.platform == "win32":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                    r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine()


def source_hashes() -> dict[str, str]:
    paths = [PROJECT_DIRECTORY / "main.py", Path(__file__).resolve()]
    paths.extend(sorted((PROJECT_DIRECTORY / "ect_analysis").glob("*.py")))
    return {p.name if p.parent == PROJECT_DIRECTORY else f"ect_analysis/{p.name}":
            hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot encode {type(value).__name__}")


class StageProfiler:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.rows = []
        self.mapped_labels = []
        self.gram_labels = {}
        self.spectral_calls = 0
        self.originals = {}

    def _details(self, stage, arguments):
        if stage == STAGES[0]:
            meta, grid = arguments["meta"], arguments["grid"]
            return {"configuration": meta.label, "native_cells": meta.n_cells,
                    "measurement_channels": meta.n_channels,
                    "grid_bins_before_filter": grid.n_bins}
        if stage == STAGES[1]:
            if self.spectral_calls >= len(self.mapped_labels):
                raise RuntimeError("Unexpected spectral call order; this profiler supports v1.0.0.")
            label = self.mapped_labels[self.spectral_calls]
            self.spectral_calls += 1
            sensitivity = arguments["sensitivity"]
            return {"configuration": label, "common_parameters": sensitivity.shape[0],
                    "measurement_channels": sensitivity.shape[1]}
        if stage == STAGES[2]:
            gram = arguments["gram"]
            return {"configuration": self.gram_labels.get(id(gram), "unknown"),
                    "measurement_channels": gram.shape[0],
                    "selected_channels": int(arguments["k"])}
        sensitivity = arguments["sensitivity"]
        targets = len(arguments["positions"])
        return {"configuration": arguments["label"],
                "common_parameters": sensitivity.shape[0],
                "measurement_channels": sensitivity.shape[1],
                "target_positions": targets,
                "reconstruction_vectors": targets * int(arguments["repeats"])}

    def _wrap(self, function, stage):
        signature = inspect.signature(function)

        @functools.wraps(function)
        def measured(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            details = self._details(stage, bound.arguments)
            row = {"stage": stage, "status": "error", **details}
            start = time.perf_counter()
            try:
                result = function(*args, **kwargs)
            except BaseException:
                row["elapsed_seconds"] = time.perf_counter() - start
                self.rows.append(row)
                raise
            row["elapsed_seconds"] = time.perf_counter() - start
            row["status"] = "ok"
            if stage == STAGES[0]:
                self.mapped_labels.append(details["configuration"])
            elif stage == STAGES[1]:
                self.gram_labels[id(result[0])] = details["configuration"]
            elif stage == STAGES[2]:
                row["selected_channels"] = len(result[0])
            else:
                row["successful_reconstructions"] = int(result["status"].eq("ok").sum())
            self.rows.append(row)
            print(f"  BENCHMARK {details['configuration']} {stage}: "
                  f"{row['elapsed_seconds']:.3f} s", flush=True)
            return result
        return measured

    def __enter__(self):
        for name, stage in zip(("map_to_common_grid", "gram_and_eigenvalues",
                                "greedy_regularized_logdet", "run_phantoms"), STAGES):
            self.originals[name] = getattr(self.pipeline, name)
            setattr(self.pipeline, name, self._wrap(self.originals[name], stage))
        return self

    def __exit__(self, *exc):
        for name, function in self.originals.items():
            setattr(self.pipeline, name, function)

    def save(self, output):
        with (output / "runtime_by_stage.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        summary_fields = ("stage", "calls", "total_seconds", "median_seconds",
                          "min_seconds", "max_seconds")
        with (output / "runtime_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, summary_fields)
            writer.writeheader()
            for stage in STAGES:
                values = [r["elapsed_seconds"] for r in self.rows
                          if r["stage"] == stage and r["status"] == "ok"]
                if values:
                    writer.writerow(dict(zip(summary_fields, (
                        stage, len(values), sum(values), statistics.median(values),
                        min(values), max(values),
                    ))))


def main(argv=None) -> int:
    if not (PROJECT_DIRECTORY / "ect_analysis" / "pipeline.py").is_file():
        raise RuntimeError("Place benchmark_revision.py beside the project's main.py, "
                           "and use the project's existing Python environment.")
    from ect_analysis import pipeline
    from ect_analysis.cli import parse_args
    import numpy as np
    import scipy

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv:
        args = parse_args(argv)
        config_mode = "original_command_line_parser"
    else:
        import main as project_main
        if project_main.USE_PYCHARM_CONFIG:
            args = project_main.pycharm_arguments()
            config_mode = "main.py PYCHARM_CONFIG"
        else:
            args = parse_args([])
            config_mode = "original_command_line_defaults"
    explicit_output = any(a == "--output-dir" or a.startswith("--output-dir=") for a in argv)
    base = args.output_dir if explicit_output else PROJECT_DIRECTORY / "revision_benchmark_results"
    stamp = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    output = (base / stamp).resolve()
    output.mkdir(parents=True, exist_ok=False)
    args.output_dir = output

    blas_config = io.StringIO()
    with redirect_stdout(blas_config):
        print("NumPy build configuration:")
        np.show_config()
        print("SciPy build configuration:")
        scipy.show_config()
    try:
        from threadpoolctl import threadpool_info
        threadpools = threadpool_info()
    except ImportError:
        threadpools = None
    environment = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration_source": config_mode, "parameters": vars(args),
        "platform": platform.platform(), "python": platform.python_version(),
        "cpu_model": cpu_model(), "logical_cpu_count": os.cpu_count(),
        "physical_ram_bytes": physical_ram_bytes(),
        "blas_build_configuration": blas_config.getvalue(),
        "threadpools_if_available": threadpools,
        "thread_environment": {name: os.environ.get(name) for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "source_sha256": source_hashes(),
        "timing_scopes": dict(zip(STAGES, (
            "Native HDF5 read, centroids, volumes, bin assignment and sparse aggregation.",
            "Full-set noise scaling, Gram construction, symmetrisation and all eigenvalues.",
            "Greedy selection using a precomputed Gram matrix; excludes final subset scorecards.",
            "Phantom routine: Gram/eigenvalue recomputation, synthetic data and noise, "
            "Cholesky factorisation, batched inversion and image metrics.",
        ))),
        "notes": [
            "Peak RSS is the cumulative lifetime peak for the entire Python process, "
            "including imports, all configurations and native-library allocations.",
            "No isolated per-stage memory peaks are claimed.",
            "This is one execution. Disk-cache state is not controlled.",
            "Total pipeline time includes auxiliary metrics, figures, statistics, "
            "report generation and SHA-256 input hashing, so it exceeds the timed-stage sum.",
            "The main pipeline's run_manifest.json contains exact input hashes and software versions.",
        ],
    }
    profiler = StageProfiler(pipeline)
    start = time.perf_counter()
    failure = None
    code = 1
    print(f"Benchmark output: {output}", flush=True)
    try:
        with profiler:
            code = pipeline.run_analysis(args_override=args, entrypoint_path=Path(__file__))
        return int(code)
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        environment["pipeline_elapsed_seconds"] = time.perf_counter() - start
        environment["exit_code"] = code
        environment["error"] = failure
        peak, method = process_peak_rss()
        environment["peak_process_rss_bytes"] = peak
        environment["peak_process_rss_MB"] = None if peak is None else peak / 1e6
        environment["peak_process_rss_method"] = method
        environment["has_stage_measurements"] = bool(profiler.rows)
        expected_labels = [chr(65 + i) for i in range(15)]
        environment["baseline_dimensions_expected"] = {
            "configurations": expected_labels, "common_parameters": 2851,
            "measurement_channels": 496, "selected_channels": 128,
            "target_positions": 40, "reconstruction_vectors_per_configuration": 200,
        }
        environment["baseline_dimensions_match_manuscript"] = (
            sorted(profiler.mapped_labels) == expected_labels
            and all(sum(r["stage"] == stage and r["status"] == "ok"
                        for r in profiler.rows) == 15 for stage in STAGES)
            and all(r["measurement_channels"] == 496 for r in profiler.rows)
            and all(r.get("common_parameters", 2851) == 2851 for r in profiler.rows)
            and all(r.get("selected_channels", 128) == 128 for r in profiler.rows)
            and all(r.get("target_positions", 40) == 40 for r in profiler.rows)
            and all(r.get("reconstruction_vectors", 200) == 200 for r in profiler.rows)
        )
        manifest_path = output / "run_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            environment["original_run_manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            environment["study_common_grid"] = manifest.get("common_grid")
            environment["study_regularization"] = manifest.get("regularization")
            environment["study_software"] = manifest.get("software")
            environment["study_inputs"] = manifest.get("inputs")
        profiler.save(output)
        with (output / "benchmark_environment.json").open("w", encoding="utf-8") as stream:
            json.dump(environment, stream, indent=2, ensure_ascii=False, default=json_ready)
        print(f"Benchmark files saved in: {output}", flush=True)
        if not profiler.rows:
            print("No FEM stages were measured (for example, --self-test only).", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
