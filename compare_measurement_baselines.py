#!/usr/bin/env python3
"""Compare reduced-measurement strategies on the original shared ECT basis.

Place beside main.py and run in the project's existing Python environment:

    python compare_measurement_baselines.py

With no arguments, settings come from main.py's PYCHARM_CONFIG. Explicit
arguments use the original project's parser. The helper runs the original
global/subset analysis, adds comparison tables, and preserves all original
numerical routines and input files. It supports the v1.0.0 pipeline.

Five selection methods use identical common parameters, channel count,
scalar noise scaling and the original shared regularisation parameter:
  d_optimal: the project's original greedy regularised log-determinant routine;
  highest_energy: largest diagonal entries of the full channel Gram matrix;
  low_coherence: highest-energy seed, then minimum maximum absolute normalised
      correlation with selected channels, breaking ties by energy and index;
      exactly zero-sensitivity channels are deferred until needed to fill k;
  pivoted_qr: first k column pivots of QR of the noise-scaled sensitivity
      matrix (SciPy, mode='r', pivoting=True; Q is not explicitly generated);
  random: uniform selection without replacement, repeated RANDOM_DRAWS times.

The same random sets are used across configurations, allowing paired geometry
comparisons. Random-set quantiles describe the distribution over draws, not
confidence intervals for a mean. Deterministic selection timing is repeated;
all repeats must return the same ordered channel indices.

Selection timers exclude shared mapping/Gram construction and score evaluation.
QR's own factorisation is included. These are marginal selection costs within
this common-basis analysis, not standalone runtimes starting from native data.
Reconstruction, model-mismatch and acquisition-hardware tests are not included.

Outputs in a new folder under baseline_comparison_results:
  baseline_scorecard.csv: individual sets, metric values, retention and timings;
  baseline_summary.csv: per-configuration summaries, including random quantiles;
  baseline_selected_channels.csv: every selected electrode pair/index;
  baseline_preprocessing.csv: shared Gram/eigenvalue timings;
  baseline_comparison_manifest.json: settings, methods, scopes and provenance.
The original pipeline's tables, figures and run_manifest.json are also retained.
Use --self-test for installation and numerical checks without .mat inputs.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import qr

PROJECT_DIRECTORY = Path(__file__).resolve().parent
RANDOM_DRAWS = 100
TIMING_REPEATS = 5


def highest_energy(gram, k):
    energy = np.diag(gram)
    indices = np.arange(len(energy))
    return np.lexsort((indices, -energy))[:k].tolist()


def low_coherence(gram, k):
    energy = np.maximum(np.diag(gram), 0.0)
    norms = np.sqrt(energy)
    denominator = np.outer(norms, norms)
    correlation = np.zeros_like(gram, dtype=float)
    np.divide(np.abs(gram), denominator, out=correlation, where=denominator > 0)
    np.clip(correlation, 0.0, 1.0, out=correlation)
    available = np.ones(len(energy), dtype=bool)
    maximum_correlation = np.zeros(len(energy))
    selected = []
    for _ in range(k):
        remaining = np.flatnonzero(available)
        positive = remaining[energy[remaining] > 0]
        candidates = positive if len(positive) else remaining
        order = np.lexsort((candidates, -energy[candidates], maximum_correlation[candidates]))
        best = int(candidates[order[0]])
        selected.append(best)
        available[best] = False
        maximum_correlation = np.maximum(maximum_correlation, correlation[best])
    return selected


def pivoted_qr(sensitivity, noise_std, k):
    _, pivots = qr(sensitivity / noise_std, mode='r', pivoting=True, check_finite=False)
    return pivots[:k].astype(int).tolist()


def check_selection(selected, n_channels, k):
    if len(selected) != k or len(set(selected)) != k:
        raise ValueError('A method did not return k distinct channels.')
    if any(not 0 <= int(channel) < n_channels for channel in selected):
        raise ValueError('A selected channel is outside the matrix.')


def eigenvalues_from_gram(gram):
    values = np.linalg.eigvalsh(0.5 * (gram + gram.T))
    floor = 100.0 * np.finfo(float).eps * max(float(np.max(np.abs(values))), 1.0)
    if np.any(values < -floor):
        raise ValueError('Subset Gram matrix has materially negative eigenvalues.')
    return np.maximum(values, 0.0)


def repeated_selection(function, repeats=TIMING_REPEATS, first=None):
    times = [] if first is None else [first[1]]
    selected = None if first is None else list(first[0])
    for _ in range(repeats - len(times)):
        start = time.perf_counter()
        current = list(function())
        times.append(time.perf_counter() - start)
        if selected is None:
            selected = current
        elif current != selected:
            raise RuntimeError('Deterministic selection changed between timing repeats.')
    return selected, {
        'selection_seconds_mean': statistics.mean(times),
        'selection_seconds_median': statistics.median(times),
        'selection_seconds_min': min(times),
        'selection_seconds_max': max(times),
        'selection_timing_repeats': len(times),
    }


def score_selection(sensitivity, gram, eigenvalues, alpha, noise_std, selected):
    from ect_analysis.metrics import regularized_spectral_metrics, spatial_metrics

    full_information = float(np.log1p(eigenvalues / alpha).sum())
    full_energy = float(eigenvalues.sum())
    if full_information <= 0 or full_energy <= 0:
        raise ValueError('The complete sensitivity operator has no positive information/energy.')
    subset = sensitivity[:, selected]
    sub_gram = gram[np.ix_(selected, selected)]
    sub_values = eigenvalues_from_gram(sub_gram)
    spectral = regularized_spectral_metrics(subset, alpha, noise_std, sub_gram, sub_values)
    # The actual common bins are equal-volume: equal weights give the same
    # spatial summaries as the original bin-volume weights.
    spatial, _, _ = spatial_metrics(subset, noise_std, np.ones(subset.shape[0]))
    output = {name: spectral[name] for name in (
        'regularized_logdet', 'spectral_energy', 'effective_rank_energy',
        'stable_rank', 'numerical_rank', 'coherence', 'largest_eigenvalue',
    )}
    output.update(spatial)
    output['regularized_information_retention'] = spectral['regularized_logdet'] / full_information
    output['spectral_energy_retention'] = spectral['spectral_energy'] / full_energy
    for key in ('regularized_information_retention', 'spectral_energy_retention'):
        if not -1e-8 <= output[key] <= 1.0 + 1e-8:
            raise FloatingPointError(f'Invalid retention value for {key}: {output[key]}')
    return output


class BaselineComparison:
    def __init__(self, pipeline, seed, random_draws=RANDOM_DRAWS, timing_repeats=TIMING_REPEATS):
        self.pipeline = pipeline
        self.seed = int(seed)
        self.random_draws = int(random_draws)
        self.timing_repeats = int(timing_repeats)
        if self.random_draws <= 0 or self.timing_repeats <= 0:
            raise ValueError('Draws and timing repeats must be positive.')
        self.labels = []
        self.spectral_calls = 0
        self.data_by_gram_id = {}
        self.originals = {}
        self.rows = []
        self.selected_rows = []
        self.preprocessing = []
        self.channel_definitions = {}
        self.primary_alpha = None
        self.n_channels = None
        self.k = None

    def _record(self, label, method, draw, sensitivity, gram, eigenvalues, alpha,
                noise_std, selected, timings):
        n_channels = gram.shape[0]
        check_selection(selected, n_channels, len(selected))
        metrics = score_selection(sensitivity, gram, eigenvalues, alpha, noise_std, selected)
        row = {
            'configuration': label, 'method': method, 'draw_index': draw,
            'common_parameters': sensitivity.shape[0], 'available_channels': n_channels,
            'selected_channels': len(selected), 'alpha_common': float(alpha),
            'score_noise_std': float(noise_std), **metrics, **timings,
        }
        self.rows.append(row)
        from ect_analysis.channels import channel_table
        if n_channels not in self.channel_definitions:
            self.channel_definitions[n_channels] = channel_table(n_channels).set_index(
                'channel_index_zero_based').to_dict(orient='index')
        lookup = self.channel_definitions[n_channels]
        for order, channel in enumerate(selected, 1):
            self.selected_rows.append({
                'configuration': label, 'method': method, 'draw_index': draw,
                'selection_order': order, 'channel_index_zero_based': channel,
                **lookup[channel],
            })
        return row

    def compare(self, label, sensitivity, gram, eigenvalues, noise_std, alpha, k,
                original_result, original_seconds):
        k = len(original_result[0])
        if self.primary_alpha is None:
            self.primary_alpha = float(alpha)
            self.n_channels, self.k = gram.shape[0], k
        if not np.isclose(alpha, self.primary_alpha, rtol=1e-14, atol=0):
            raise ValueError('The pipeline no longer uses one shared alpha.')
        if (gram.shape[0], k) != (self.n_channels, self.k):
            raise ValueError('Configurations do not use identical available/subset channel counts.')
        check_selection(original_result[0], self.n_channels, k)
        empty_timing = {key: np.nan for key in (
            'selection_seconds_mean', 'selection_seconds_median',
            'selection_seconds_min', 'selection_seconds_max',
        )}
        empty_timing['selection_timing_repeats'] = 0
        self._record(label, 'all_channels', 0, sensitivity, gram, eigenvalues, alpha,
                     noise_std, list(range(self.n_channels)), empty_timing)
        original = self.originals['greedy_regularized_logdet']
        strategies = {
            'd_optimal': lambda: original(gram, alpha, k)[0],
            'highest_energy': lambda: highest_energy(gram, k),
            'low_coherence': lambda: low_coherence(gram, k),
            'pivoted_qr': lambda: pivoted_qr(sensitivity, noise_std, k),
        }
        for method, selector in strategies.items():
            first = (original_result[0], original_seconds) if method == 'd_optimal' else None
            selected, timings = repeated_selection(selector, self.timing_repeats, first)
            check_selection(selected, self.n_channels, k)
            row = self._record(label, method, 0, sensitivity, gram, eigenvalues, alpha,
                               noise_std, selected, timings)
            if method == 'd_optimal' and not np.isclose(
                    row['regularized_logdet'], original_result[1][-1], rtol=1e-7, atol=1e-6):
                raise FloatingPointError('Subset log-determinant disagrees with the original greedy curve.')
        # Reset to the same seed for each configuration: the random sets are paired.
        rng = np.random.default_rng(self.seed)
        for draw in range(1, self.random_draws + 1):
            start = time.perf_counter()
            selected = rng.choice(self.n_channels, size=k, replace=False).astype(int).tolist()
            elapsed = time.perf_counter() - start
            timings = {
                'selection_seconds_mean': elapsed, 'selection_seconds_median': elapsed,
                'selection_seconds_min': elapsed, 'selection_seconds_max': elapsed,
                'selection_timing_repeats': 1,
            }
            self._record(label, 'random', draw, sensitivity, gram, eigenvalues, alpha,
                         noise_std, selected, timings)
        print(f'  BASELINES {label}: 4 deterministic methods and {self.random_draws} random sets completed.', flush=True)

    def __enter__(self):
        for name in ('map_to_common_grid', 'gram_and_eigenvalues', 'greedy_regularized_logdet'):
            self.originals[name] = getattr(self.pipeline, name)

        @functools.wraps(self.originals['map_to_common_grid'])
        def mapped(meta, *args, **kwargs):
            result = self.originals['map_to_common_grid'](meta, *args, **kwargs)
            self.labels.append(meta.label)
            return result

        @functools.wraps(self.originals['gram_and_eigenvalues'])
        def spectral(sensitivity, noise_std):
            if self.spectral_calls >= len(self.labels):
                raise RuntimeError('Unexpected Gram call order; this helper supports v1.0.0.')
            label = self.labels[self.spectral_calls]
            self.spectral_calls += 1
            start = time.perf_counter()
            gram, eigenvalues = self.originals['gram_and_eigenvalues'](sensitivity, noise_std)
            elapsed = time.perf_counter() - start
            self.data_by_gram_id[id(gram)] = (label, sensitivity, gram, eigenvalues, noise_std)
            self.preprocessing.append({
                'configuration': label, 'common_parameters': sensitivity.shape[0],
                'measurement_channels': sensitivity.shape[1],
                'shared_gram_eigenvalues_seconds': elapsed,
            })
            return gram, eigenvalues

        @functools.wraps(self.originals['greedy_regularized_logdet'])
        def selected(gram, alpha, k, candidate_mask=None):
            if candidate_mask is not None:
                raise ValueError('This baseline comparison assumes unrestricted selection.')
            start = time.perf_counter()
            result = self.originals['greedy_regularized_logdet'](gram, alpha, k)
            elapsed = time.perf_counter() - start
            if id(gram) not in self.data_by_gram_id:
                raise RuntimeError('Selected Gram matrix does not match a captured full operator.')
            label, sensitivity, _, eigenvalues, noise_std = self.data_by_gram_id[id(gram)]
            self.compare(label, sensitivity, gram, eigenvalues, noise_std, alpha, k, result, elapsed)
            return result

        self.pipeline.map_to_common_grid = mapped
        self.pipeline.gram_and_eigenvalues = spectral
        self.pipeline.greedy_regularized_logdet = selected
        return self

    def __exit__(self, *exc):
        for name, function in self.originals.items():
            setattr(self.pipeline, name, function)

    def save(self, output):
        frame = pd.DataFrame(self.rows)
        frame.to_csv(output / 'baseline_scorecard.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(self.selected_rows).to_csv(
            output / 'baseline_selected_channels.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(self.preprocessing).to_csv(
            output / 'baseline_preprocessing.csv', index=False, encoding='utf-8-sig')
        metrics = (
            'regularized_information_retention', 'spectral_energy_retention',
            'effective_rank_energy', 'stable_rank', 'numerical_rank', 'coherence',
            'local_norm_p50', 'local_norm_p90', 'local_norm_p99',
            'envelope_gini', 'weak_volume_5pct_e99', 'weak_volume_10pct_e99',
            'selection_seconds_mean',
        )
        summaries = []
        if len(frame):
            for (label, method), group in frame.groupby(['configuration', 'method'], sort=True):
                summary = {'configuration': label, 'method': method, 'draws': len(group),
                           'selected_channels': int(group['selected_channels'].iloc[0])}
                for metric in metrics:
                    values = group[metric].dropna().to_numpy(dtype=float)
                    summary[f'{metric}_mean'] = float(values.mean()) if len(values) else np.nan
                    summary[f'{metric}_std'] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
                    for name, fraction in [('q025', .025), ('median', .5), ('q975', .975)]:
                        summary[f'{metric}_{name}'] = float(np.quantile(values, fraction)) if len(values) else np.nan
                summaries.append(summary)
        pd.DataFrame(summaries).to_csv(output / 'baseline_summary.csv', index=False, encoding='utf-8-sig')


def numerical_self_test():
    from ect_analysis.channels import greedy_regularized_logdet
    from ect_analysis.metrics import gram_and_eigenvalues
    from ect_analysis.selftest import run_self_test

    run_self_test()
    # Orthogonal equal-energy channels have analytically identical subset scores.
    sensitivity = np.eye(8)
    gram, eigenvalues = gram_and_eigenvalues(sensitivity, 1.0)
    for selected in (highest_energy(gram, 3), low_coherence(gram, 3),
                     pivoted_qr(sensitivity, 1.0, 3),
                     greedy_regularized_logdet(gram, .2, 3)[0]):
        check_selection(selected, 8, 3)
        scores = score_selection(sensitivity, gram, eigenvalues, .2, 1.0, selected)
        assert np.isclose(scores['regularized_information_retention'], 3 / 8)
        assert np.isclose(scores['spectral_energy_retention'], 3 / 8)
        assert np.isclose(scores['effective_rank_energy'], 3.0)
        assert scores['coherence'] == 0.0
    # Duplicate strong channels test energy versus independent information.
    sensitivity = np.array([[10., 10., 0., 0.], [0., 0., 3., 0.],
                            [0., 0., 0., 0.], [0., 0., 0., 0.]])
    gram, eigenvalues = gram_and_eigenvalues(sensitivity, 1.0)
    energy_set = highest_energy(gram, 2)
    assert energy_set == [0, 1]
    energy_score = score_selection(sensitivity, gram, eigenvalues, 1., 1., energy_set)
    for selected in (low_coherence(gram, 2), pivoted_qr(sensitivity, 1., 2),
                     greedy_regularized_logdet(gram, 1., 2)[0]):
        assert 2 in selected and 3 not in selected
        scores = score_selection(sensitivity, gram, eigenvalues, 1., 1., selected)
        assert np.isclose(scores['regularized_logdet'], np.log(1010.))
        assert scores['regularized_logdet'] > energy_score['regularized_logdet']
        assert scores['spectral_energy'] < energy_score['spectral_energy']
        scaled_gram, scaled_values = gram_and_eigenvalues(4 * sensitivity, 1.)
        scaled = score_selection(4 * sensitivity, scaled_gram, scaled_values, 16., 1., selected)
        assert np.isclose(scores['regularized_information_retention'], scaled['regularized_information_retention'])
    assert low_coherence(gram, 4)[-1] == 3
    print('BASELINE SELF-TEST PASSED', flush=True)


def main(argv=None):
    if not (PROJECT_DIRECTORY / 'ect_analysis' / 'pipeline.py').is_file():
        raise RuntimeError('Place compare_measurement_baselines.py beside main.py.')
    from ect_analysis import pipeline
    from ect_analysis.cli import parse_args
    from ect_analysis.reporting import json_ready

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv:
        args = parse_args(argv)
        mode = 'original_command_line_parser'
    else:
        import main as project_main
        if project_main.USE_PYCHARM_CONFIG:
            args = project_main.pycharm_arguments()
            mode = 'main.py PYCHARM_CONFIG'
        else:
            args = parse_args([])
            mode = 'original_command_line_defaults'
    if args.self_test:
        numerical_self_test()
        return 0
    source_parameters = dict(vars(args))
    args = argparse.Namespace(**source_parameters)
    args.analyses = 'global,subsets'
    explicit_output = any(a == '--output-dir' or a.startswith('--output-dir=') for a in argv)
    base = args.output_dir if explicit_output else PROJECT_DIRECTORY / 'baseline_comparison_results'
    stamp = datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S_%fZ')
    output = (base / stamp).resolve()
    output.mkdir(parents=True, exist_ok=False)
    args.output_dir = output
    comparison = BaselineComparison(pipeline, args.seed, RANDOM_DRAWS, TIMING_REPEATS)
    metadata = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'configuration_source': mode, 'source_parameters': source_parameters,
        'executed_parameters': vars(args), 'random_draws_per_configuration': RANDOM_DRAWS,
        'deterministic_timing_repeats': TIMING_REPEATS, 'random_seed': args.seed,
        'same_random_sets_across_configurations': True,
        'platform': platform.platform(), 'processor': platform.processor(),
        'logical_processor_count': os.cpu_count(),
        'methods': {
            'd_optimal': 'Original unrestricted greedy regularised D-optimal selector.',
            'highest_energy': 'Top k diagonal Gram entries; ties resolved by increasing channel index.',
            'low_coherence': 'Highest-energy seed, then minimum maximum absolute normalised correlation to selected channels. Ties: larger energy, then lower index. Zero-sensitivity channels deferred.',
            'pivoted_qr': "First k pivots from scipy.linalg.qr(S/noise_std, mode='r', pivoting=True); no explicit Q.",
            'random': 'Uniform sampling without replacement; identical seeded sequence of channel sets for each configuration.',
            'all_channels': 'Complete measurement set used as the retention denominator.',
        },
        'notes': [
            'Information and spectral-energy retention use each configuration\'s full operator as denominator.',
            'All subsets use the original shared alpha, scalar score noise, and occupied common spatial support.',
            'Equal spatial weights are equivalent to the original equal bin-volume weights.',
            'Random q025/q975 are draw-distribution quantiles, not confidence intervals for the mean.',
            'Deterministic timing statistics describe repeats on identical input. Random timings describe one draw per set.',
            'Selection timings exclude shared mapping/Gram/eigenvalues and score evaluation. They are marginal selection costs within this analysis. QR factorisation is included in QR selection.',
            'Repeated selection may benefit from caches. Thread limits are not imposed and disk-cache state is not controlled.',
            'This experiment evaluates spectral/spatial criteria and selection costs. It does not run reconstruction or physical-acquisition tests.',
            'Positive-part clipping and reconstruction-noise assumptions are irrelevant to this operator comparison.',
        ],
    }
    start = time.perf_counter()
    exit_code, error = 1, None
    print(f'Baseline comparison output: {output}', flush=True)
    try:
        with comparison:
            exit_code = pipeline.run_analysis(args_override=args, entrypoint_path=Path(__file__))
        return int(exit_code)
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        metadata['complete_analysis_elapsed_seconds'] = time.perf_counter() - start
        metadata['exit_code'], metadata['error'] = exit_code, error
        metadata['completed_configurations'] = sorted({r['configuration'] for r in comparison.rows
                                                       if r['method'] == 'random' and r['draw_index'] == RANDOM_DRAWS})
        metadata['completed_subset_sets'] = sum(r['method'] != 'all_channels' for r in comparison.rows)
        comparison.save(output)
        manifest_path = output / 'run_manifest.json'
        if manifest_path.is_file():
            original_bytes = manifest_path.read_bytes()
            original = json.loads(original_bytes.decode('utf-8'))
            metadata['original_run_manifest_sha256'] = hashlib.sha256(original_bytes).hexdigest()
            for key in ('common_grid', 'regularization', 'inputs', 'software'):
                metadata[key] = original.get(key)
        paths = [Path(__file__), PROJECT_DIRECTORY / 'main.py']
        paths.extend(sorted((PROJECT_DIRECTORY / 'ect_analysis').glob('*.py')))
        metadata['source_sha256'] = {
            p.relative_to(PROJECT_DIRECTORY).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.is_file()
        }
        metadata['baseline_dimensions_match_manuscript'] = (
            exit_code == 0 and error is None
            and metadata['completed_configurations'] == list('ABCDEFGHIJKLMNO')
            and comparison.n_channels == 496 and comparison.k == 128
            and metadata.get('common_grid', {}).get('common_occupied_bins') == 2851
            and len(comparison.rows) == 15 * (RANDOM_DRAWS + 5)
        )
        with (output / 'baseline_comparison_manifest.json').open('w', encoding='utf-8') as stream:
            json.dump(metadata, stream, indent=2, ensure_ascii=False, default=json_ready)
        print(f'Baseline comparison files saved in: {output}', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
