"""Build complete auxiliary tables from frozen, hash-checked evaluation records."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

import preview_unified_baselines as gate
import preview_saved_artifacts as archived

CONDITIONS = (('test', 'nominal'), ('regression', 'nominal'),
              ('test', 'tight'), ('test', 'low_jerk'))
METHODS = ('absolute', 'nnc', 'two_step')
LABELS = {'absolute': 'V2 absolute', 'nnc': 'NNC', 'two_step': 'Two-step'}
PATH_LABELS = {'mixed_lengths_v2_test': 'Mixed lengths',
               'dense_corners_v2_test': 'Dense corners',
               'line_curve_v2_test': 'Line + curve',
               'square': 'Square', 'circle': 'Circle', 'butterfly': 'Butterfly'}


def summarize(records):
    """Count every run; condition time statistics explicitly on strict success."""
    values = [record['metrics'] for record in records]
    successful = [value for value in values if value['success']]
    times = [value['elapsed_time_s'] for value in successful]
    failures = Counter(reason for value in values for reason in value['failure_reasons'])
    dynamics = sum(any(c['satisfied'] is False for c in value['constraints'].values())
                   for value in values)
    return dict(n=len(values), completed=sum(value['path_completed'] for value in values),
                success=len(successful), failures=len(values) - len(successful),
                dynamic_violation_runs=dynamics,
                success_time_n=len(times), success_time_mean_s=float(np.mean(times)) if times else None,
                success_time_sd_s=float(np.std(times, ddof=1)) if len(times) > 1 else None,
                all_runs_peak_error_max_mm=max(value['contour_error_max_mm'] for value in values),
                all_runs_elapsed_mean_s=float(np.mean([value['elapsed_time_s'] for value in values])),
                all_runs_terminal_feed_min_mm_s=min(value['terminal_feed_mm_s'] for value in values),
                all_runs_terminal_feed_max_mm_s=max(value['terminal_feed_mm_s'] for value in values),
                terminal_rest_checked_runs=sum(value['terminal_rest_checked'] for value in values),
                failure_reason_counts=dict(sorted(failures.items())),
                successful_seeds=[record.get('seed') for record in records if record['metrics']['success']])


def checked_file(path, expected, sources):
    actual = gate.sha(path)
    if actual != expected:
        raise ValueError(f'Artifact hash mismatch: {path}')
    sources[path.relative_to(gate.ROOT).as_posix()] = actual


def collect(candidate):
    _, final_runs = archived.final_saved(candidate, gate.v2.DATA / 'manifest.json')
    sources = {}
    legacy = {(item['baseline'], item['seed']): item for item in archived.legacy_saved(gate.v2.OUT, gate.SEEDS, gate.v2.DATA / 'manifest.json')}
    for item in [*final_runs.values(), *legacy.values()]:
        for name in ['provenance.json', 'training_summary.json', 'source.zip']:
            path = item['run'] / name
            sources[path.relative_to(gate.ROOT).as_posix()] = gate.sha(path)
    for path in (candidate.OUT / 'freeze.json', candidate.OUT / 'protocol.json'):
        sources[path.relative_to(gate.ROOT).as_posix()] = gate.sha(path)
    manifest_path = gate.v2.DATA / 'manifest.json'
    sources[manifest_path.relative_to(gate.ROOT).as_posix()] = gate.sha(manifest_path)
    for path in gate.reference_specs('test'):
        checked_file(gate.v2.DATA / path['reference_file'], path['reference_sha256'], sources)
    history_path = candidate.OUT / 'regression_reference_provenance.json'
    history = gate.read(history_path)
    if history['protocol_sha256'] != gate.sha(candidate.OUT / 'protocol.json'):
        raise ValueError('Historical references belong to another registered protocol')
    historical_config = gate.ROOT / '论文项目/generated/configs/full_method_snapshot.yaml'
    checked_file(historical_config, history['historical_config_sha256'], sources)
    checked_file(candidate.OUT / 'regression_references/full_method_snapshot.yaml', history['historical_config_sha256'], sources)
    sources[history_path.relative_to(gate.ROOT).as_posix()] = gate.sha(history_path)
    for item in history['references']:
        checked_file(candidate.OUT / item['file'], item['sha256'], sources)
    rows = []
    for split, regime in CONDITIONS:
        expected_paths = [path['id'] for path in gate.reference_specs(split)]
        for method in METHODS:
            seeds = (None,) if method == 'two_step' else gate.SEEDS
            for seed in seeds:
                if method == 'two_step':
                    folder = candidate.OUT / 'two_step' / f'{split}_{regime}'
                else:
                    folder = candidate.OUT / 'unified_baselines' / gate.OUTPUT_VERSION / f'evaluate_{split}_{regime}' / f'{method}_seed{seed}'
                protocol_path = folder / 'protocol.json'
                protocol = gate.read(protocol_path)
                manifest_key = 'manifest_sha256' if method == 'two_step' else 'path_manifest_sha256'
                if protocol[manifest_key] != gate.sha(manifest_path):
                    raise ValueError('Auxiliary protocol reference manifest mismatch')
                if method != 'two_step' and protocol['final_freeze_sha256'] != gate.sha(candidate.OUT / 'freeze.json'):
                    raise ValueError('Auxiliary run belongs to another final freeze')
                if method != 'two_step':
                    original = legacy[(method, seed)]
                    if split == 'regression' and protocol['historical_path_config_sha256'] != history['historical_config_sha256']:
                        raise ValueError('Auxiliary historical configuration differs from registered references')
                    for name, expected in [('checkpoint.pt', original['checkpoint_sha256']),
                                           ('config.yaml', original['config_sha256']),
                                           ('source.zip', original['source_archive_sha256'])]:
                        checked_file(original['run'] / name, expected, sources)
                    if (protocol['checkpoint_sha256'] != original['checkpoint_sha256']
                            or protocol['original_config_sha256'] != original['config_sha256']
                            or protocol['original_source_archive_sha256'] != original['source_archive_sha256']
                            or protocol['original_training_summary'] != original['training']
                            or protocol['original_training_provenance'] != original['provenance']):
                        raise ValueError('Auxiliary protocol model/config/source archive is not the frozen legacy evidence')
                for path_id in expected_paths:
                    path = folder / path_id / 'result.json' if method == 'two_step' else folder / f'{path_id}.json'
                    record = gate.read(path)
                    if (record['path_id'], record['split'], record['regime']) != (path_id, split, regime):
                        raise ValueError('Unexpected path/split/regime in auxiliary matrix')
                    if method != 'two_step' and (record['baseline'], record['seed']) != (method, seed):
                        raise ValueError('Unexpected method/seed in auxiliary matrix')
                    if method != 'two_step':
                        if record['checkpoint_sha256'] != original['checkpoint_sha256']:
                            raise ValueError('Auxiliary result checkpoint is not the frozen legacy model')
                        checked_file(folder / 'effective_config.yaml', record['effective_config_sha256'], sources)
                    checked_file(protocol_path, record['protocol_sha256'], sources)
                    trace = path.parent / 'trace.csv' if method == 'two_step' else folder / f'{path_id}.npz'
                    checked_file(trace, record['raw_trace_sha256'], sources)
                    sources[path.relative_to(gate.ROOT).as_posix()] = gate.sha(path)
                    rows.append(dict(method=method, seed=seed, split=split, regime=regime,
                                     path_id=path_id, result_file=path.relative_to(gate.ROOT).as_posix(),
                                     result_sha256=gate.sha(path), metrics=record['metrics']))
    return rows, sources


def time_cell(summary):
    if not summary['success_time_n']:
        return '--'
    mean = summary['success_time_mean_s']
    sd = summary['success_time_sd_s']
    return f'{mean:.3f}' if sd is None else rf'${mean:.3f}\pm{sd:.3f}$'


def render_table(groups, selected, label, caption):
    lines = [r'\begin{table}[htbp]', r'\centering', r'\small',
             r'\caption{' + caption + '}', r'\label{' + label + '}',
             r'\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{llrrrrrr}',
             r'\toprule', r'Path & Method & $n$ & $C$ & $S$ & $V$ & $E_{\max}$ (mm) & $T_S$ (s) \\',
             r'\midrule']
    for split, regime in selected:
        title = {'test_nominal': 'Held-out test; nominal', 'regression_nominal': 'Historical regression; nominal',
                 'test_tight': 'Held-out test; tighter tolerance', 'test_low_jerk': 'Held-out test; lower scalar jerk'}[f'{split}_{regime}']
        lines.append(r'\multicolumn{8}{l}{\textit{' + title + r'}} \\')
        subset = [group for group in groups if (group['split'], group['regime']) == (split, regime)]
        for group in subset:
            s = group['summary']
            lines.append(' & '.join([PATH_LABELS[group['path_id']], LABELS[group['method']], str(s['n']),
                                    str(s['completed']), str(s['success']), str(s['dynamic_violation_runs']),
                                    f"{s['all_runs_peak_error_max_mm']:.3f}", time_cell(s)]) + r' \\')
        lines.append(r'\midrule')
    lines[-1] = r'\bottomrule'
    lines += [r'\end{tabular}', r'\end{table}']
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=['direct-explore'], default='direct-explore')
    args = parser.parse_args()
    candidate = gate.final_study(args.study)
    rows, sources = collect(candidate)
    groups = []
    for split, regime in CONDITIONS:
        for path in gate.reference_specs(split):
            for method in METHODS:
                selected = [row for row in rows if (row['split'], row['regime'], row['path_id'], row['method']) == (split, regime, path['id'], method)]
                groups.append(dict(split=split, regime=regime, path_id=path['id'], method=method, summary=summarize(selected)))
    aggregates = [dict(split=split, regime=regime, method=method,
                       summary=summarize([row for row in rows if (row['split'], row['regime'], row['method']) == (split, regime, method)]))
                  for split, regime in CONDITIONS for method in METHODS]
    out = candidate.OUT / 'report' / 'auxiliary'
    out.mkdir(parents=True, exist_ok=True)
    gate.write(out / 'all_results.json', dict(rows=rows, by_path=groups, by_condition=aggregates,
        statistics_note='Independent training seeds and paths are units; trajectory samples are not replications. Time is conditional on each method own strict successes; cross-method success sets may differ.',
        terminal_note='Two-step requests terminal rest; transferred PPO/NNC require ordered arrival without rest. No equivalent-terminal-state time comparison.',
        constraint_scope='Independent scalar feed and heading derivatives only; Cartesian axis signals are diagnostic without supplied limits.',
        comparison_role='Transferred old frozen models and unchanged deterministic two-step planner; not part of the final equal-budget preview contribution estimate.',
        latency_note='Concurrent rollout latency is retained in raw JSON for traceability but is not used as a stable inference benchmark.'))
    csv_rows = []
    for row in rows:
        m = row['metrics']
        csv_rows.append({**{key: row[key] for key in ['method', 'seed', 'split', 'regime', 'path_id', 'result_file', 'result_sha256']},
            **{key: m[key] for key in ['success', 'path_completed', 'ordered_progress', 'geometry_feasible', 'elapsed_time_s', 'contour_error_max_mm', 'contour_error_mean_mm', 'contour_error_rmse_mm', 'terminal_feed_mm_s', 'terminal_rest_checked']},
            'failure_reasons': '|'.join(m['failure_reasons']),
            'violated_constraints': '|'.join(key for key, value in m['constraints'].items() if value['satisfied'] is False)})
    with (out / 'all_runs.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    explanation = (r'$n$ is the number of runs; $C$ counts ordered completion, $S$ counts completion satisfying geometry and audited scalar constraints, and $V$ counts runs with any audited dynamic violation. Failures equal $n-S$. '
        r'$E_{\max}$ is the largest contour error over all runs, including failures. $T_S$ is the mean $\pm$ sample standard deviation over this method\textquotesingle s strict successful runs only; -- means none. '
        r'V2 absolute and NNC each use five frozen historical training seeds with their original state, preview, and guidance settings; two-step is deterministic. No extra training was performed for these transfers. '
        r'Two-step requests terminal rest, whereas PPO/NNC require ordered endpoint arrival without rest. The scalar feed and heading limits are independently audited; no Cartesian axis guarantee is made. These auxiliary transfers are separate from the final equally trained preview comparison.')
    lines = ['% Generated by preview_auxiliary_report.py from 132 complete, hash-checked evaluation records.',
             r'\paragraph{Frozen auxiliary methods under the common evaluation protocol.}', explanation]
    lines += render_table(groups, CONDITIONS[:2], 'tab:final_auxiliary_nominal',
                          'Auxiliary results on the held-out test and historical regression paths. All failures are included; symbols and conditional-time definitions are given in the text.')
    lines += render_table(groups, CONDITIONS[2:], 'tab:final_auxiliary_sensitivity',
                          r'Auxiliary sensitivity checks on the same held-out paths. Tighter tolerance uses a one-sided limit of $0.625$ mm; the lower-jerk condition uses $31{,}250$ mm/s$^3$ with the nominal one-sided limit of $1.25$ mm. No model or control parameter was tuned after test evaluation.')
    tex = gate.ROOT / '论文项目' / 'generated' / 'final_auxiliary_results.tex'
    tex.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    gate.write(out / 'sources.json', dict(source_hashes=sources, generator_sha256=gate.sha(Path(__file__)),
        archived_verifier_sha256=gate.sha(Path(archived.__file__)),
        verification_scope='Read-only source.zip member/freeze/provenance/model/config/data/trace checks; no live-source equality requirement and no policy execution.',
        output_hashes={path.relative_to(gate.ROOT).as_posix(): gate.sha(path) for path in [out / 'all_results.json', out / 'all_runs.csv', tex]},
        run_count=len(rows), group_count=len(groups), command='python ' + ' '.join(__import__('sys').argv)))
    print(json.dumps(aggregates, indent=2))
    print(tex)


if __name__ == '__main__':
    main()
