"""Attribute actual joint-composition rejection to unchanged physical metrics."""
import argparse
import numpy as np
from duration_joint_metrics import guard_names
from search_complete_return import read
from study_io import BASE, write_json, sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',default='canonical_gate_band_coupled_context_composition_learning_20260920')
    stage = BASE / parser.parse_args().stage
    history = read(stage / 'state_joint/results.json')['candidates']
    zero = history[0]
    paths = [r['path'] for r in zero['records']]
    old_names = ('incoming', 'outgoing', 'reverse', 'energy', 'peak', 'apex_speed',
                 'time_old', 'cruise_old', 'first_turn', 'corridor', 'lateral')
    names = [f'{p}:{n}' for p in paths for n in old_names]
    names += [f'{p}:{n}' for p in paths for n in guard_names()]
    rows = []
    for arm in read(stage / 'physical_phase_interaction.json')['arms']:
        c = next(c for c in history if c['tag'] == arm['tag'])
        values = np.asarray(c['measured_inequalities'])
        assert len(values) == len(names)
        differences = []
        for r, b in zip(c['records'], zero['records']):
            delta = {}
            for group in ('shape', 'motion', 'duration_audit'):
                for key in b[group]:
                    if isinstance(b[group][key], (float, int)) and isinstance(r[group].get(key), (float, int)):
                        change = r[group][key] - b[group][key]
                        if change:
                            delta[f'{group}.{key}'] = change
            if delta:
                differences.append(dict(path=r['path'], change=delta))
        row = dict(arm=arm['arm'], loss_change=c['score']-zero['score'],
                   failures=[dict(name=n, normalized=float(v)) for n, v in zip(names, values) if v < -1e-8],
                   changes=differences)
        rows.append(row)
        print(row['arm'], 'loss_change', row['loss_change'], 'failure_count',len(row['failures']),
              'largest_failures',sorted(row['failures'],key=lambda r:r['normalized'])[:5], flush=True)
    write_json(stage / 'actual_guard_attribution.json', dict(
        source_sha256=sha(stage / 'state_joint/results.json'), new_physical_runs=0,
        acceptance_thresholds_unchanged=True, arms=rows))


if __name__ == '__main__':
    main()
