"""Hash-bound initial execution equality, including all failures without filtering."""
import json
from pathlib import Path
import numpy as np
from evaluate_historical_dense import ROOT, sha, write


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    stage = ROOT/'experimental_supplement_20260923/paired_seed2_initial_recovery/seed_923202/attempt_2'
    target = stage/'initial_execution_equivalence.json'
    assert not target.exists()
    bindings = {str(Path(__file__)): sha(__file__)}
    records = []
    for case in ['original', 'dense']:
        for key in ['1ms', 'half_ms']:
            traces, outcomes = [], []
            for arm in ['two', 'three']:
                folder = stage/arm
                assert read(folder/'evaluation/replay_receipt.json')['all_four_evaluations_recorded']
                path = folder/f'evaluation/eval_0000_{case}_{key}.npz'
                outcome = read(path.with_suffix('.json'))
                assert outcome['trace_sha256'] == sha(path)
                assert outcome['checkpoint_sha256'] == sha(folder/'initial.pt')
                with np.load(path) as data:
                    traces.append({k:data[k].copy() for k in ['rows','observations','actions','decision_times_s']})
                outcomes.append(outcome)
                for p in [path, path.with_suffix('.json'), folder/'initial.pt']:
                    bindings[str(p)] = sha(p)
            a, b = traces
            equal = (np.array_equal(a['observations'], b['observations']) and
                     np.array_equal(a['actions'][:, :2], b['actions'][:, :2]) and
                     np.array_equal(a['decision_times_s'], b['decision_times_s']))
            diff = abs(a['rows']-b['rows']).max(0) if a['rows'].shape == b['rows'].shape else None
            records.append(dict(case=case, resolution=key,
                success=[o['success'] for o in outcomes], times=[o['complete_time_s'] for o in outcomes],
                reasons=[o['reason'] for o in outcomes],
                input_and_first_two_latent_equal=equal,
                maximum_row_difference=None if diff is None else diff.tolist()))
    passed = all(all(r['success']) and r['input_and_first_two_latent_equal'] and
        r['times'][0] == r['times'][1] and r['maximum_row_difference'] is not None and
        max(r['maximum_row_difference'][:6]) < 1e-6 for r in records)
    write(target, dict(records=records, source_sha256=bindings,
        all_four_complete_success=all(all(r['success']) for r in records),
        initial_pair_gate_pass=passed, pair_training_not_started=True))
    print('INITIAL_PAIR_GATE', passed, [(r['case'],r['resolution'],r['success'],r['times']) for r in records])


if __name__ == '__main__':
    main()
