"""Separate recorded-label fitting error from actual frozen-teacher imitation."""
import json
import numpy as np
import torch
from common import ROOT, sha, utc, write_json
from frozen_policy import FrozenPolicy
from strong_policy_v4 import load_strong_policy


def stats(error):
    return dict(rmse=np.sqrt(np.mean(error**2, axis=0)).tolist(),
                maximum=np.max(np.abs(error), axis=0).tolist(),
                mean=np.mean(error, axis=0).tolist())


def main():
    torch.set_num_threads(1)
    model = ROOT / 'baseline_records/supplementary_reproduction/models/main.pt'
    pilot_file = ROOT / 'results/strong_fit_v4/pilot_seed101/adaptive/final.pt'
    pilot, _ = load_strong_policy(pilot_file)
    teacher = FrozenPolicy(model)
    data_file = ROOT / 'baseline_records/supplementary_reproduction/training/recorded_actions.npz'
    with np.load(data_file) as data:
        x = data['observations']
        labels = data['targets_three']
        with torch.no_grad():
            t = np.vstack([teacher.mean(part).numpy() for part in np.array_split(x, 16)])
            p = np.vstack([pilot.mean(part).numpy() for part in np.array_split(x, 16)])
    base = ROOT / 'results/strong_fit_v4/pilot_seed101/adaptive/evaluation_v4b/original'
    with np.load(base.with_suffix('.npz')) as data:
        tail = data['observations'][-10:]
        points = data['rows'][-200:, :2]
        last = data['rows'][-1].tolist()
        predicted = pilot.mean(tail).numpy()
        frozen = teacher.mean(tail).numpy()
    result = dict(created_utc=utc(), script_sha256=sha(__file__),
        model_sha256=sha(model), pilot_sha256=sha(pilot_file), dataset_sha256=sha(data_file),
        recorded_labels_minus_frozen_final_function_clipped=stats(np.clip(labels, -1, 1)-np.clip(t, -1, 1)),
        fresh_fit_minus_recorded_labels_clipped=stats(np.clip(p, -1, 1)-np.clip(labels, -1, 1)),
        fresh_fit_minus_frozen_final_function_clipped=stats(np.clip(p, -1, 1)-np.clip(t, -1, 1)),
        failed_execution=json.loads(base.with_suffix('.json').read_text('utf-8')),
        last_executed_row=last,
        last_ten_decisions=dict(observations=tail.tolist(), pilot_actions=predicted.tolist(),
                               frozen_function_on_same_observations=frozen.tolist()),
        interpretation='Recorded historical labels and the final frozen teacher are distinct regression targets. These are conditional action comparisons, not counterfactual successful trajectories. A complete replay is required to assess either new training recipe.')
    write_json(ROOT / 'audit/pilot_long_failure_v4.json', result)
    print(json.dumps({k: v for k, v in result.items() if k.endswith('_clipped')}, indent=2))
    print('last three clipped action differences:', (np.clip(predicted[-3:], -1, 1)-np.clip(frozen[-3:], -1, 1)).tolist())


if __name__ == '__main__':
    main()
