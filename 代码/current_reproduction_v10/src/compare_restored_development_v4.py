"""Compare restored strong-policy records with the completed v3 planners."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRIOR = Path(r'D:\PPO_article_work\JNN_revision_20261011_v3')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    rows, hashes = [], {}
    for case in ['original', 'dense']:
        learned_path = ROOT / 'results/strong_checkpoint_v4/v1' / (case + '.json')
        learned = json.loads(learned_path.read_text('utf-8'))
        with np.load(learned_path.with_suffix('.npz')) as trace:
            reference = trace['reference'].copy()
        for method in ['S2', 'S2R', 'SCMO']:
            path = PRIOR / 'results/baseline_v3/development_unified' / (method + '_' + case + '_dt1000us_scale1000.json')
            baseline = json.loads(path.read_text('utf-8'))
            with np.load(path.with_suffix('.npz')) as trace:
                assert np.array_equal(reference, trace['reference']), (case, method)
            for record in [learned, baseline]:
                assert record['success'] and record['kinematic_pass'] and record['rest_pass']
                assert record['geometry']['ordered_complete'] and record['geometry']['corridor_inside']
                assert record['geometry']['gates_passed'] == record['geometry']['gate_count']
                assert record['dt_s'] == .001 and record['limits'] == [100., 2000., 62500.]
            assert abs(learned['tolerance_mm'] - baseline['tolerance_mm']) < 1e-12
            t, b = learned['complete_time_s'], baseline['complete_time_s']
            rows.append({'case': case, 'comparator': method, 'strong_JNNC_s': t,
                         'comparator_s': b, 'saving_s': b-t, 'saving_percent': 100*(b-t)/b,
                         'tolerance_mm': learned['tolerance_mm'], 'all_constraints_pass': True,
                         'scope': 'Previously used development programs, one restored model; not independent final evaluation'})
            for entry in [learned_path, learned_path.with_suffix('.npz'), path, path.with_suffix('.npz')]:
                hashes[str(entry)] = sha(entry)
    out = ROOT / 'audit'
    with (out / 'restored_development_comparison_v4.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {'scope': 'All restored strong-model development comparisons against the existing completed planner grid',
              'model_sha256': learned['model_sha256'], 'source_sha256': hashes, 'rows': rows,
              'new_simulation_performed_by_this_script': False,
              'validation': 'Same complete reference arrays, tolerance, vector limits, 1 ms sampling and all completion conditions; recorded successful full trajectories.'}
    (out / 'restored_development_comparison_v4.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
