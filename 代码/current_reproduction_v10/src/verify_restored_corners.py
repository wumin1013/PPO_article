"""Check the restored all-corner figure against complete retained positions."""
import argparse
import json
from pathlib import Path

import numpy as np

from common import ROOT, sha, utc, write_json


def longest_true_run(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return int((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).max(initial=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Verify without writing the audit receipt.')
    args = parser.parse_args()
    source = ROOT / 'baseline_records/supplementary_reproduction/records'
    checksums, results = {}, {}
    for method in ['J-NNC', 'Two-action', 'S2']:
        trace = source / method / 'original_1ms.npz'
        corners_file = source / method / 'original_1ms_corners.json'
        result_file = source / method / 'original_1ms.json'
        for path in [trace, corners_file, result_file]:
            checksums[path.relative_to(ROOT).as_posix()] = sha(path)
        record = json.loads(result_file.read_text('utf-8'))
        corners = json.loads(corners_file.read_text('utf-8'))
        assert record['success'] and corners['all34_windows_complete']
        assert len(corners['corners']) == 34
        with np.load(trace, allow_pickle=False) as data:
            rows, reference = data['rows'], data['reference']
            if 'dt_s' in data:
                dt = float(data['dt_s'])
            else:
                time = data['t']
                dt = float((time[-1] - time[0]) / (len(time) - 1))
                assert len(time) == len(rows) + 1 and np.allclose(np.diff(time), dt, rtol=0, atol=1e-10)
        speed = np.linalg.norm(np.diff(np.vstack([reference[0], rows[:, :2]]), axis=0) / dt, axis=1)
        recalculated = []
        for corner in corners['corners']:
            assert corner['complete_window']
            first, last = corner['first_index'], corner['last_index']
            window = speed[first:last + 1]
            deficit = float(np.maximum(100.0 - window, 0.0).sum() * dt)
            assert abs(deficit - corner['speed_deficit_mm']) < 1e-6, (method, corner['id'])
            recalculated.append(dict(id=corner['id'], deficit_mm=deficit,
                                     recorded_deficit_mm=corner['speed_deficit_mm']))
        cruise_time = float((speed >= 99.0).sum() * dt)
        cruise_longest = longest_true_run(speed >= 99.0) * dt
        assert abs(cruise_time - corners['cruise']['99']['time_s']) < 1e-9
        assert abs(cruise_longest - corners['cruise']['99']['longest_time_s']) < 1e-9
        results[method] = dict(corners=recalculated,
            total_corner_deficit_mm=sum(c['deficit_mm'] for c in recalculated),
            cruising_time_s=cruise_time, longest_cruising_interval_s=cruise_longest,
            complete_time_s=len(rows) * dt,
            maximum_deficit_recalculation_difference_mm=max(abs(c['deficit_mm'] - c['recorded_deficit_mm']) for c in recalculated))
    counts = {}
    for method in ['Two-action', 'S2']:
        difference = [a['deficit_mm'] - b['deficit_mm']
                      for a, b in zip(results['J-NNC']['corners'], results[method]['corners'])]
        assert max(difference) <= 1e-7, method
        counts[method] = dict(no_greater_count=sum(d <= 1e-7 for d in difference),
                             comparison_count=len(difference), numerical_tie_tolerance_mm=1e-7)
    figure = ROOT / 'manuscript/Figure_8.pdf'
    assert sha(figure) == sha(ROOT / 'baseline_source/Figure_8.pdf')
    checksums['manuscript/Figure_8.pdf'] = sha(figure)
    checksums['baseline_source/Figure_8.pdf'] = sha(figure)
    audit = dict(
        created_utc=utc(), status='passed', source_sha256=checksums,
        script_sha256=sha(Path(__file__)), methods=results, all_corner_comparisons=counts,
        scope='Velocity recomputed from all planar positions; corner windows retain their original ordered-reference index bounds. No trajectory was rerun or replaced.')
    if not args.check_only:
        write_json(ROOT / 'audit/restored_corner_evidence_v3.json', audit)
    print(json.dumps({method: {k: v for k, v in value.items() if k != 'corners'}
                      for method, value in results.items()}, indent=2))


if __name__ == '__main__':
    main()
