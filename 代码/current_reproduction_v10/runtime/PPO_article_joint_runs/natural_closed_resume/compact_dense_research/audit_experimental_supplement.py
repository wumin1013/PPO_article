"""Read-only window sensitivity and provenance audits of frozen paper evidence."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from evaluate_historical_dense import ROOT, BASE, sha, write
from natural_case_metrics import ordered_projection

OUT = ROOT / 'experimental_supplement_20260923'
DELIVERY = BASE / 'paper_evidence_review/verified_results/lowdim_original_revision'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def window_audit(binding):
    case = binding['cases']['original']
    ref = np.loadtxt(case['reference'], delimiter=',', skiprows=1)
    geometry = read(case['geometry'])
    corners = geometry['sharp_corners']
    centers = np.array([c['reference_s_mm'] for c in corners])
    length = np.linalg.norm(np.diff(ref, axis=0), axis=1).sum()
    boundaries = np.r_[0., (centers[:-1] + centers[1:]) / 2., length + 1e-6]
    assert len(centers) == 34 and np.all(np.diff(centers) > 60)
    records = []
    aggregate = []
    for method, cases in binding['methods'].items():
        for entry in cases['original']:
            for key in ('trace', 'metrics', 'result'):
                assert sha(entry[key]) == entry[key + '_sha256']
            with np.load(entry['trace']) as data:
                rows = data['rows'].copy()
            dt = entry['dt_s']
            progress = ordered_projection(ref, rows[:, :2])
            deficit = dt * np.maximum(100. - np.linalg.norm(rows[:, 2:4], axis=1), 0.)
            payload = read(entry['metrics'])
            registered = {c['id']: c for c in payload.get('metrics', payload)['corners']}
            masks = {str(w): np.zeros(len(rows), bool) for w in (10, 20, 30)}
            cell_sum = 0.
            for i, corner in enumerate(corners):
                scope = {}
                for w in (10, 20, 30):
                    mask = np.abs(progress - centers[i]) <= w
                    assert not np.any(mask & masks[str(w)])
                    masks[str(w)] |= mask
                    scope[str(w)] = float(deficit[mask].sum())
                error = abs(scope['10'] - registered[corner['id']]['speed_deficit_mm'])
                assert error < 1e-9, (method, dt, corner['id'], error)
                cell = (progress >= boundaries[i]) & (progress < boundaries[i + 1])
                scope['midpoint_partition'] = float(deficit[cell].sum())
                cell_sum += scope['midpoint_partition']
                # Threshold is diagnostic only; the score retains every speed deficit.
                material = cell & (np.linalg.norm(rows[:, 2:4], axis=1) < 99.)
                extent = [float(progress[material].min() - centers[i]),
                          float(progress[material].max() - centers[i])] if material.any() else None
                records.append(dict(method=method, dt_s=dt, corner=corner['id'],
                    deficit_mm=scope, below99_extent_relative_mm=extent,
                    midpoint_cell_mm=boundaries[i:i + 2].tolist(),
                    contains_start_or_stop=i in (0, 33)))
            assert abs(cell_sum - deficit.sum()) < 1e-8
            aggregate.append(dict(method=method, dt_s=dt, full_deficit_mm=float(deficit.sum()),
                summed_windows_mm={w: float(deficit[m].sum()) for w, m in masks.items()},
                outside_windows_mm={w: float(deficit[~m].sum()) for w, m in masks.items()},
                midpoint_sum_mm=cell_sum))
            print('WINDOW_RECORD', method, dt, aggregate[-1], flush=True)
    comparisons = []
    for dt in (.001, .0005):
        main = {r['corner']: r for r in records if r['method'] == 'J-NNC' and r['dt_s'] == dt}
        for other in ('S2', 'Two-action'):
            comparator = {r['corner']: r for r in records if r['method'] == other and r['dt_s'] == dt}
            for window in ('10', '20', '30', 'midpoint_partition'):
                differences = {c: main[c]['deficit_mm'][window] - comparator[c]['deficit_mm'][window] for c in main}
                failures = {c: d for c, d in differences.items() if d > 1e-7}
                comparisons.append(dict(dt_s=dt, comparator=other, window=window,
                    noninferior_count=34 - len(failures), count=34, exceptions_mm=failures,
                    maximum_excess_mm=max(differences.values())))
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        kind='Post-freeze sensitivity analysis of existing complete trajectories; no training or simulation',
        binding=str(DELIVERY / 'paper_result_binding.json'), binding_sha256=sha(DELIVERY / 'paper_result_binding.json'),
        script_sha256=sha(__file__), source_sha256={entry[k]: entry[k+'_sha256']
            for cases in binding['methods'].values() for entry in cases['original'] for k in ('trace', 'metrics', 'result')},
        original_window_reproduced=True, original_metric_unchanged=True,
        tie_tolerance_mm=1e-7, midpoint_partition_note='All motion assigned once; first/last cells include startup/stop, so these are route partitions, not isolated corner losses.',
        all_records=records, aggregate=aggregate, comparisons=comparisons)
    write(OUT / 'window_sensitivity.json', result)
    with (OUT / 'window_sensitivity.csv').open('w', encoding='utf-8', newline='') as f:
        writer=csv.writer(f);writer.writerow(['method','dt_s','corner','D10_mm','D20_mm','D30_mm','Dpartition_mm'])
        for r in records:
            writer.writerow([r['method'],r['dt_s'],r['corner']]+[r['deficit_mm'][k] for k in ('10','20','30','midpoint_partition')])
    print('WINDOW_COMPARISONS', json.dumps(comparisons), flush=True)


def budget_audit(binding):
    import torch
    torch.set_num_threads(1)
    main = read(ROOT / 'direct14_dictionary02_resume_round73/candidate0_ancestry_audit.json')
    starts = {'J-NNC': Path(binding['selected_checkpoint']),
        'Two-action': Path(read(binding['methods']['Two-action']['original'][0]['result'])['checkpoint'])}
    output = {}
    for name, start in starts.items():
        paths = [start]; seen = set(); records = []
        while paths:
            path = paths.pop(0)
            if not path.is_absolute(): path = ROOT / path
            path = path.resolve()
            if str(path) in seen: continue
            seen.add(str(path))
            payload = torch.load(path, map_location='cpu', weights_only=False)
            keys = ['training_kind','policy_class','updates','steps','new_ppo_updates','ppo_updates_this_round',
                    'inherited_ppo_updates','offline_iterations','checkpoint_index_kind','interface','head_bias_delta']
            record = dict(path=str(path), sha256=sha(path), metadata={k:payload[k] for k in keys if k in payload}, links=[])
            for key in ('parent_checkpoint', 'initial_checkpoint', 'source_checkpoint'):
                if payload.get(key):
                    parent = Path(payload[key]); parent = parent if parent.is_absolute() else ROOT / parent
                    if parent.exists():
                        record['links'].append(dict(kind=key,path=str(parent),sha256=sha(parent)))
                        paths.append(parent)
                    else:record['links'].append(dict(kind=key,path=str(parent),available=False))
            for filename in ('registration.json','receipt.json','result.json'):
                candidate=path.parent/filename
                if not candidate.exists() and path.parent.name.startswith(('d_', 'candidate_')):
                    candidate=path.parent.parent/filename
                if candidate.exists():
                    data=read(candidate)
                    fields=['kind','seed','updates','ppo_updates','new_decisions','wall_s','combined_wall_s','status',
                            'offline_iterations','scheduled_dense_checks','independence','fairness_scope','procedure']
                    record[filename]=dict(path=str(candidate),sha256=sha(candidate),values={k:data[k] for k in fields if k in data})
                    if filename == 'registration.json':
                        for key in ('initial_checkpoint','initial','parent'):
                            if not data.get(key):continue
                            parent=Path(data[key])
                            parent=parent if parent.is_absolute() else ROOT/parent
                            if parent.is_file() and parent.suffix=='.pt' and parent.resolve()!=path:
                                digest=sha(parent)
                                expected=data.get(key+'_sha256')
                                if key=='initial_checkpoint':expected=expected or data.get('initial_sha256')
                                if expected:assert digest==expected
                                record['links'].append(dict(kind='registered_'+key,path=str(parent),sha256=digest))
                                paths.append(parent)
            records.append(record)
        output[name] = records
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),records=output,
        scope='Checkpoint ancestry and available per-stage receipts. Does not include every discarded search branch or ancestral teacher-generation cost; do not label as exhaustive end-to-end wall budget.',
        main_verified_ancestral_ppo_updates=88,
        two_action_final_stage_updates_offered=48,two_action_final_stage_selected_update=24,
        budgets_matched=False,independent_full_pipeline_repetitions=0,
        previous_main_ancestry_sha256=sha(ROOT/'direct14_dictionary02_resume_round73/candidate0_ancestry_audit.json'))
    write(OUT/'training_budget_audit.json',result)
    print('BUDGET_AUDIT', {k:len(v) for k,v in output.items()},flush=True)


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    binding=read(DELIVERY/'paper_result_binding.json')
    if '--budget-only' not in sys.argv:window_audit(binding)
    budget_audit(binding)
