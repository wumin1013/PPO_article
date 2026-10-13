"""Read-only analysis of a candidate's own two complete-task interventions."""
from pathlib import Path
import json
import sys
import numpy as np

OUT = Path(__file__).resolve().parent
BASE = Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
sys.path[:0] = [str(BASE), str(OUT)]
from study_io import sha, write_json
from natural_case_metrics import ordered_projection

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def main():
    candidate = Path(sys.argv[1])
    stage = candidate / 'channel_interventions'
    registration = read(stage / 'registration.json')
    assert sha(candidate/'selected_policy.json') == registration['selection_sha256']
    case = BASE/'paper_evidence_review/natural_closed_case'
    ref = np.loadtxt(case/'reference.csv', delimiter=',', skiprows=1)
    geom = read(case/'geometry.json')
    records = []
    for mode in ['zero_steering', 'maximum_feed']:
        folder = stage/mode
        r = read(folder/'result.json')
        assert sha(folder/'trace.npz') == r['trace_sha256']
        z = np.load(folder/'trace.npz')
        rows = z['rows']
        progress = ordered_projection(ref, rows[:,:2])
        corner = min(geom['sharp_corners'],key=lambda c:abs(c['reference_s_mm']-progress[-1]))
        records.append(dict(mode=mode, success=r['success'], reason=r['reason'], error=r['error'],
            complete_time_s=r['complete_time_s'], prefix_time_s=r['elapsed_prefix_or_complete_s'],
            failed_corner=corner['id'] if not r['success'] else None,
            failed_corner_angle_deg=corner['interior_angle_deg'],
            last_reference_progress_mm=float(progress[-1]),
            failure_progress_from_corner_mm=float(progress[-1]-corner['reference_s_mm']),
            last_speed_mm_s=float(np.linalg.norm(rows[-1,2:4])),
            last_position_mm=rows[-1,:2].tolist(),
            vector_bounds_pass=r['audit']['kinematic_pass'],
            analytic_corridor_pass=r['analytic']['analytic_corridor_pass'],
            maximum_evaluated_distance_mm=r['analytic']['maximum_evaluated_distance_mm'],
            gates_passed=r['geometry']['gates_passed'],gate_count=r['geometry']['gate_count'],
            source_sha256={str(folder/'result.json'):sha(folder/'result.json'),
                           str(folder/'trace.npz'):sha(folder/'trace.npz')}))
    report = dict(candidate=str(candidate), records=records,
        both_fail_geometry_with_vector_bounds_preserved=all(not r['success'] and r['reason']=='corridor_violation'
            and r['vector_bounds_pass'] and r['complete_time_s'] is None for r in records),
        new_policy_calls=0,new_physics_ticks=0,
        interpretation='Causal output interventions of the selected composite policy, without retraining. '
            'Memory and remaining channels evolve causally. No complete-time comparison for failed prefixes; '
            'not an assertion that independently retrained two-action controllers must fail.',
        receipt=read(stage/'receipt.json'),source_sha256={str(Path(__file__)):sha(Path(__file__)),
            str(stage/'registration.json'):sha(stage/'registration.json'),str(case/'geometry.json'):sha(case/'geometry.json')})
    write_json(stage/'mechanism_audit.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
