"""Recheck stored raw outcomes; no physical experiment is repeated."""
import numpy as np
import argparse
from study_io import BASE,sha,write_json
from search_complete_return import read


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',default='canonical_gate_band_joint_teacher_state_distillation_20260919')
    parser.add_argument('--expected-changed',type=int,nargs='+',default=[13])
    parser.add_argument('--allow-action-roundoff',action='store_true')
    parser.add_argument('--candidate',default=None)
    parser.add_argument('--method',default='Offline joint-action fitting followed by actual complete training-pool closed-loop execution; not PPO.')
    args=parser.parse_args()
    stage=BASE/args.stage
    result=read(stage/'results.json')
    selected=next(c for c in result['candidates'] if c['tag']==(args.candidate or result['best']))
    model_file=stage/args.candidate/'candidate.pt' if args.candidate else stage/'research_candidate.pt'
    baseline_file=stage/'baseline.json'
    if not baseline_file.exists():baseline_file=BASE/'canonical_gate_band_current_bend_capacity_learning_20260919/baseline.json'
    baseline=read(baseline_file)['baseline']
    assert len(selected['records'])==len(baseline['records'])
    physical=[];identity=[];traces=[]
    for r,b in zip(selected['records'],baseline['records']):
        assert r['path']==b['path'] and r['success'] and r['audit']['kinematic_pass'] and r['audit']['rest_pass']
        assert r['geometry']['ordered_complete'] and r['geometry']['geometry_complete']
        assert r['corridor']['max_distance_upper_mm']<=1.25+1e-8
        file=stage/r['trace']
        with np.load(file) as z:rows,actions=z['rows'].copy(),z['actions'].copy()
        with np.load(b['trace']) as z:old,old_actions=z['rows'].copy(),z['actions'].copy()
        physical.append([np.linalg.norm(rows[:,2:4],axis=1).max(),np.linalg.norm(rows[:,4:6],axis=1).max(),
                         np.linalg.norm(rows[:,6:8],axis=1).max()])
        identity.append(dict(path=r['path'],xy_and_motion_bitwise=np.array_equal(rows,old),
                             actions_bitwise=np.array_equal(actions,old_actions),
                             max_action_delta=float(abs(actions-old_actions).max())
                             if actions.shape==old_actions.shape else None))
        traces.append(dict(path=r['path'],sha256=sha(file)))
    changed=[r['path'] for r in identity if not r['xy_and_motion_bitwise'] or not r['actions_bitwise']]
    motion_changed=[r['path'] for r in identity if not r['xy_and_motion_bitwise']]
    if args.allow_action_roundoff:
        assert motion_changed==args.expected_changed,(motion_changed,args.expected_changed)
        assert all(r['max_action_delta']<=1e-12 for r in identity if r['path'] not in args.expected_changed)
    else:
        assert changed==args.expected_changed,(changed,args.expected_changed)
    audit=dict(model_sha256=sha(model_file),selected=selected['tag'],
        successes=len(selected['records']),all_ordered_complete=True,all_braking_rest_pass=True,
        vector_peaks=np.max(physical,axis=0).tolist(),
        original_decision_ms=20,original_execution_ms=1,original_tolerance_mm=1.25,
        maximum_corridor_distance_upper_mm=max(r['corridor']['max_distance_upper_mm'] for r in selected['records']),
        unchanged_other_paths_bitwise=all(r['xy_and_motion_bitwise'] and r['actions_bitwise'] for r in identity if r['path'] not in args.expected_changed),
        unchanged_other_motion_bitwise=all(r['xy_and_motion_bitwise'] for r in identity if r['path'] not in args.expected_changed),
        allowed_action_only_roundoff=1e-12 if args.allow_action_roundoff else 0.,
        motion_changed_paths=motion_changed,
        per_path_identity=identity,raw_trace_sha256=traces,
        full_summary=selected['summary'],changed_paths=changed,
        changed_metrics=[dict(path=p,before=next(r for r in baseline['records'] if r['path']==p),
            after=next(r for r in selected['records'] if r['path']==p)) for p in changed],
        research_selection_pass=selected['research_acceptable'],final_adopted=False,
        goal_complete=False,independent_fair_generalization_done=False,
        method=args.method)
    write_json(stage/(f'raw_result_audit_{args.candidate}.json' if args.candidate else 'raw_result_audit.json'),audit)
    print('RAW_RESULT_AUDIT',audit['model_sha256'],audit['vector_peaks'],audit['unchanged_other_paths_bitwise'])


if __name__=='__main__':main()
