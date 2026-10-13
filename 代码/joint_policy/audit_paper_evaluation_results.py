"""Independent saved-only checks and descriptive historical baseline comparisons."""
import json
import time
import numpy as np
from geometry_spec import Corridor, GateTracker, audit_commands
from audit_function_constraint_interference import command_segment_margin, distances_to_segments
from complete_motion_metrics import complete_motion_metrics
from complete_corner_metrics import complete_corner_metrics
from validate_actions import difference_audit
from study_io import BASE, register, receipt, sha, write_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def paired(new, old):
    deltas=[a['complete_time_s']-b['complete_time_s'] for a,b in zip(new,old)]
    total_old=sum(r['complete_time_s'] for r in old)
    return dict(paths=len(deltas), current_total_s=sum(r['complete_time_s'] for r in new),
        baseline_total_s=total_old, time_change_s=sum(deltas),relative_time_change_percent=100*sum(deltas)/total_old,
        faster=sum(d<-1e-9 for d in deltas),equal=sum(abs(d)<=1e-9 for d in deltas),slower=sum(d>1e-9 for d in deltas),
        per_path_time_change_s=deltas, budget_matched=False,
        causal_joint_learning_advantage_established=False,scope='Previously used development paths; historical system-level comparison only')


def main():
    review=BASE/'paper_evidence_review';development=read(review/'development/results.json')
    long=read(review/'public_long/evaluation.json')
    old_fixed=read(BASE/'fixed3_gsde_refinement_v7/final_validation/summary.json')['results']
    old_s2=read(BASE/'s2_line_development_v2/results.json')['results']
    stage=register('paper_evidence_review/evaluation_audit',
        [__file__, review/'development/results.json',review/'development/receipt.json',
         review/'public_long/evaluation.json',review/'public_long/receipt.json',
         BASE/'fixed3_gsde_refinement_v7/final_validation/summary.json',
         BASE/'fixed3_gsde_refinement_v7/receipt.json',BASE/'s2_line_development_v2/results.json',
         BASE/'s2_line_development_v2/receipt.json',BASE/'geometry_spec.py',
         BASE/'audit_function_constraint_interference.py',BASE/'complete_motion_metrics.py',
         BASE/'complete_corner_metrics.py',BASE/'validate_actions.py'],
        dict(new_simulation_ticks=0,new_actor_calls=0,wall_s=120,saved_development_baseline_paths=20,failed_long_prefixes=1),
        'Independent geometry and motion checks on stored commands. Verify the long failure against '
        'the full finite reference and exact gate/corridor code, not a cropped segment subset. '
        'Reuse all12 strong fixed-D and all8 compatible S2 saved development paths with identical '
        'reference arrays. Report system-level paired times with unmatched training/sensing confounds. '
        'Do not infer long completion from its cruise prefix, causality from failure geometry, or adjust the candidate.')
    started=time.perf_counter(); current=development['records'];baselines={};bindings=[]
    for name, original, folder, filename in [
        ('fixed3',old_fixed,BASE/'fixed3_gsde_refinement_v7/final_validation','{i}.npz'),
        ('S2',old_s2,BASE/'s2_line_development_v2','path_{i}.npz')]:
        records=[]
        for r in original:
            i=r['path'];file=folder/filename.format(i=i)
            with np.load(file) as z:ref,rows=z['reference'].copy(),z['rows'].copy()
            with np.load(BASE/current[i]['trace']) as z:assert np.array_equal(ref,z['reference'])
            audit=difference_audit(ref[0],rows)
            geometry,_=audit_commands(ref,np.vstack([ref[0],rows[:,:2]]),epsilon=1.25,
                endpoint_tolerance=1.25,closed=bool(np.linalg.norm(ref[0]-ref[-1])<1e-9))
            assert r['success'] and audit['kinematic_pass'] and audit['rest_pass'] and geometry['geometry_complete']
            assert abs(len(rows)*.001-r['completion_time_s'])<1e-9
            record=dict(path=i,complete_time_s=r['completion_time_s'],success=True,audit=audit,
                geometry=geometry,motion=complete_motion_metrics(ref,rows),trace_sha256=sha(file))
            if len(ref)==3:record['shape']=complete_corner_metrics(ref,rows)
            records.append(record);bindings.append(dict(path=str(file.relative_to(BASE)),sha256=sha(file)))
        baselines[name]=dict(records=records,paired=paired([current[r['path']]for r in records],records))
    file=review/'public_long/trace.npz';assert sha(file)==long['trace_sha256']
    with np.load(file) as z:ref,rows,actions=[z[k].copy()for k in ('reference','rows','actions')]
    corridor=Corridor(ref);tracker=GateTracker(corridor,ref[0]);first_invalid=None
    for i,xy in enumerate(rows[:,:2]):
        tracker.update(xy)
        if not tracker.inside and first_invalid is None:first_invalid=i
    assert first_invalid is not None and first_invalid==len(rows)-1
    enclosure=command_segment_margin(ref,rows[:,:2],tolerance=1e-10)
    distance=distances_to_segments(ref,rows[-2:,:2])
    nearest=distance.argmin(axis=1);values=distance.min(axis=1)
    assert values[-1]>1.25 and values[-2]<=1.25
    assert tracker.frontier==long['geometry']['gates_passed']==7
    assert not long['success'] and long['complete_time_s'] is None
    vertex=tracker.frontier+1;u,v=np.diff(ref[vertex-1:vertex+2],axis=0)
    u/=np.linalg.norm(u);v/=np.linalg.norm(v);sign=np.sign(u[0]*v[1]-u[1]*v[0]);n=sign*np.array([-u[1],u[0]])
    q=rows[-1,:2]-ref[vertex]
    failure=dict(independent_failure_reproduced_without_simulation=True,first_invalid_tick=first_invalid+1,
        first_invalid_time_s=(first_invalid+1)*.001,xy_mm=rows[-1,:2].tolist(),
        previous_tick_distance_mm=float(values[-2]),first_invalid_distance_mm=float(values[-1]),
        violation_mm=float(values[-1]-1.25),nearest_reference_segments_zero_based=nearest.tolist(),
        continuous_command_segment_enclosure=enclosure,ordered_gates_passed=tracker.frontier,total_gates=len(corridor.gates),
        next_reference_vertex=vertex,next_turn_deg=float(np.rad2deg(np.arctan2(u[0]*v[1]-u[1]*v[0],u@v))),
        incoming_inward_offset_mm=float(q@n),distance_before_vertex_along_incoming_mm=float(-q@u),
        last_velocity_mm_s=rows[-1,2:4].tolist(),last_decoded_action=actions[-1].tolist(),
        interpretation='The incoming inward offset exceeds the full finite-polyline corridor before the next turn. This is a spatial observation, not a unique causal attribution to an action head.',
        complete_time_s=None,prefix_cruise_excluded_from_complete_efficiency_claim=True)
    write_json(stage/'results.json',dict(baselines=baselines,long_failure=failure,baseline_trace_bindings=bindings,
        fixed_joint_causal_comparison_confounds=['different registered sensing/guide representations',
            'different fitting/parameter-search histories and checkpoint opportunities','no matched end-to-end independent seeds'],
        dataset_mixing_forbidden=True,new_actor_calls=0,new_simulation_ticks=0))
    receipt(stage,dict(status='completed_saved_evaluation_comparison_and_failure_audit',
        baseline_paths=20,failed_long_prefixes=1,new_actor_calls=0,new_simulation_ticks=0,
        results_sha256=sha(stage/'results.json'),wall_s=time.perf_counter()-started,goal_complete=False))
    print('PAIRED',json.dumps({k:v['paired']for k,v in baselines.items()}),flush=True)
    print('FAILURE',json.dumps(failure),flush=True)


if __name__=='__main__':main()
