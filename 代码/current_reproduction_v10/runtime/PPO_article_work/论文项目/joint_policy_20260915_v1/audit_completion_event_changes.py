"""Saved-state cause checks for completion changes; no altered rollout or adoption."""
import argparse
import itertools
import time
import numpy as np
from joint_runtime import terminal
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha


def terminal_query(state, endpoint):
    try:
        plan=terminal.plan_planar_terminal_brake(state[:2],state[2:4],state[4:6],
                                               dt=.001,limits=terminal.LIMITS)
    except ValueError as error:
        return dict(feasible=False,reason=str(error))
    c=plan['columns'];end=np.array([c['x_mm'][-1],c['y_mm'][-1]])
    return dict(feasible=True,count=plan['braking_steps'],endpoint=end.tolist(),
                endpoint_distance_mm=float(np.linalg.norm(end-endpoint)),
                endpoint_eligible=bool(np.linalg.norm(end-endpoint)<=1.))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--feedback',required=True)
    parser.add_argument('--stage',required=True)
    parser.add_argument('--full-parent',help='Use the full64 pools and bind their original parent stage')
    args=parser.parse_args();source=BASE/args.feedback
    data=read(source/'results.json');base=data['baseline'];candidate=data['candidates'][0]
    extra_sources=[]
    if args.full_parent:
        parent=BASE/args.full_parent;parent_data=read(parent/'results.json')
        base=parent_data['full_candidate'];candidate=data['full_candidate']
        assert candidate is not None and len(base['results'])==len(candidate['results'])==64
        assert sha(parent/parent_data['chosen']['model'])==data['parent_sha256']
        extra_sources=[parent/'results.json',parent/'receipt.json']
    stage=register(args.stage,[__file__,source/'results.json',source/'receipt.json',
        source/candidate['model'],BASE/'joint_runtime.py']+extra_sources,
        dict(new_actor_decisions=0,new_physics_ticks=0,terminal_planner_queries=256,wall_s=60),
        'Compare each saved actual complete trace with its bound parent. Separate '
        'changed ordinary tick count from complete brake/hold duration. At each '
        'parent initiation tick on routes with changed completion time, query the '
        'unchanged planner on all eight parent/candidate combinations of actual '
        'position, velocity and acceleration. These are offline state interventions '
        'for the missed terminal event, not deployed acceleration resets or new '
        'policy success. Preserve branch failures and endpoint rejection. Do not '
        'infer that equal tick counts imply equal speed or shape.')
    start=time.perf_counter();records=[];queries=0
    parent={r['original_path']:(r,t) for r,t in zip(base['results'],base['traces'])}
    for result,trace in zip(candidate['results'],candidate['traces']):
        path=result['original_path'];old,old_trace=parent[path]
        assert result['success'] and old['success']
        with np.load(BASE/trace) as z:ref,rows=z['reference'].copy(),z['rows'].copy()
        with np.load(BASE/old_trace) as z:previous=z['rows'].copy()
        first=int(np.flatnonzero(previous[:,12])[0]);second=int(np.flatnonzero(rows[:,12])[0])
        detail=dict(path=path,time_delta_ms=len(rows)-len(previous),
            ordinary_delta_ms=second-first,brake_delta_ms=len(rows)-second-len(previous)+first,
            parent_brake_tick=first,candidate_brake_tick=second,
            peak_speed_delta_mm_s=float(np.linalg.norm(rows[:,2:4],axis=1).max()-
                                       np.linalg.norm(previous[:,2:4],axis=1).max()))
        if detail['time_delta_ms'] and first and first<=len(rows):
            old_state,new_state=previous[first-1,:6],rows[first-1,:6]
            detail['candidate_minus_parent_at_parent_brake']=(new_state-old_state).tolist()
            outcomes=[]
            for selected in itertools.product((0,1),repeat=3):
                state=np.concatenate([(new_state if flag else old_state)[2*j:2*j+2]
                                      for j,flag in enumerate(selected)])
                outcomes.append(dict(candidate_components=list(selected),**terminal_query(state,ref[-1])))
                queries+=1
            detail['planner_state_interventions']=outcomes
        records.append(detail)
    assert queries<=256 and time.perf_counter()-start<60
    summary=dict(time_delta_ms=sum(r['time_delta_ms'] for r in records),
        ordinary_delta_ms=sum(r['ordinary_delta_ms'] for r in records),
        brake_delta_ms=sum(r['brake_delta_ms'] for r in records),
        changed_paths=[r['path'] for r in records if r['time_delta_ms']])
    write_json(stage/'results.json',dict(parent_sha256=data['parent_sha256'],records=records,
        summary=summary,goal_complete=False))
    receipt(stage,dict(status='completed_saved_completion_event_diagnosis',new_actor_decisions=0,
        new_physics_ticks=0,terminal_planner_queries=queries,wall_s=time.perf_counter()-start,
        goal_complete=False))
    print('COMPLETION_CHANGES',summary,flush=True)
    for r in records:
        if r['ordinary_delta_ms']:print('ORDINARY_EVENT_CHANGE',r,flush=True)


if __name__=='__main__':main()
