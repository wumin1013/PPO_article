"""Check cruise-target coverage on saved current-policy straight segments."""
import time
import numpy as np
import torch
from canonical_gate_state import checkpoint_environment_factory
from probe_coordinated_suffix import reconstruct
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    parent=BASE/'canonical_gate_band_duration_local_shape_learning_20260918'
    data=read(parent/'results.json');full=data['full_candidate'];model=parent/data['chosen']['model']
    assert full['checks']['eligible'] and full['summary']['successes']==64
    stage=register('canonical_gate_band_segment_cruise_support_20260918',
        [__file__,model,parent/'results.json',parent/'receipt.json',parent/'actual_phase_audit.json',
         BASE/'prepare_physical_update_geometry.py',BASE/'probe_coordinated_suffix.py',BASE/'joint_runtime.py'],
        dict(saved_geometry_ticks=100000,new_actor_decisions=0,new_physics_ticks=0,wall_s=90,
             original_endpoint_margin_mm=6.2,diagnostic_window_margin_mm=12.4,
             forthcoming_probe_paths=4,max_intervention_decisions=10),
        'The retained learned update improves shape and99mm/s cruise only on pure '
        'straight routes. Existing prepare_physical_update_geometry code restricts '
        'cruise-target support to len(reference)==2. Audit ALL64 saved traces for '
        'otherwise equivalent straight SEGMENT interiors in isolated/mixed paths, '
        'using ordered progress and the same6.2mm endpoint exclusion. These are '
        'offline target-coverage candidates, not a claim that geometry alone proves '
        'dynamic speed feasibility. Report learned feed, actual vector speed and '
        'a/j projector guard usage. Select at most four training-only windows by '
        'longest consecutive coverage, two isolated and two multisegment paths, '
        'using12.4mm clearance from both segment ends and at most10 decisions. '
        'Only a subsequent bounded feed-only full-path intervention can test '
        'whether this omitted support contains reachable cruise. No policy change, '
        'angle table, deployed reference-speed rule, generalization or sealed access.')
    start=time.perf_counter();costs=dict(saved_prefix_geometry_ticks=0,new_physics_ticks=0)
    md=torch.load(model,map_location='cpu',weights_only=False);factory=checkpoint_environment_factory(md['architecture'])
    records=[];states_data=[]
    for result,trace in zip(full['results'],full['traces']):
        with np.load(BASE/trace) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        states=[];env=reconstruct(ref,rows,actions,(len(actions)-1)*20,costs,state_snapshots=states,env_factory=factory)
        states.append(env.state());category='straight' if len(ref)==2 else 'isolated' if len(ref)==3 else 'multisegment'
        eligible=[];windows=[];run=[];last_segment=None
        for k,state in enumerate(states[:-1]):
            env.restore(state);progress,segment=env.progress();length=env.corridor.arcs[segment+1]-env.corridor.arcs[segment]
            along=progress-env.corridor.arcs[segment];left=length-along
            speed=float(np.linalg.norm(state['velocity']));feed=float(actions[k,1])
            if min(along,left)>=6.2:
                eligible.append(k);states_data.append(dict(path=result['original_path'],decision=k,category=category,
                    segment=int(segment),distance_from_start_mm=float(along),distance_to_end_mm=float(left),
                    feed_mm_s=feed,actual_speed_mm_s=speed,projector_guard_ticks=int(rows[k*20:(k+1)*20,11].sum())))
            if min(along,left)>=12.4:
                if run and (k!=run[-1]+1 or segment!=last_segment):windows.append(run);run=[]
                run.append(k);last_segment=segment
            elif run:windows.append(run);run=[]
        if run:windows.append(run)
        longest=max(windows,key=len) if windows else []
        records.append(dict(path=result['original_path'],category=category,trace=trace,
            eligible_decisions=len(eligible),longest_conservative_window_decisions=len(longest),
            window_start=longest[0] if longest else None,probe_decisions=min(10,len(longest))))
        assert time.perf_counter()-start<85 and costs['saved_prefix_geometry_ticks']<=100000
    groups=[]
    for category in ('straight','isolated','multisegment'):
        selected=[r for r in states_data if r['category']==category]
        groups.append(dict(category=category,states=len(selected),
            target_supported_by_existing_rule=category=='straight',
            mean_feed_mm_s=float(np.mean([r['feed_mm_s'] for r in selected])),
            minimum_feed_mm_s=float(min(r['feed_mm_s'] for r in selected)),
            maximum_feed_mm_s=float(max(r['feed_mm_s'] for r in selected)),
            mean_actual_speed_mm_s=float(np.mean([r['actual_speed_mm_s'] for r in selected])),
            decisions_actual_ge99=sum(r['actual_speed_mm_s']>=99 for r in selected),
            projector_guard_ticks=sum(r['projector_guard_ticks'] for r in selected)))
    probes=[]
    for category in ('isolated','multisegment'):
        eligible=[r for r in records if r['category']==category and r['probe_decisions']>=4]
        probes+=sorted(eligible,key=lambda r:(-r['longest_conservative_window_decisions'],r['path']))[:2]
    write_json(stage/'results.json',dict(parent_sha256=sha(model),parent_model=str(model.relative_to(BASE)),
        records=records,groups=groups,states=states_data,selected_probe_windows=probes,costs=costs,
        dynamic_cruise_feasibility_established=False,goal_complete=False))
    receipt(stage,dict(status='completed_saved_segment_cruise_support_audit',costs=costs,
        wall_s=time.perf_counter()-start,goal_complete=False))
    print('SEGMENT_CRUISE_SUPPORT',groups,'selected',probes,flush=True)


if __name__=='__main__':main()
