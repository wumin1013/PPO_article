"""Geometry continuity, default-executor regression, and matched guide contrast."""
import time
import numpy as np
import torch
from geometry_spec import audit_commands
from joint_runtime_v5 import environment_factory as original_factory
from continuous_sensor_runtime import environment_factory as sensor_factory
from continuous_gate_guidance import environment_factory,continuous_gate_target,GATE_TRANSITION_WIDTH_MM
from kernel_local_feedback_policy import KernelLocalFeedbackPolicy
from probe_coordinated_suffix import reconstruct
from learn_single_decision_credit import review
from probe_direct_constraint_composition import metric_rows
from solve_continuous_sensor_credit import NAMES
from train_pilot import evaluate
from search_complete_return import read
from study_io import BASE,ROOT,register,receipt,write_json,sha


def main():
    torch.set_num_threads(1)
    source=BASE/'continuous_sensor_matched_function_transfer_20260917'
    selected=next(r for r in read(source/'results.json')['candidates'] if r['arm']=='updated')
    composition=BASE/'continuous_sensor_complete_composition_20260917'
    models=[('base',source/'updated.pt'),('learned_positive',composition/'step+0.125.pt')]
    stage=register('continuous_gate_guidance_mechanism_probe_20260917',
        [__file__,BASE/'continuous_gate_guidance.py',BASE/'continuous_sensor_runtime.py',
         BASE/'joint_runtime.py',ROOT/'tmp/joint_runtime_before_guidance_hook_20260917.py',
         BASE/'one_ms_gate_guidance_causal_probe_20260917/results.json',
         source/'results.json',composition/'results.json',*[m for _,m in models]],
        dict(guide_transition_width_mm=GATE_TRANSITION_WIDTH_MM,
            geometry_preflight_paths=29,default_regression_decisions=6,
            evaluation_actor_calls=6000,evaluation_physics_ticks=140000,
            saved_geometry_ticks=10000,wall_s=150,training_updates=0),
        'After bidirectional1ms causal evidence, test a generic continuous target '
        'query at ordered segment transitions. A global0.1mm gate-normal band equals '
        'the100mm/s maximum1ms execution displacement. Within the actual gate channel, '
        'blend the adjacent projected-arc reference targets with a C1 cubic weight; '
        'outside it use the original target exactly. This is guide signal processing, '
        'not a prescribed corner trajectory, angle speed table, policy output override '
        'or deployed probe schedule. Coverage, corridor, V/A/J projection, braking '
        'and decision/execution clocks remain original. An explicit guidance_target '
        'hook leaves the original default behavior algebraically identical; verify '
        'six actual20ms saved-state replays before the new guide. Check same-position '
        'frontier continuity on every saved gate, and reject overlapping transition '
        'neighborhoods rather than silently falling back. Evaluate unchanged base '
        'and the already learned positive step on29 training routes, reusing their '
        'old-guide traces. This matched architecture mechanism study is not new '
        'training or final acceptance. A useful result requires complete success '
        'and actual shape evidence, then new matched learning and fair controls.')
    start=time.perf_counter();costs=dict(saved_prefix_geometry_ticks=0,evaluation_actor_calls=0,
        evaluation_physics_ticks=0,default_regression_actor_calls=0,default_regression_physics_ticks=0)
    p=KernelLocalFeedbackPolicy(256);p.load_state_dict(torch.load(models[0][1],map_location='cpu',weights_only=False)['policy'])
    event_checks=[];regression=[]
    for r,trace in zip(selected['results'],selected['traces']):
        path=r['original_path']
        with np.load(BASE/trace) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        _,events=audit_commands(ref,np.vstack([ref[0],rows[:,:2]]))
        env=original_factory(ref)
        for event in events:
            tick=event['command_segment'];gate_index=event['gate'];gate=env.corridor.gates[gate_index]
            env.position=np.asarray(event['point']);env.held=actions[min(tick//20,len(actions)-1)]
            points=[];original=[]
            for frontier in (gate_index,gate_index+1):
                env.tracker.frontier=frontier
                points.append(continuous_gate_target(env));original.append(env.guidance_target())
            assert np.linalg.norm(points[1]-points[0])<1e-12
            pair=[]
            for sign in (-1,1):
                env.position=np.asarray(event['point'])+sign*1e-6*gate.normal
                env.tracker.frontier=gate_index+(sign>0)
                pair.append(continuous_gate_target(env))
                env.position=np.asarray(event['point'])+sign*(GATE_TRANSITION_WIDTH_MM/2+1e-6)*gate.normal
                assert np.array_equal(continuous_gate_target(env),env.guidance_target())
            event_checks.append(dict(path=path,gate=gate_index,tick=tick,
                original_same_position_target_jump_mm=float(np.linalg.norm(original[1]-original[0])),
                new_same_position_target_jump_mm=float(np.linalg.norm(points[1]-points[0])),
                new_two_nanometre_probe_change_mm=float(np.linalg.norm(pair[1]-pair[0]))))
        if path in (8,15):
            anchor=events[0]['command_segment']//20
            for decision in (0,anchor,anchor+1):
                old=reconstruct(ref,rows,actions,decision*20,costs)
                default=sensor_factory(ref);default.restore(old.state())
                a=p.act(default.observation(),True)[0];costs['default_regression_actor_calls']+=1
                assert np.array_equal(default.decode_action(a),actions[decision])
                _,_,_,info=default.step(a);costs['default_regression_physics_ticks']+=len(info['rows'])
                assert np.array_equal(info['rows'],rows[decision*20:decision*20+len(info['rows'])])
                regression.append(dict(path=path,decision=decision,rows_bitwise_equal=True))
    write_json(stage/'preflight.json',dict(events=event_checks,default_regression=regression,
        original_tracking_sha256=sha(ROOT/'论文项目/revision_20260913_preview_efficiency_v2/no_kcm_coverage_v2/tracking.py')))
    ids=[r['original_path'] for r in selected['results']]
    paths=read(BASE/'learning_preparation_v4/paths.json')['training'];candidates=[]
    for tag,model in models:
        p.load_state_dict(torch.load(model,map_location='cpu',weights_only=False)['policy'])
        rs=evaluate(p,[np.asarray(paths[i]) for i in ids],stage,tag,None,costs,start+150,environment_factory)
        rs=[dict(r,original_path=i) for r,i in zip(rs,ids)];traces=[stage/tag/f'{j}.npz' for j in range(len(ids))]
        value=review(rs,traces);metrics=metric_rows(rs,traces,value,NAMES)
        candidates.append(dict(tag=tag,source_model=str(model.relative_to(BASE)),summary=value,metrics=metrics,
            results=rs,traces=[str(t.relative_to(BASE)) for t in traces]))
        write_json(stage/'results.json',dict(candidates=candidates,metric_names=NAMES,costs=costs,
            guide_change_only=True,training_performed=False,goal_complete=False,promoted=False))
        print('CONTINUOUS_GATE_GUIDE',tag,{k:value[k] for k in ('successes','entry_sum_mm','exit_sum_mm','full_reverse_sum_deg','full_time_s')},flush=True)
        if value['successes']!=len(ids):break
    assert costs['evaluation_actor_calls']<=6000 and costs['evaluation_physics_ticks']<=140000
    assert costs['saved_prefix_geometry_ticks']<=10000
    receipt(stage,dict(status='completed_continuous_gate_guidance_mechanism_probe',costs=costs,
        gate_events_checked=len(event_checks),candidates=len(candidates),wall_s=time.perf_counter()-start,goal_complete=False))


if __name__=='__main__':main()
