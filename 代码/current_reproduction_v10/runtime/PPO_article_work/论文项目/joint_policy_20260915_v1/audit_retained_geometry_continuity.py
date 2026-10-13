"""Check whether the diagnosed sensing jump survives in the retained actor."""
import argparse
import time
import numpy as np
import torch
from joint_runtime import SENSOR_DISTANCES
from kernel_local_feedback_policy import KernelLocalFeedbackPolicy
from continuous_tangent_policy import interval_geometry
from probe_coordinated_suffix import reconstruct
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',default='retained_policy_geometry_continuity_audit_20260917')
    args=parser.parse_args()
    torch.set_num_threads(1)
    parent=BASE/'margin_constrained_function_credit_learning_20260917'
    selected=read(parent/'training_selection.json');model=parent/selected['model']
    stage=register(args.stage,
        [__file__,model,parent/'training_selection.json',BASE/'joint_runtime.py',
         BASE/'continuous_tangent_policy.py',BASE/'kernel_local_feedback_policy.py',
         BASE/'probe_coordinated_suffix.py',
         BASE/'preswing_sampled_geometry_causal_probe_20260917/receipt.json'],
        dict(paths=[3,13],new_physics_ticks=0,actor_calls=250,saved_geometry_ticks=40000,wall_s=60),
        'The old point-tangent discontinuity has causal evidence, but recent shape '
        'updates do not structurally remove it. Check the CURRENT retained64/64 actor '
        'at geometric limits on its own saved training states, not the old actor. '
        'Choose states ONLY while ordered frontier==0, and only sensor events within '
        'the actually represented incoming progress range. Unrestricted XY projection '
        'would incorrectly select a later outgoing pass on an obtuse corner. '
        'Hold dynamics and tracker state fixed, move position +/-1e-5mm around each '
        'incoming sensor/vertex crossing, and compare actions. Replace only tangent '
        'features with continuous interval geometry for a functional continuity '
        'contrast; this untrained input transform is not a rollout, candidate or '
        'deployed correction. Record jumps and other-input changes without claiming '
        'they explain every remaining reversal. No new physics or training.')
    started=time.perf_counter();costs=dict(saved_prefix_geometry_ticks=0,actor_calls=0,new_physics_ticks=0)
    p=KernelLocalFeedbackPolicy(256);p.load_state_dict(torch.load(model,map_location='cpu',weights_only=False)['policy'])
    records=[];skipped=[]
    for path in (3,13):
        with np.load(BASE/selected['traces'][path]) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        u=ref[1]-ref[0];length=float(np.linalg.norm(u));u/=length
        observations=[];states=[]
        env=reconstruct(ref,rows,actions,(len(actions)-1)*20,costs,
            observations=observations,state_snapshots=states)
        observations.append(env.observation());states.append(env.state())
        for x,a in zip(observations,actions):
            assert np.array_equal(env.decode_action(p.act(x,True)[0]),a)
            costs['actor_calls']+=1
        incoming=[j for j,state in enumerate(states) if state['tracker']['frontier']==0]
        progress=np.asarray([(states[j]['position']-ref[0])@u for j in incoming])
        for sensor,distance in enumerate(SENSOR_DISTANCES):
            if not 0<distance<length:continue
            location=length-distance
            if not progress.min()<=location<=progress.max():
                skipped.append(dict(path=path,distance_mm=float(distance),reason='outside_saved_incoming_progress_range'))
                continue
            anchor=incoming[int(np.argmin(abs(progress-location)))];state=states[anchor]
            observations=[]
            for sign in (-1,1):
                env.restore(state)
                env.position+=(location+sign*1e-5-(env.position-ref[0])@u)*u
                observations.append(env.observation())
            x=torch.tensor(np.asarray(observations));continuous=interval_geometry(x)
            with torch.no_grad():
                actual=p.distribution(x).mean.numpy();alternative=p.distribution(continuous).mean.numpy()
            costs['actor_calls']+=4
            actual=np.array([env.decode_action(a) for a in actual])
            alternate=np.array([env.decode_action(a) for a in alternative])
            other=np.r_[np.arange(42),np.arange(62,72)]
            records.append(dict(path=path,sensor=sensor,distance_mm=float(distance),anchor=anchor,
                observation_before=observations[0].tolist(),observation_after=observations[1].tolist(),
                original_actions=actual.tolist(),original_jump=(actual[1]-actual[0]).tolist(),
                interval_input_actions=alternate.tolist(),interval_input_jump=(alternate[1]-alternate[0]).tolist(),
                maximum_other_input_change=float(abs(x[1,other]-x[0,other]).max()),
                original_tangent_block_change=float(abs(x[1,42:62]-x[0,42:62]).max()),
                interval_block_change=float(abs(continuous[1,42:62]-continuous[0,42:62]).max())))
    assert costs['actor_calls']<=250 and costs['saved_prefix_geometry_ticks']<=40000 and time.perf_counter()-started<60
    write_json(stage/'results.json',dict(records=records,skipped=skipped,costs=costs,model_sha256=sha(model),
        functional_contrast_only=True,goal_complete=False))
    receipt(stage,dict(status='completed_retained_policy_geometry_continuity_audit',costs=costs,
        wall_s=time.perf_counter()-started))
    for r in records:
        if r['distance_mm'] in (7.,10.):
            print('RETAINED_GEOMETRY_JUMP',r['path'],r['distance_mm'],r['original_jump'],r['interval_input_jump'],flush=True)


if __name__=='__main__':main()
