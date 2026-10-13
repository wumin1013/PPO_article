"""Saved-trace audit of sampled geometry discontinuities and actor responses."""
import time
import numpy as np
import torch
from local_geometry_feedback_policy import LocalGeometryFeedbackPolicy
from probe_coordinated_suffix import reconstruct
from joint_runtime import SENSOR_DISTANCES
from search_complete_return import read
from study_io import BASE, register, receipt, write_json


def main():
    torch.set_num_threads(1)
    parent=BASE/'exit_local_credit_parameter_learning_20260916'
    learning=BASE/'matched_single_decision_credit_learning_20260917'
    stage=register('preswing_sampled_geometry_event_audit_20260917',
        [__file__,parent/'candidate_1.pt',learning/'receipt.json',
         BASE/'single_decision_temporal_composition_audit_20260917/receipt.json',
         BASE/'joint_runtime.py',BASE/'joint_runtime_v2.py',BASE/'joint_runtime_v3.py',
         BASE/'local_geometry_feedback_policy.py',BASE/'probe_coordinated_suffix.py'],
        dict(new_physics_ticks=0,actor_calls=1000,wall_s=60),
        'Audit an additional, code-grounded source of finite-update nonlinearity: point-sampled '
        'reference tangent features jump at vertices although reference positions are continuous. '
        'Reconstruct saved nominal and updated states for path13 and path31 without integrating '
        'physics; verify every saved action against its model. Identify first sampled-direction '
        'event mismatch and evaluate actor at the same state with only the sampled tangent block '
        'changed. This latter substitution is a functional input attribution, not physical '
        'feasibility evidence. For isolated path13 also probe two actual geometric states1e-5mm '
        'either side of each in-approach sensor event, holding dynamics/gate fixed. No training '
        'or controller change. Only an ensuing separately registered physical intervention can '
        'attribute the observed preswing increase to a particular feature event.')
    started=time.perf_counter();costs=dict(saved_prefix_geometry_ticks=0,actor_calls=0,new_physics_ticks=0)
    records=[];events=[];continuous=[]
    policy=LocalGeometryFeedbackPolicy(mask_endpoint=True)
    for path,index in [(13,13),(31,18)]:
        series={}
        for tag,folder,trace_index,model in [
            ('parent',parent/'training_1',path,parent/'candidate_1.pt'),
            ('quarter',learning/'independent_zero_return_0.25',index,learning/'independent_zero_return_0.25.pt'),
            ('half',learning/'independent_zero_return_0.5',index,learning/'independent_zero_return_0.5.pt')]:
            with np.load(folder/f'{trace_index}.npz') as z:
                ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
            obs=[];env=reconstruct(ref,rows,actions,(len(actions)-1)*20,costs,observations=obs)
            obs.append(env.observation());obs=np.asarray(obs)
            policy.load_state_dict(torch.load(model,map_location='cpu',weights_only=False)['policy'])
            for x,a in zip(obs,actions):
                assert np.array_equal(env.decode_action(policy.act(x,True)[0]),a)
                costs['actor_calls']+=1
            series[tag]=(obs,rows,actions,ref,model)
            np.savez_compressed(stage/f'p{path}_{tag}.npz',observation=obs,rows=rows,actions=actions,reference=ref)
        original=series['parent']
        for tag in ('quarter','half'):
            obs,rows,actions,ref,model=series[tag]
            policy.load_state_dict(torch.load(model,map_location='cpu',weights_only=False)['policy'])
            m=min(len(obs),len(original[0]))
            changed=np.flatnonzero(np.max(abs(obs[:m,42:62]-original[0][:m,42:62]),axis=1)>.1)
            for anchor in changed:
                actual=policy.act(obs[anchor],True)[0]
                swapped=obs[anchor].copy();swapped[42:62]=original[0][anchor,42:62]
                alternate=policy.act(swapped,True)[0];costs['actor_calls']+=2
                events.append(dict(path=path,tag=tag,anchor=int(anchor),
                    actual_action=env.decode_action(actual).tolist(),
                    only_direction_block_replaced_action=env.decode_action(alternate).tolist(),
                    original_action=original[2][anchor].tolist(),
                    different_sensor_indices=np.flatnonzero(np.max(abs(obs[anchor,42:62].reshape(10,2)-
                        original[0][anchor,42:62].reshape(10,2)),axis=1)>.1).tolist()))
            records.append(dict(path=path,tag=tag,direction_event_mismatches=changed.tolist()))
        if path==13:
            obs,rows,actions,ref,model=original
            policy.load_state_dict(torch.load(model,map_location='cpu',weights_only=False)['policy'])
            u=ref[1]-ref[0];length=float(np.linalg.norm(u));u/=length
            for sensor,distance in enumerate(SENSOR_DISTANCES):
                if distance==0 or distance>=length:continue
                target_progress=length-distance
                at=int(np.argmin(abs((rows[:,:2]-ref[0])@u-target_progress)))//20
                env=reconstruct(ref,rows,actions,at*20,costs);state=env.state()
                values=[]
                for sign in (-1,1):
                    env.restore(state)
                    current=(env.position-ref[0])@u
                    env.position+=(target_progress+sign*1e-5-current)*u
                    x=env.observation();a=env.decode_action(policy.act(x,True)[0]);costs['actor_calls']+=1
                    values.append(dict(sign=sign,action=a.tolist(),observation=x.tolist()))
                continuous.append(dict(path=path,sensor=sensor,distance_mm=float(distance),anchor=at,
                    before=values[0],after=values[1],
                    action_jump=(np.array(values[1]['action'])-values[0]['action']).tolist()))
    assert costs['actor_calls']<=1000 and time.perf_counter()-started<60
    write_json(stage/'results.json',dict(records=records,events=events,geometric_limit_probes=continuous,costs=costs))
    receipt(stage,dict(status='completed_saved_geometry_feature_event_audit',costs=costs,wall_s=time.perf_counter()-started))
    print('EVENTS',records,events,flush=True)
    print('JUMPS',[(r['sensor'],r['action_jump']) for r in continuous],flush=True)


if __name__=='__main__':main()
