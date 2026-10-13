"""One-decision causal tests of steering changes versus preview coupling."""
import time
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy,checkpoint_environment_factory
from probe_coordinated_suffix import reconstruct,encode
from audit_current_command_coordinates import command_signals
from complete_corner_metrics import complete_corner_metrics
from complete_motion_metrics import complete_motion_metrics
from validate_actions import difference_audit
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    torch.set_num_threads(1)
    parent=BASE/'canonical_gate_band_physical_cruise_small_learning_20260917'
    audit=BASE/'canonical_gate_band_implicit_lateral_target_audit_20260917'
    data=read(parent/'results.json');selected=data['full_candidate'];model=parent/data['chosen']['model']
    information=read(audit/'results.json');assert information['parent_sha256']==sha(model)
    cases=[3,6,14,15]
    stage=register('canonical_gate_band_implicit_target_time_probe_20260917',
        [__file__,model,parent/'results.json',parent/'receipt.json',audit/'results.json',audit/'receipt.json',
         BASE/'probe_coordinated_suffix.py',BASE/'joint_runtime_v2.py',BASE/'canonical_gate_state.py'],
        dict(paths=cases,arms=['previous_steering','previous_implicit_target'],intervention_decisions=1,
             suffixes=8,actor_calls=2000,new_physics_ticks=50000,saved_prefix_geometry_ticks=3000,wall_s=120),
        'At the first verified incoming-straight negative-bearing onset of four declared '
        'training corners, compare two single20ms changes: retain the preceding steering '
        'angle, or retain preceding q=d*tan(delta) by adjusting steering for the CURRENT '
        'learned preview. Current feed and preview stay unchanged in BOTH arms. Resume '
        'the unmodified learned policy and finish the entire route. The contrast isolates '
        'preview coupling within this one-decision temporal intervention; it does not '
        'prove how historical PPO learned the commands. Keep actual XY, reverse, entry/exit, '
        'time, cruise and all original physical checks. These probes are diagnostic only '
        'and cannot be deployed or promoted as learned trajectories.')
    start=time.perf_counter();costs=dict(actor_calls=0,new_physics_ticks=0,saved_prefix_geometry_ticks=0)
    meta=torch.load(model,map_location='cpu',weights_only=False)
    policy=CanonicalGatePolicy(256);policy.load_state_dict(meta['policy']);factory=checkpoint_environment_factory(meta['architecture'])
    results=[]
    for path in cases:
        event=next(r for r in information['records'] if r['path']==path)['events'][0]
        assert event['q_change_mm']<0 and event['onset_at_decision_start']
        anchor=event['decision'];frame=anchor*20
        j=next(j for j,r in enumerate(selected['results']) if r['original_path']==path)
        with np.load(BASE/selected['traces'][j]) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        initial=reconstruct(ref,rows,actions,frame,costs,env_factory=factory).state()
        original_shape=complete_corner_metrics(ref,rows)
        q_previous=actions[anchor-1,2]*np.tan(actions[anchor-1,0])
        for arm in ('previous_steering','previous_implicit_target'):
            env=factory(ref);env.restore(initial)
            latent=policy.act(env.observation(),True)[0];costs['actor_calls']+=1
            assert np.array_equal(env.decode_action(latent),actions[anchor])
            altered=actions[anchor].copy()
            altered[0]=actions[anchor-1,0] if arm=='previous_steering' else np.arctan(q_previous/altered[2])
            intervention=encode(altered)
            assert np.max(abs(env.decode_action(intervention)-altered))<1e-12
            chunks=[];held=[]
            while not env.done:
                assert time.perf_counter()-start<115 and costs['actor_calls']<2000 and costs['new_physics_ticks']<47880
                if chunks:latent=policy.act(env.observation(),True)[0];costs['actor_calls']+=1
                else:latent=intervention
                _,_,_,info=env.step(latent);chunks.append(info['rows']);held.append(info['held_action'])
                costs['new_physics_ticks']+=len(info['rows'])
            full=np.vstack([rows[:frame],*chunks]);all_actions=np.vstack([actions[:anchor],held])
            check=difference_audit(ref[0],full);success=bool(env.success and check['kinematic_pass'] and check['rest_pass'])
            signals=command_signals(ref,full,all_actions);name=f'path{path}_{arm}.npz'
            np.savez_compressed(stage/name,reference=ref,rows=full,actions=all_actions,anchor=anchor)
            result=dict(path=path,arm=arm,anchor=anchor,success=success,reason=env.reason,
                baseline_shape=original_shape,shape=complete_corner_metrics(ref,full),
                motion=complete_motion_metrics(ref,full),complete_time_s=len(full)*.001 if success else None,
                audit=check,original_action=actions[anchor].tolist(),intervention_action=altered.tolist(),
                first_decision_negative_target_ms=int((signals['combined_deg'][frame:frame+20]<-.05).sum()),trace=name)
            results.append(result)
            write_json(stage/'results.json',dict(parent_sha256=sha(model),records=results,costs=costs,diagnostic_only=True,goal_complete=False))
            print('TARGET_TIME_PROBE',path,arm,success,'negative_target_ms',result['first_decision_negative_target_ms'],
                'reverse',original_shape['full_physical_reverse_deg'],result['shape']['full_physical_reverse_deg'],flush=True)
    receipt(stage,dict(status='completed_implicit_target_time_coupling_probe',costs=costs,
        successes=sum(r['success'] for r in results),wall_s=time.perf_counter()-start,diagnostic_only=True))


if __name__=='__main__':main()
