"""Exact saved-state attribution of q changes to base and learned residual."""
import time
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy,checkpoint_environment_factory
from local_geometry_feedback_policy import LocalGeometryFeedbackPolicy
from probe_coordinated_suffix import reconstruct
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    torch.set_num_threads(1)
    parent=BASE/'canonical_gate_band_physical_cruise_small_learning_20260917'
    evidence=BASE/'canonical_gate_band_implicit_lateral_target_audit_20260917'
    data=read(parent/'results.json');pool=data['full_candidate'];model=parent/data['chosen']['model']
    events=read(evidence/'results.json');assert events['parent_sha256']==sha(model)
    paths=[3,6,14,15]
    stage=register('canonical_gate_band_implicit_target_policy_components_20260917',
        [__file__,model,parent/'results.json',parent/'receipt.json',evidence/'results.json',
         evidence/'receipt.json',BASE/'canonical_gate_state.py',BASE/'kernel_local_feedback_policy.py',
         BASE/'local_geometry_feedback_policy.py',BASE/'frame_feedback_policy.py',BASE/'probe_coordinated_suffix.py'],
        dict(new_physics_ticks=0,replayed_actor_calls=8,saved_prefix_geometry_ticks=3000,wall_s=45),
        'At the SAME four verified q-drop events, reconstruct the preceding and current '
        'saved observations. Separate the frozen base actor mean from the learned kernel '
        'residual, verify their exact recomposition, and use symmetric two-factor '
        'differences to attribute the actual q=d*tan(delta) change to their state responses. '
        'This algebraic attribution diagnoses the remaining policy function; it is not '
        'a new physical intervention or proof of historical optimizer causality.')
    start=time.perf_counter();costs=dict(saved_prefix_geometry_ticks=0,new_physics_ticks=0,replayed_actor_calls=0)
    metadata=torch.load(model,map_location='cpu',weights_only=False);p=CanonicalGatePolicy(256);p.load_state_dict(metadata['policy'])
    factory=checkpoint_environment_factory(metadata['architecture']);records=[]
    for path in paths:
        event=next(r for r in events['records'] if r['path']==path)['events'][0];k=event['decision']
        j=next(j for j,r in enumerate(pool['results']) if r['original_path']==path)
        with np.load(BASE/pool['traces'][j]) as z:ref,rows,actions=[z[n].copy() for n in ('reference','rows','actions')]
        states=[];env=reconstruct(ref,rows,actions,k*20,costs,state_snapshots=states,env_factory=factory);states.append(env.state())
        parts=[]
        for index in (k-1,k):
            env.restore(states[index]);obs=torch.tensor(env.observation())
            with torch.no_grad():
                full=p.distribution(obs).mean.numpy().astype(float)
                base=LocalGeometryFeedbackPolicy.distribution(p,obs).mean.numpy().astype(float)
                residual=(p.residual_features(obs)*p.residual_coefficients).sum(-1).numpy()*p.residual_scale.numpy()
            assert np.max(abs(full-base-residual))<2e-7
            assert np.array_equal(env.decode_action(full),actions[index]);costs['replayed_actor_calls']+=1
            parts.append((base,full-base))
        sign=np.sign(np.cross(ref[1]-ref[0],ref[2]-ref[1]))
        def q(z):
            a=env.decode_action(z);return float(sign*a[2]*np.tan(a[0]))
        b0,r0=parts[0];b1,r1=parts[1]
        q00,q01,q10,q11=q(b0+r0),q(b0+r1),q(b1+r0),q(b1+r1)
        base_change=.5*(q10-q00+q11-q01);residual_change=.5*(q01-q00+q11-q10)
        assert abs(base_change+residual_change-event['q_change_mm'])<1e-10
        result=dict(path=path,decision=k,q_change_mm=q11-q00,
            frozen_base_state_response_mm=base_change,learned_residual_state_response_mm=residual_change,
            base_latent_before=b0.tolist(),base_latent_after=b1.tolist(),
            residual_latent_before=r0.tolist(),residual_latent_after=r1.tolist())
        records.append(result);print('Q_POLICY_COMPONENTS',path,result,flush=True)
    write_json(stage/'results.json',dict(parent_sha256=sha(model),records=records,costs=costs,goal_complete=False))
    receipt(stage,dict(status='completed_implicit_target_policy_component_audit',costs=costs,wall_s=time.perf_counter()-start))


if __name__=='__main__':main()
