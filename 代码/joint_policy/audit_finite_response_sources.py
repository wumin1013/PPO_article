"""Telescope saved finite-response errors into actor and executor/guide residuals."""
import time
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy,checkpoint_environment_factory
from probe_coordinated_suffix import reconstruct
from probe_closed_loop_tangent_credit import normalized_state,shift_state,ACTION_SCALE
from smooth_credit_policy import CenteredSmoothActor
from transition_waveform_model import chain_responses
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_smooth_credit_completed_20260917'
    proposal=BASE/'canonical_gate_band_joint_regret_learning_20260917'
    data=read(source/'results.json');parent_model=BASE/data['parent_model']
    candidate_model=proposal/'candidate_1.pt'
    stage=register('canonical_gate_band_finite_response_sources_20260918',
        [__file__,source/'results.json',source/'receipt.json',parent_model,candidate_model,
         proposal/'results.json',proposal/'receipt.json',BASE/'smooth_credit_policy.py',
         BASE/'transition_waveform_model.py',BASE/'probe_closed_loop_tangent_credit.py'],
        dict(paths=[2,6],new_physics_ticks=0,saved_geometry_ticks=10000,
             actor_calls=2500,wall_s=120),
        'At each failing full-step exit peak, telescope measured decision-state deviations '
        'through the saved SAME730d local transition model. Sum each local defect propagated '
        'to that physical XY point, with the final within-decision defect, and require '
        'algebraic closure. Split each defect into the measured actor-latent nonlinear '
        'remainder propagated through the saved action Jacobian, and the remaining '
        'executor/guide/model remainder. Actor state derivatives use centered float64 '
        'differences at1e-4; actual actions stay the deployed float32 values. '
        'This is an exact error-accounting identity with an approximate local attribution, '
        'not independent causal intervention or proof that its components add under deployment. '
        'Use the result to decide which nonlinear response must be preserved in the next '
        'learning model. No parameter update, extra physics, error inflation or sealed access.')
    start=time.perf_counter();costs=dict(actor_calls=0,new_physics_ticks=0,saved_prefix_geometry_ticks=0)
    metadata=torch.load(parent_model,map_location='cpu',weights_only=False)
    newer=torch.load(candidate_model,map_location='cpu',weights_only=False)
    p=CanonicalGatePolicy(256);p.load_state_dict(metadata['policy'])
    q=CanonicalGatePolicy(256);q.load_state_dict(newer['policy'])
    smooth=CenteredSmoothActor(p);factory=checkpoint_environment_factory(metadata['architecture'])
    direction=(newer['policy']['residual_coefficients'].double()-metadata['policy']['residual_coefficients'].double()).numpy()
    chosen=next(c for c in read(proposal/'results.json')['candidates'] if c['step']==1.)
    records=[]
    for path in (2,6):
        local_record=next(r for r in data['records'] if r['path']==path)
        with np.load(source/local_record['file']) as z:local={k:z[k].copy() for k in z.files}
        j=next(j for j,r in enumerate(chosen['results']) if r['original_path']==path)
        with np.load(BASE/chosen['traces'][j]) as z:actual={k:z[k].copy() for k in ('reference','rows','actions')}
        ref=local['reference'];u,v=np.diff(ref,axis=0);sign=np.sign(np.linalg.det(np.stack([u,v])))
        v=v/np.linalg.norm(v);out=-sign*np.array([-v[1],v[0]])
        peak=int(np.argmax((actual['rows'][:,:2]-ref[1])@out));decision=peak//20
        assert not actual['rows'][peak,12] and not local['rows'][peak,12]
        snapshots=[]
        for saved in (local,actual):
            states=[];env=reconstruct(ref,saved['rows'],saved['actions'],(decision+1)*20,costs,
                                     state_snapshots=states,env_factory=factory)
            states.append(env.state());snapshots.append(states)
        delta=np.array([normalized_state(b)-normalized_state(a) for a,b in zip(*snapshots)])
        assert abs(delta[:,6]).max()<np.pi and abs(delta[0]).max()==0.
        actor_defects=[];total_defects=[];env=factory(ref)
        for k in range(decision+1):
            a,b=snapshots[0][k],snapshots[1][k]
            env.restore(a);env.observation_dtype=np.float32
            z0=p.act(env.observation(),True)[0].astype(float);costs['actor_calls']+=1
            assert np.array_equal(env.decode_action(z0),local['actions'][k])
            env.observation_dtype=np.float64;smooth.center(env.observation(),z0);costs['actor_calls']+=1
            jac=np.empty((3,7))
            for axis in range(7):
                pair=[]
                for polarity in (-1,1):
                    env.restore(shift_state(a,axis,polarity*1e-4));pair.append(smooth.act(env.observation())[0]);costs['actor_calls']+=1
                jac[:,axis]=(pair[1]-pair[0])/2e-4
            env.restore(b);env.observation_dtype=np.float32
            zb=q.act(env.observation(),True)[0].astype(float);costs['actor_calls']+=1
            assert np.array_equal(env.decode_action(zb),actual['actions'][k])
            direct=np.einsum('hk,hk->h',local['features'][k],direction)
            actor_defects.append(zb-z0-jac@delta[k]-direct)
            if k<decision:
                sj=local['state_jacobian'][k]
                total_defects.append(delta[k+1]-sj[:,:7]@delta[k]-sj[:,7:]@(direct/ACTION_SCALE))
            assert time.perf_counter()-start<115 and costs['actor_calls']<=2500
        response=chain_responses(local,direction.reshape(1,-1))[:,:,0]
        row_error=actual['rows'][peak,:6]-local['rows'][peak,:6]-response[peak]
        rj=local['row_jacobian'][peak];direct=np.einsum('hk,hk->h',local['features'][decision],direction)
        final_total=actual['rows'][peak,:6]-local['rows'][peak,:6]-rj[:,:7]@delta[decision]-rj[:,7:]@(direct/ACTION_SCALE)
        final_actor=rj[:,7:]@(actor_defects[decision]/ACTION_SCALE)
        contributions=[dict(decision=decision,time_s=decision*.02,
            actor_mm=float(out@final_actor[:2]),remaining_mm=float(out@(final_total-final_actor)[:2]),within_peak_decision=True)]
        adjoint=out@rj[:2,:7]
        for k in range(decision-1,-1,-1):
            sj=local['state_jacobian'][k]
            actor=sj[:,7:]@(actor_defects[k]/ACTION_SCALE)
            contributions.append(dict(decision=k,time_s=k*.02,actor_mm=float(adjoint@actor),
                remaining_mm=float(adjoint@(total_defects[k]-actor)),within_peak_decision=False))
            adjoint=adjoint@sj[:,:7]
        actor_sum=sum(r['actor_mm'] for r in contributions);remaining_sum=sum(r['remaining_mm'] for r in contributions)
        error=float(out@row_error[:2]);closure=abs(actor_sum+remaining_sum-error)
        assert closure<1e-9,(path,error,closure)
        record=dict(path=path,peak_tick=peak,finite_xy_prediction_error_at_actual_peak_mm=error,
            actor_nonlinear_remainder_contribution_mm=actor_sum,executor_guide_remaining_contribution_mm=remaining_sum,
            exact_telescope_closure_error_mm=closure,
            largest_contributions=sorted(contributions,key=lambda r:-abs(r['actor_mm']+r['remaining_mm']))[:8])
        records.append(record)
        np.savez_compressed(stage/f'path{path}_local_defects.npz',actor_latent_defects=actor_defects,
            normalized_state_defects=total_defects,state_deviations=delta)
        write_json(stage/'results.json',dict(parent_sha256=sha(parent_model),candidate_sha256=sha(candidate_model),
            records=records,costs=costs,causal_intervention=False,goal_complete=False))
        print('FINITE_RESPONSE_SOURCES',record,flush=True)
    receipt(stage,dict(status='completed_finite_response_source_accounting',costs=costs,
        wall_s=time.perf_counter()-start,goal_complete=False))


if __name__=='__main__':main()
