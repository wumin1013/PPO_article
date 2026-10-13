"""Keep nonlinear actor feedback inside a fixed local physical transition model."""
import copy
import time
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy,checkpoint_environment_factory
from probe_coordinated_suffix import reconstruct
from probe_closed_loop_tangent_credit import shift_state,STATE_SCALE,ACTION_SCALE
from smooth_credit_policy import CenteredSmoothActor
from transition_waveform_model import chain_responses
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


def displaced_state(state,delta):
    value=copy.deepcopy(state);physical=delta*STATE_SCALE
    value['position']+=physical[:2];value['velocity']+=physical[2:4]
    value['acceleration']+=physical[4:6];value['heading']+=physical[6]
    value['tracker']['position']=value['position'].copy()
    return value


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_smooth_credit_completed_20260917'
    small=BASE/'canonical_gate_band_joint_regret_composition_20260917'
    large=BASE/'canonical_gate_band_joint_regret_learning_20260917'
    ds=read(source/'results.json');model=BASE/ds['parent_model']
    stage=register('canonical_gate_band_actor_exact_prediction_20260918',
        [__file__,source/'results.json',source/'receipt.json',model,small/'results.json',
         large/'results.json',large/'receipt.json',BASE/'canonical_gate_band_joint_regret_solve_20260917/direction.npz',
         BASE/'canonical_gate_band_finite_response_sources_20260918/results.json',
         BASE/'smooth_credit_policy.py',BASE/'closed_loop_transition_credit.py'],
        dict(paths=[2,6,7],scales=[0.,-.125,.125,.5,1.],new_physics_ticks=0,
             saved_geometry_ticks=12000,actor_calls=7000,wall_s=120),
        'Saved-error telescoping attributes most path2/6 finite exit error to omitted actor nonlinear feedback. '
        'Single-factor predictor test: retain the complete centered-float64 actor evaluation in the propagated '
        'state; remove its nominal state Jacobian from the saved closed-loop transition to recover a local '
        'executor/guide Jacobian. All physical/guide derivatives, observations, policy coefficients and source '
        '730d center remain fixed. Compare against the original affine predictor at already saved signed '
        'small and0.5/1 full-policy updates. Zero update must reproduce nominal ordinary rows. '
        'Path7 is an additional transfer control. Report ordinary-prefix XY prediction error and signed '
        'out-edge peak error; never score prefix time as completion. Support criterion: at both failing '
        'paths, at least50% lower absolute out-edge metric prediction error at0.5 and1, and correct '
        'regression sign when actual increase exceeds1e-6, with no more than10% XY error worsening on '
        'control path7. If unsupported, retain negative evidence and revise the mechanism. '
        'No new physical rollout, training, deployable teacher trajectory, margin inflation or sealed access.')
    start=time.perf_counter();costs=dict(actor_calls=0,new_physics_ticks=0,saved_prefix_geometry_ticks=0)
    md=torch.load(model,map_location='cpu',weights_only=False);policy=CanonicalGatePolicy(256);policy.load_state_dict(md['policy'])
    smooth=CenteredSmoothActor(policy);candidate=copy.deepcopy(smooth.policy);factory=checkpoint_environment_factory(md['architecture'])
    with np.load(BASE/'canonical_gate_band_joint_regret_solve_20260917/direction.npz') as z:direction=z['direction'].copy()
    variations={c['step']:c for folder in (small,large) for c in read(folder/'results.json')['candidates']}
    records=[]
    for path in (2,6,7):
        loc=next(r for r in ds['records'] if r['path']==path)
        with np.load(source/loc['file']) as z:local={k:z[k].copy() for k in z.files}
        ref,rows,actions=[local[k] for k in ('reference','rows','actions')];ordinary=len(actions)-1;length=ordinary*20
        snapshots=[];env=reconstruct(ref,rows,actions,length,costs,state_snapshots=snapshots,env_factory=factory)
        env=factory(ref);jacobians=[];latents=[];offsets=[]
        for k,state in enumerate(snapshots):
            env.restore(state);env.observation_dtype=np.float32
            latent=policy.act(env.observation(),True)[0].astype(float);costs['actor_calls']+=1
            assert np.array_equal(env.decode_action(latent),actions[k])
            env.observation_dtype=np.float64;smooth.center(env.observation(),latent);costs['actor_calls']+=1
            offsets.append(smooth.offset.copy());latents.append(latent)
            jac=np.empty((3,7))
            for axis in range(7):
                pair=[]
                for sign in (-1,1):
                    env.restore(shift_state(state,axis,sign*1e-4));pair.append(smooth.act(env.observation())[0]);costs['actor_calls']+=1
                jac[:,axis]=(pair[1]-pair[0])/2e-4
            jacobians.append(jac)
        jacobians=np.asarray(jacobians);response=chain_responses(local,direction.reshape(1,-1))[:,:,0]
        u,v=np.diff(ref,axis=0);turn_sign=np.sign(np.linalg.det(np.stack([u,v])))
        v=v/np.linalg.norm(v);out=-turn_sign*np.array([-v[1],v[0]])
        exit_value=lambda xy:float(max(0.,np.max((xy-ref[1])@out)))
        baseline_exit=exit_value(rows[:length,:2])
        for scale in (0.,-.125,.125,.5,1.):
            with torch.no_grad():candidate.residual_coefficients.copy_(smooth.policy.residual_coefficients+torch.tensor(scale*direction,dtype=torch.float64))
            delta=np.zeros(7);predictions=[]
            for k,state in enumerate(snapshots):
                env.restore(displaced_state(state,delta));env.observation_dtype=np.float64
                with torch.no_grad():latent=candidate.distribution(torch.as_tensor(env.observation(),dtype=torch.float64)).mean.numpy()+offsets[k]
                costs['actor_calls']+=1
                dz=(latent-latents[k])/ACTION_SCALE;ja=jacobians[k]/ACTION_SCALE[:,None]
                sj=local['state_jacobian'][k];rj=local['row_jacobian'][k*20:(k+1)*20]
                physical_row=rj[:,:,:7]-np.einsum('tch,hq->tcq',rj[:,:,7:],ja)
                pred=rows[k*20:(k+1)*20,:6]+np.einsum('tcq,q->tc',physical_row,delta)+np.einsum('tch,h->tc',rj[:,:,7:],dz)
                predictions.append(pred)
                delta=(sj[:,:7]-sj[:,7:]@ja)@delta+sj[:,7:]@dz
                assert np.isfinite(delta).all() and costs['actor_calls']<=7000 and time.perf_counter()-start<115
            nonlinear=np.concatenate(predictions);affine=rows[:length,:6]+scale*response[:length]
            if scale==0.:
                error=float(abs(nonlinear-rows[:length,:6]).max());assert error<1e-8,error
                records.append(dict(path=path,scale=scale,nominal_reproduction_error=error));continue
            variant=variations[scale];j=next(j for j,r in enumerate(variant['results']) if r['original_path']==path)
            with np.load(BASE/variant['traces'][j]) as z:actual=z['rows'][:length,:6]
            assert len(actual)==length
            measured=exit_value(actual[:,:2]);a=exit_value(affine[:,:2]);n=exit_value(nonlinear[:,:2])
            item=dict(path=path,scale=scale,actual_exit_change_mm=measured-baseline_exit,
                affine_exit_change_mm=a-baseline_exit,actor_exact_exit_change_mm=n-baseline_exit,
                affine_exit_error_mm=a-measured,actor_exact_exit_error_mm=n-measured,
                affine_xy_max_error_mm=float(np.linalg.norm(affine[:,:2]-actual[:,:2],axis=1).max()),
                actor_exact_xy_max_error_mm=float(np.linalg.norm(nonlinear[:,:2]-actual[:,:2],axis=1).max()))
            records.append(item)
            np.savez_compressed(stage/f'path{path}_scale{scale:g}.npz',affine_rows=affine,actor_exact_rows=nonlinear,actual_rows=actual,reference=ref)
            print('ACTOR_EXACT_PREDICTION',item,flush=True)
        np.savez_compressed(stage/f'path{path}_actor_jacobians.npz',jacobians=jacobians,latents=latents,offsets=offsets)
    target=[r for r in records if r['path'] in (2,6) and r['scale'] in (.5,1.)]
    controls=[r for r in records if r['path']==7 and r['scale']!=0.]
    supported=bool(all(abs(r['actor_exact_exit_error_mm'])<=.5*abs(r['affine_exit_error_mm']) for r in target)
        and all(r['actor_exact_exit_change_mm']>0 for r in target if r['actual_exit_change_mm']>1e-6)
        and all(r['actor_exact_xy_max_error_mm']<=1.1*r['affine_xy_max_error_mm'] for r in controls))
    write_json(stage/'results.json',dict(records=records,costs=costs,predictor_hypothesis_supported=supported,
        new_physical_rollouts=0,no_completion_claim_from_prefixes=True,goal_complete=False))
    receipt(stage,dict(status='completed_actor_exact_prediction_comparison',costs=costs,
        predictor_hypothesis_supported=supported,wall_s=time.perf_counter()-start,goal_complete=False))
    print('ACTOR_EXACT_SUPPORTED',supported,flush=True)


if __name__=='__main__':main()
