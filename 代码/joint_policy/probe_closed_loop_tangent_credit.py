"""Validate a local closed-loop state-transition credit model, no policy update."""
import copy
import math
import time
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy, checkpoint_environment_factory
from probe_coordinated_suffix import reconstruct
from prepare_current_waveform_credit import wave_signals
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha

STATE_SCALE=np.array([1.25,1.25,100.,100.,2000.,2000.,1.])
ACTION_SCALE=np.array([.025,.3,.3])


def normalized_state(state):
    return np.r_[state['position'],state['velocity'],state['acceleration'],state['heading']]/STATE_SCALE


def shift_state(state,axis,amount):
    value=copy.deepcopy(state)
    field,index=(('position',axis) if axis<2 else ('velocity',axis-2) if axis<4
                 else ('acceleration',axis-4) if axis<6 else ('heading',None))
    if index is None:value[field]+=amount*STATE_SCALE[axis]
    else:value[field][index]+=amount*STATE_SCALE[axis]
    value['tracker']['position']=value['position'].copy()
    return value


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_coherent_parameter_credit_20260917'
    data=read(source/'results.json');model=BASE/data['parent_model']
    metadata=torch.load(model,map_location='cpu',weights_only=False)
    assert sha(model)==data['parent_sha256']
    stage=register('canonical_gate_band_closed_loop_tangent_credit_20260917',
        [__file__,model,source/'basis.npz',source/'results.json',source/'receipt.json',
         BASE/'canonical_gate_state.py',BASE/'gate_band_state.py',BASE/'joint_runtime.py',
         BASE/'probe_coordinated_suffix.py',BASE/'prepare_current_waveform_credit.py'],
        dict(training_paths=[14,15],difference_scales=[1e-4,5e-5],
             state_variables=7,action_variables=3,actor_calls=4500,
             new_physics_ticks=120000,saved_prefix_geometry_ticks=5000,wall_s=180,
             policy_updates=0,comparison_variants_reused=12),
        'Change credit construction rather than extend the failed six-direction '
        'search. Numerically linearize each ordinary20ms closed-loop transition '
        'in normalized physical position, velocity, acceleration and filtered '
        'heading, including policy state feedback. Add separate open-loop action '
        'Jacobians and analytic direct residual-parameter injections, then chain '
        'them along the exact saved current-parent states. Two finite scales '
        'test derivative sensitivity. Feasible one-sided derivatives are reported '
        'where the physical input viability boundary rejects one sign; both-sign '
        'failure invalidates that model. These local off-trajectory states are '
        'derivative queries, not complete candidate successes. Compare predicted '
        'preterminal1ms waveforms with all12 ALREADY completed coherent parameter '
        'variants on paths14/15. Stop before terminal initiation because its '
        'discrete trigger needs separate treatment. No terminal-time credit, '
        'full-path acceptance, policy update, new directions or constraint change. '
        'Proceed only if both-scale predictions are stable and measured waveform '
        'agreement supports a useful local model; otherwise report failure and stop.')
    start=time.perf_counter();costs=dict(actor_calls=0,new_physics_ticks=0,saved_prefix_geometry_ticks=0)
    policy=CanonicalGatePolicy(256);policy.load_state_dict(metadata['policy'])
    factory=checkpoint_environment_factory(metadata['architecture'])
    with np.load(source/'basis.npz') as z:
        basis=z['basis'].reshape(6,3,256).copy()*z['amplitudes'][:,None,None]
    records=[]
    def check_budget():
        if time.perf_counter()-start>175:raise TimeoutError('Tangent-credit wall budget')
        assert costs['actor_calls']<4500 and costs['new_physics_ticks']<117880
    try:
        for path in (14,15):
            j=next(j for j,r in enumerate(data['baseline']['results']) if r['original_path']==path)
            with np.load(BASE/data['baseline']['traces'][j]) as z:
                ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
            states=[]
            env=reconstruct(ref,rows,actions,(len(actions)-1)*20,costs,
                            state_snapshots=states,env_factory=factory)
            states.append(env.state());last_decision=len(actions)-1
            assert rows[last_decision*20,12]==1
            comparison=[]
            for candidate in data['candidates']:
                with np.load(BASE/candidate['traces'][j]) as z:actual=z['rows'].copy()
                assert candidate['results'][j]['original_path']==path
                comparison.append((candidate,actual))
            for epsilon in (1e-4,5e-5):
                env=factory(ref);sensitivity=np.zeros((7,6));row_sens=[]
                state_jacs=[];row_jacs=[];modes=[];failures=[]
                for k in range(last_decision):
                    check_budget();state=states[k];env.restore(state)
                    obs=env.observation();latent=policy.act(obs,True)[0].astype(float);costs['actor_calls']+=1
                    assert np.array_equal(env.decode_action(latent),actions[k])
                    nominal_state=normalized_state(states[k+1])
                    nominal_rows=rows[k*20:(k+1)*20,:6]
                    sj=np.zeros((7,10));rj=np.zeros((20,6,10))
                    for axis in range(10):
                        pair=[]
                        for sign in (-1,1):
                            check_budget()
                            query=shift_state(state,axis,sign*epsilon) if axis<7 else state
                            env.restore(query)
                            action=latent.copy()
                            if axis<7:
                                action=policy.act(env.observation(),True)[0];costs['actor_calls']+=1
                            else:action[axis-7]+=sign*epsilon*ACTION_SCALE[axis-7]
                            try:
                                _,_,_,info=env.step(action)
                                costs['new_physics_ticks']+=len(info['rows'])
                                valid=len(info['rows'])==20 and not env.done and not info['rows'][:,12].any()
                                if not valid:raise ValueError('Local transition changed termination or length')
                                pair.append((normalized_state(env.state()),info['rows'][:,:6].copy()))
                            except ValueError as exc:
                                pair.append(None);failures.append(dict(decision=k,axis=axis,sign=sign,error=str(exc)))
                        if all(v is not None for v in pair):
                            sj[:,axis]=(pair[1][0]-pair[0][0])/(2*epsilon)
                            rj[:,:,axis]=(pair[1][1]-pair[0][1])/(2*epsilon);mode='central'
                        elif any(v is not None for v in pair):
                            good=0 if pair[0] is not None else 1;sign=(-1,1)[good]
                            sj[:,axis]=(pair[good][0]-nominal_state)/(sign*epsilon)
                            rj[:,:,axis]=(pair[good][1]-nominal_rows)/(sign*epsilon);mode='one_sided'
                        else:raise ValueError(f'No valid state-action derivative at path{path} decision{k} axis{axis}')
                        modes.append(mode)
                    with torch.no_grad():
                        features=policy.residual_features(torch.tensor(obs)).numpy()*policy.residual_scale.numpy()[:,None]
                    direct=np.einsum('hk,jhk->hj',features,basis)/ACTION_SCALE[:,None]
                    row_sens.append(np.einsum('tcq,qj->tcj',rj[:,:,:7],sensitivity)+
                                    np.einsum('tch,hj->tcj',rj[:,:,7:],direct))
                    sensitivity=sj[:,:7]@sensitivity+sj[:,7:]@direct
                    state_jacs.append(sj);row_jacs.append(rj)
                jacobian=np.concatenate(row_sens);count=len(jacobian)
                nominal,valid=wave_signals(ref,rows[:count])
                errors=[]
                for candidate,actual in comparison:
                    length=min(count,len(actual));direction=candidate['direction'];sign=candidate['sign']
                    predicted_rows=rows[:length].copy()
                    predicted_rows[:,:6]+=sign*jacobian[:length,:,direction]
                    predicted,pred_valid=wave_signals(ref,predicted_rows)
                    truth,truth_valid=wave_signals(ref,actual[:length])
                    common=valid[:length]&pred_valid&truth_valid
                    rmse=[];signal=[]
                    for channel in range(6):
                        keep=common[:,channel]
                        rmse.append(float(np.sqrt(np.mean((predicted[keep,channel]-truth[keep,channel])**2))))
                        signal.append(float(np.sqrt(np.mean((truth[keep,channel]-nominal[:length][keep,channel])**2))))
                    errors.append(dict(direction=direction,sign=sign,rmse=rmse,measured_change_rms=signal))
                name=f'path{path}_eps{epsilon:g}.npz'
                np.savez_compressed(stage/name,reference=ref,rows=rows[:count],row_sensitivity=jacobian,
                                    state_jacobian=state_jacs,local_row_jacobian=row_jacs)
                record=dict(path=path,epsilon=epsilon,file=name,ordinary_decisions=last_decision,
                    central_derivatives=modes.count('central'),one_sided_derivatives=modes.count('one_sided'),
                    invalid_query_details=failures,errors=errors)
                records.append(record)
                write_json(stage/'results.json',dict(parent_sha256=sha(model),records=records,costs=costs,
                    preterminal_only=True,goal_complete=False))
                print('CLOSED_LOOP_TANGENT',path,epsilon,'one_sided',record['one_sided_derivatives'],
                    'heading_error_sum',sum(r['rmse'][2] for r in errors),
                    'heading_signal_sum',sum(r['measured_change_rms'][2] for r in errors),flush=True)
        receipt(stage,dict(status='completed_closed_loop_tangent_credit_comparison',
            records=len(records),costs=costs,wall_s=time.perf_counter()-start,policy_updates=0))
    except Exception as exc:
        receipt(stage,dict(status='failed_closed_loop_tangent_credit_comparison',error=str(exc),
            records=len(records),costs=costs,wall_s=time.perf_counter()-start,policy_updates=0))
        raise


if __name__=='__main__':main()
