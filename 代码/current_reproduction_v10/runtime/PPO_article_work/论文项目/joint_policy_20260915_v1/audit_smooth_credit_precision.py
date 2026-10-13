"""Centered smooth actor differences isolate float32 evaluation in credit."""
import argparse
import time
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy,checkpoint_environment_factory
from closed_loop_transition_credit import transition_jacobian
from probe_coordinated_suffix import reconstruct
from transition_waveform_model import chain_responses,signal_jacobians
from waveform_constraint_model import waveform_facets
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha
from smooth_credit_policy import CenteredSmoothActor


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',default='canonical_gate_band_smooth_credit_precision_20260917')
    args=parser.parse_args();torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_shape_cruise_derivative_repaired_transition_20260917'
    direction_source=BASE/'canonical_gate_band_shape_priority_solve_20260917'
    data=read(source/'results.json');model=BASE/data['parent_model']
    assert sha(model)==read(direction_source/'optimizer.json')['parent_sha256']
    stage=register(args.stage,[__file__,model,source/'results.json',direction_source/'direction.npz',
        direction_source/'optimizer.json',BASE/'joint_runtime.py',BASE/'canonical_gate_state.py',
        BASE/'closed_loop_transition_credit.py',BASE/'transition_waveform_model.py',
        BASE/'canonical_gate_band_shape_priority_composition_20260917/results.json'],
        dict(paths=[13,15],reused_float32_epsilon=.0001,smooth_epsilons=[.0001,.00001],
            actor_calls=4000,new_physics_ticks=110000,wall_s=180,policy_updates=0),
        'Both-sign actual parameter responses show wrong-sign physical-curvature '
        'credit on13/15. Isolate numerical differentiation of float32 observations '
        'and actor outputs: keep exact saved states, features, execution and actual '
        'float32 nominal action; evaluate only local actor differences in float64 '
        'around that center. Compare1e-4/1e-5 smooth estimates with existing '
        'float32 estimates and saved complete-policy responses. This tests the '
        'combined precision of state encoding and actor evaluation, not their '
        'separate contributions. No changed deployed dtype or controller. '
        'Do not inflate empirical errors, enlarge radius or accept the failed '
        'direction. A stable derivative must predict independent smaller '
        'complete-parameter responses before it can guide another solve.')
    meta=torch.load(model,map_location='cpu',weights_only=False)
    policy=CanonicalGatePolicy(256);policy.load_state_dict(meta['policy']);smooth=CenteredSmoothActor(policy)
    factory=checkpoint_environment_factory(meta['architecture'])
    with np.load(direction_source/'direction.npz') as z:direction=z['direction'].ravel()
    start=time.perf_counter();costs=dict(actor_calls=0,new_physics_ticks=0,saved_prefix_geometry_ticks=0);records=[]
    def check():
        assert costs['actor_calls']<4000 and costs['new_physics_ticks']<108000
        if time.perf_counter()-start>175:raise TimeoutError('Smooth-credit precision budget')
    try:
        for path in (13,15):
            record=next(r for r in data['records'] if r['path']==path)
            with np.load(source/record['file']) as z:local={k:z[k].copy() for k in z.files}
            ref,rows,actions=[local[k] for k in ('reference','rows','actions')]
            states=[];env=reconstruct(ref,rows,actions,(len(actions)-1)*20,costs,state_snapshots=states,env_factory=factory)
            states.append(env.state());env=factory(ref)
            for epsilon in (None,.0001,.00001):
                revised=dict(local);invalid=[];modes=[];center_max=0.
                if epsilon is not None:
                    sj=[];rj=[]
                    for k,state in enumerate(states):
                        check();env.restore(state);env.observation_dtype=np.float32
                        latent=policy.act(env.observation(),True)[0].astype(float);costs['actor_calls']+=1
                        assert np.array_equal(env.decode_action(latent),actions[k])
                        env.observation_dtype=np.float64
                        smooth.center(env.observation(),latent);costs['actor_calls']+=1
                        center_max=max(center_max,float(abs(smooth.act(env.observation())[0]-latent).max()))
                        costs['actor_calls']+=1
                        a,b=local['row_offsets'][k:k+2];terminal=k==len(states)-1
                        s,r,m,bad=transition_jacobian(env,smooth,state,latent,rows[a:b],
                            None if terminal else states[k+1],epsilon,terminal,costs,check)
                        sj.append(s);rj.append(r);modes.extend(m);invalid.extend(dict(decision=k,**q) for q in bad)
                    revised.update(state_jacobian=np.asarray(sj),row_jacobian=np.concatenate(rj))
                    name=f'path{path}_smooth_eps{epsilon:g}.npz'
                    np.savez_compressed(stage/name,**revised)
                else:name=str((source/record['file']).resolve().relative_to(BASE))
                delta=chain_responses(revised,direction[None,:])
                nominal,valid,gradient,_,_,_=signal_jacobians(ref,rows,delta)
                u,v=np.diff(ref,axis=0);turn=float(np.arctan2(u[0]*v[1]-u[1]*v[0],u@v)*180/np.pi)
                facets=waveform_facets(nominal,gradient,valid,turn,np.zeros(1))
                peak=int(np.flatnonzero(valid[:,4])[abs(nominal[valid[:,4],4]).argmax()])
                item=dict(path=path,precision='original_float32' if epsilon is None else 'centered_float64',
                    epsilon=.0001 if epsilon is None else epsilon,file=name,
                    physical_peak_tick=peak,physical_peak_response_per_unit=float(facets['full_physical_peak_curvature_mm_inv'][1][0]),
                    exit_response_per_unit=float(facets['out_edge_outward_mm'][1][0]),
                    reverse_response_per_unit=float(facets['full_physical_reverse_deg'][1][0]),
                    centered_nominal_error=center_max,invalid_queries=invalid,one_sided=modes.count('one_sided'))
                assert center_max<1e-15
                records.append(item)
                write_json(stage/'results.json',dict(parent_sha256=sha(model),records=records,costs=costs,goal_complete=False))
                print('SMOOTH_CREDIT',path,item['precision'],item['epsilon'],item['physical_peak_response_per_unit'],flush=True)
        receipt(stage,dict(status='completed_centered_smooth_credit_precision_audit',costs=costs,wall_s=time.perf_counter()-start,goal_complete=False))
    except Exception as exc:
        receipt(stage,dict(status='failed_centered_smooth_credit_precision_audit',error=str(exc),costs=costs,wall_s=time.perf_counter()-start,goal_complete=False))
        raise


if __name__=='__main__':main()
