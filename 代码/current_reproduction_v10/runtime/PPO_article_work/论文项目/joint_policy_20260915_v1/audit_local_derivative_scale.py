"""Check local finite-difference accuracy only on a diagnosed training path."""
import argparse
import time
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy,checkpoint_environment_factory
from closed_loop_transition_credit import transition_jacobian
from probe_coordinated_suffix import reconstruct
from transition_waveform_model import chain_responses
from prepare_current_waveform_credit import wave_signals
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    parser=argparse.ArgumentParser()
    for name in ('collection','direction','stage'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--path',type=int,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    source=BASE/args.collection;direction_source=BASE/args.direction
    data=read(source/'results.json');record=next(r for r in data['records'] if r['path']==args.path)
    model=BASE/data['parent_model'];assert sha(model)==data['parent_sha256']
    assert read(direction_source/'optimizer.json')['parent_sha256']==sha(model)
    with np.load(source/record['file']) as z:local={k:z[k].copy() for k in z.files}
    with np.load(direction_source/'direction.npz') as z:direction=z['direction'].ravel()
    stage=register(args.stage,[__file__,model,source/'results.json',source/record['file'],
        direction_source/'direction.npz',direction_source/'optimizer.json',
        BASE/'closed_loop_transition_credit.py',BASE/'transition_waveform_model.py'],
        dict(paths=[args.path],epsilon=[.001,.00001],reused_epsilon=.0001,
            actor_calls=2200,new_physics_ticks=60000,wall_s=150),
        'Reuse the original1e-4 local derivatives and recompute only one diagnosed path at '
        '1e-3 and1e-5 normalized finite-difference scales. Keep the exact parent, direction, '
        'features, state construction and runtime unchanged. Compare the chained XY response '
        'at the measured exit extremum with the saved finite parameter responses. This '
        'distinguishes scale-sensitive derivative error from finite-step response; no '
        'policy acceptance or claim that one epsilon is universally correct.')
    metadata=torch.load(model,map_location='cpu',weights_only=False)
    p=CanonicalGatePolicy(256);p.load_state_dict(metadata['policy'])
    factory=checkpoint_environment_factory(metadata['architecture'])
    ref,rows,actions=[local[k] for k in ('reference','rows','actions')]
    costs=dict(actor_calls=0,new_physics_ticks=0,saved_prefix_geometry_ticks=0)
    start=time.perf_counter();states=[]
    env=reconstruct(ref,rows,actions,(len(actions)-1)*20,costs,state_snapshots=states,env_factory=factory)
    states.append(env.state());env=factory(ref)
    def check():
        assert costs['actor_calls']<2200 and costs['new_physics_ticks']<60000
        if time.perf_counter()-start>145:raise TimeoutError('Derivative-scale audit deadline')
    u,v=np.diff(ref,axis=0);u/=np.linalg.norm(u);v/=np.linalg.norm(v)
    sign=np.sign(u[0]*v[1]-u[1]*v[0]);normal=sign*np.array([-v[1],v[0]])
    wave,valid=wave_signals(ref,rows)
    peak=int(np.flatnonzero(valid[:,1])[np.argmin(wave[valid[:,1],1])])
    records=[]
    try:
        for epsilon in (.0001,.001,.00001):
            revised=dict(local);invalid=[]
            if epsilon!=.0001:
                sj=[];rj=[]
                for k,state in enumerate(states):
                    check();env.restore(state)
                    latent=p.act(env.observation(),True)[0].astype(float);costs['actor_calls']+=1
                    assert np.array_equal(env.decode_action(latent),actions[k])
                    a,b=local['row_offsets'][k:k+2];terminal=k==len(states)-1
                    s,r,modes,bad=transition_jacobian(env,p,state,latent,rows[a:b],
                        None if terminal else states[k+1],epsilon,terminal,costs,check)
                    sj.append(s);rj.append(r);invalid.extend(dict(decision=k,**q) for q in bad)
                revised.update(state_jacobian=np.asarray(sj),row_jacobian=np.concatenate(rj))
                np.savez_compressed(stage/f'path{args.path}_eps{epsilon:g}.npz',**revised)
            delta=chain_responses(revised,direction[None,:])[:,:,0]
            derivative=float(-normal@delta[peak,:2])
            records.append(dict(epsilon=epsilon,reused=epsilon==.0001,
                exit_peak_tick=peak,exit_response_per_unit=derivative,invalid_queries=invalid))
            print('DERIVATIVE_SCALE',records[-1]['epsilon'],derivative,flush=True)
            write_json(stage/'results.json',dict(parent_sha256=sha(model),path=args.path,
                records=records,costs=costs,goal_complete=False))
        receipt(stage,dict(status='completed_local_derivative_scale_audit',costs=costs,wall_s=time.perf_counter()-start))
    except Exception as exc:
        receipt(stage,dict(status='failed_local_derivative_scale_audit',error=str(exc),costs=costs,wall_s=time.perf_counter()-start))
        raise


if __name__=='__main__':main()
