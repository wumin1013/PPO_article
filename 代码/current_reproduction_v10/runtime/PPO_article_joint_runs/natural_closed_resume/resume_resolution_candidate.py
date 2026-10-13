"""Exact saved-state replay and continuation at the registered execution step."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time,argparse,shutil,pickle
import numpy as np
import torch
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from deep_joint_credit_policy import load_credit_policy
from learned_action_pulse_policy import environment_factory
from run_natural_resolution import set_resolution,physical_audit
from natural_case_audit import analytic_command_audit
from natural_case_metrics import cruise_metrics
from study_io import sha,write_json
from arc_cruise_policy import build_policy
from cache_corner_contexts import memory_state,restore_memory


def restore_physics(env,prefix,index,ticks_per_decision):
    if index:
        for row in prefix['rows'][(index-1)*ticks_per_decision:index*ticks_per_decision]:env.tracker.update(row[:2])
        row=prefix['rows'][index*ticks_per_decision-1]
        env.position=row[:2].copy();env.velocity=row[2:4].copy();env.acceleration=row[4:6].copy()
        env.heading=float(np.arctan2(row[9],row[8]));env.held=prefix['actions'][index-1].copy()
    env.ticks=ticks_per_decision*index;env.decisions=index


def replay(policy,env,prefix,dt,deadline,cache_dir=None,fingerprint=None):
    ticks_per_decision=round(.020/dt);maximum=0.;obsmax=0.;last=time.perf_counter()
    assert len(prefix['rows'])==len(prefix['observations'])*ticks_per_decision
    assert np.all(prefix['rows'][:,12]==0.)
    policy.bind(env);start_index=0
    if cache_dir is not None:
        cache_dir=Path(cache_dir);cache_dir.mkdir(exist_ok=True)
        saved=sorted(cache_dir.glob('decisions_*.pkl'))
        if saved:
            with saved[-1].open('rb') as stream:checkpoint=pickle.load(stream)
            assert checkpoint['fingerprint']==fingerprint
            start_index=checkpoint['next_index'];assert 0<start_index<=len(prefix['observations'])
            env.restore(checkpoint['state']);restore_memory(policy,checkpoint['memory'])
            assert env.decisions==start_index-1
            maximum=checkpoint['action_max_error'];obsmax=checkpoint['observation_sample_max_error']
    for i in range(start_index,len(prefix['observations'])):
        x=prefix['observations'][i]
        if time.perf_counter()>deadline:raise TimeoutError('Resolution replay budget; no new motion emitted')
        restore_physics(env,prefix,i,ticks_per_decision)
        if i%100==0:obsmax=max(obsmax,float(np.max(abs(env.observation()-x))))
        action=env.decode_action(policy.act(x,True)[0],env.fixed_d)
        maximum=max(maximum,float(np.max(abs(action-prefix['actions'][i]))))
        assert maximum<1e-7 and obsmax<1e-6,(maximum,obsmax)
        if cache_dir is not None and ((i+1)%2000==0 or i+1==len(prefix['observations'])):
            checkpoint=dict(fingerprint=fingerprint,next_index=i+1,state=env.state(),memory=memory_state(policy),
                action_max_error=maximum,observation_sample_max_error=obsmax)
            file=cache_dir/f'decisions_{i+1:06d}.pkl';assert not file.exists()
            with file.open('wb') as stream:pickle.dump(checkpoint,stream,protocol=5)
        if time.perf_counter()-last>40:
            last=time.perf_counter();print('RESOLUTION_REPLAY',i,len(prefix['observations']),maximum,obsmax,flush=True)
    restore_physics(env,prefix,len(prefix['observations']),ticks_per_decision)
    assert maximum<1e-7 and obsmax<1e-6,(maximum,obsmax)
    return dict(action_max_error=maximum,observation_sample_max_error=obsmax,replayed_decisions=len(prefix['observations']),
        cached_verified_decisions=start_index,new_replayed_decisions=len(prefix['observations'])-start_index)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('resolution_folder');ap.add_argument('--stage-name',default='exact_continuation',choices=['exact_continuation','checkpoint_continuation']);args=ap.parse_args();parent=Path(args.resolution_folder)
    old=json.loads((parent/'result.json').read_text(encoding='utf-8'))
    assert not old['success'] and 'TimeoutError' in str(old.get('error'))
    assert old['geometry']['corridor_inside'] and old['audit']['kinematic_pass']
    dt=old['dt_s'];candidate=parent.parent
    selection=json.loads((candidate/'selected_policy.json').read_text(encoding='utf-8'))
    nominal_reg=json.loads((candidate/'registration.json').read_text(encoding='utf-8'))
    stage=parent/args.stage_name;stage.mkdir(exist_ok=False)
    start=time.perf_counter();torch.set_num_threads(1);set_resolution(dt)
    prefix=dict(np.load(parent/'trace.npz'));model=Path(selection['model']);assert sha(model)==selection['model_sha256']
    write_json(stage/'registration.json',dict(kind='exact_resolution_prefix_continuation',arm=nominal_reg['arm'],dt_s=dt,
        decision_dt_s=.020,wall_budget_s=1800,replay_deadline_s=1180,total_deadline_s=1720,ppo_updates=0,independent_repeat=False,
        prefix_sha256=sha(parent/'trace.npz'),source_sha256=sha(Path(__file__)),selection_sha256=sha(candidate/'selected_policy.json'),
        helper_sha256={str(p):sha(p) for p in (OUT/'cache_corner_contexts.py',BASE/'run_natural_resolution.py')}))
    shutil.copy2(__file__,stage/Path(__file__).name)
    policy=build_policy(load_credit_policy(torch.load(model,map_location='cpu',weights_only=False)),selection['spec'])
    env=environment_factory(prefix['reference'])
    if nominal_reg['arm']=='fixed':env.fixed_d=selection['spec']['fixed_d']
    fingerprint=dict(trace_sha256=sha(parent/'trace.npz'),model_sha256=sha(model),spec=selection['spec'],dt_s=dt,
        helpers={n:sha(OUT/n) for n in ('arc_cruise_policy.py','cruise_residual_policy.py','turn_feedback_policy.py')})
    try:identity=replay(policy,env,prefix,dt,start+1180,parent/'verified_replay_cache',fingerprint)
    except Exception as exc:
        write_json(stage/'receipt.json',dict(status='replay_not_verified_no_new_motion',error=f'{type(exc).__name__}: {exc}',wall_s=time.perf_counter()-start,goal_complete=False));raise
    identity['wall_s']=time.perf_counter()-start;write_json(stage/'replay_identity.json',identity)
    chunks=[prefix['rows']];observations=list(prefix['observations']);actions=list(prefix['actions']);times=list(prefix['decision_times_s'])
    last=time.perf_counter();error=None
    try:
        while not env.done:
            if time.perf_counter()>start+1720:raise TimeoutError('Resolution continuation wall budget')
            x=env.observation();latent=policy.act(x,True)[0];assert not np.any(env.pending_pulse)
            times.append(env.ticks*dt);observations.append(x)
            _,_,_,info=env.step(latent);chunks.append(info['rows']);actions.append(info['held_action'])
            if time.perf_counter()-last>40:
                last=time.perf_counter();print('RESOLUTION_CONTINUING',env.ticks*dt,env.tracker.frontier,flush=True)
    except Exception as exc:error=f'{type(exc).__name__}: {exc}'
    rows=np.vstack(chunks);reference=prefix['reference'];physical=physical_audit(reference,rows,dt);ordered=env.tracker.snapshot(1.25)
    geometry=json.loads((BASE/'paper_evidence_review/natural_closed_case/geometry.json').read_text(encoding='utf-8'))
    analytic=analytic_command_audit(reference[0],rows,geometry)
    success=bool(error is None and env.success and ordered['geometry_complete'] and physical['kinematic_pass'] and physical['rest_pass'] and analytic['analytic_corridor_pass'])
    np.savez_compressed(stage/'trace.npz',reference=reference,rows=rows,observations=np.array(observations),actions=np.array(actions),decision_times_s=np.array(times),dt_s=dt)
    result=dict(success=success,reason=env.reason,error=error,complete_time_s=env.ticks*dt if success else None,
        elapsed_prefix_or_complete_s=env.ticks*dt,dt_s=dt,decision_dt_s=.020,geometry=ordered,audit=physical,analytic=analytic,
        cruise=cruise_metrics(np.linalg.norm(rows[:,2:4],axis=1),dt=dt),physics_ticks=len(rows),actor_calls=len(actions),
        prefix_sha256=sha(parent/'trace.npz'),trace_sha256=sha(stage/'trace.npz'),spec=selection['spec'],exact_continuation=True)
    write_json(stage/'result.json',result);write_json(stage/'receipt.json',dict(status='complete_resolution_continuation',success=success,
        wall_s=time.perf_counter()-start,result_sha256=sha(stage/'result.json'),goal_complete=False,ppo_updates=0,independent_repeat=False))
    print('RESOLUTION_CONTINUATION_COMPLETE',success,result['complete_time_s'],time.perf_counter()-start,flush=True)


if __name__=='__main__':main()
