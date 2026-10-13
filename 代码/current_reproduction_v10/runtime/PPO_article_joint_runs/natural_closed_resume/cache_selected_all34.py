"""Exact causal state replay for all 34 corners of the current selected policy."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys, json, time, pickle, argparse, shutil
import numpy as np
import torch
OUT=Path(__file__).resolve().parent
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
sys.path[:0]=[str(BASE),str(OUT)]
from deep_joint_credit_policy import load_credit_policy
from learned_action_pulse_policy import environment_factory
from natural_case_metrics import ordered_projection
from study_io import sha,write_json
from arc_cruise_policy import build_policy
from cache_corner_contexts import memory_state

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['joint','fixed'],default='joint')
    ap.add_argument('--candidate-folder',default=None);ap.add_argument('--stage',default=None);args=ap.parse_args()
    torch.set_num_threads(1)
    stage=OUT/'all34_dominance_research'/(args.stage or ('selected_causal_contexts' if args.arm=='joint' else 'fixed_causal_contexts'))
    stage.mkdir(exist_ok=False)
    shutil.copy2(Path(__file__),stage/'source.py')
    start=time.perf_counter()
    folder=Path(args.candidate_folder) if args.candidate_folder else OUT/('joint_actual_context_feedback_training/exact_continuation' if args.arm=='joint' else 'fixed_curve_context_cruise_training')
    selection=json.loads((folder/'selected_policy.json').read_text(encoding='utf-8'))
    trace_sha=sha(folder/'full_trace.npz');selection_sha=sha(folder/'selected_policy.json')
    assert sha(Path(selection['model']))==selection['model_sha256']
    geometry=json.loads((BASE/'paper_evidence_review/natural_closed_case/geometry.json').read_text(encoding='utf-8'))
    write_json(stage/'registration.json',dict(kind='exact_existing_trace_replay',wall_budget_s=900,
        physics_ticks=0,ppo_updates=0,adoption='Every replayed action must agree within 1e-7; preserve all causal memories before intervention.',
        stop='Stop at 880 seconds or first action identity mismatch.',source_trace_sha256=trace_sha,
        source_selection_sha256=selection_sha,script_sha256=sha(Path(__file__))))
    z=np.load(folder/'full_trace.npz');rows=z['rows'];obs=z['observations'];actions=z['actions']
    progress=ordered_projection(z['reference'],rows[:,:2]);targets={}
    for c in geometry['sharp_corners']:
        i=int(np.searchsorted(progress,c['reference_s_mm']-40.))//20
        assert i not in targets
        targets[i]=c
    base_spec=selection.get('source_spec',selection.get('spec'))
    policy=build_policy(load_credit_policy(torch.load(selection['model'],map_location='cpu',weights_only=False)),base_spec)
    if 'cruise_spec' in selection:
        from coordinated_cruise_policy import CoordinatedCruisePolicy
        policy=CoordinatedCruisePolicy(policy,selection['cruise_spec'])
    env=environment_factory(z['reference'])
    if args.arm=='fixed':env.fixed_d=selection.get('fixed_d',base_spec['fixed_d'])
    policy.bind(env)
    layers=[];obj=policy
    while obj is not None:
        layers.append(dict(type=type(obj).__name__,has_memory=hasattr(obj,'memory'),
            memory_shape=list(obj.memory.shape) if hasattr(obj,'memory') else None,
            source_file=str(Path(sys.modules[type(obj).__module__].__file__).resolve())))
        obj=getattr(obj,'base',None)
    write_json(stage/'bound_policy_layers.json',layers)
    contexts=[];maximum=0.;obsmax=0.;last=start
    for i in range(max(targets)+1):
        if time.perf_counter()>start+880:raise TimeoutError('Exact causal replay wall budget')
        if i:
            for r in rows[(i-1)*20:i*20]:env.tracker.update(r[:2])
            r=rows[i*20-1];env.position=r[:2].copy();env.velocity=r[2:4].copy();env.acceleration=r[4:6].copy()
            env.heading=float(np.arctan2(r[9],r[8]));env.held=actions[i-1].copy()
        env.ticks=i*20;env.decisions=i
        if i in targets:
            c=targets[i];err=float(np.max(np.abs(env.observation()-obs[i])))
            obsmax=max(obsmax,err)
            assert err<1e-6,(i,err)
            contexts.append(dict(label='selected_'+c['id'],source_name='selected',source_folder=str(folder),
                source_trace_sha256=trace_sha,selection_sha256=selection_sha,source_spec=base_spec,source_cruise_spec=selection.get('cruise_spec'),
                fixed_d=env.fixed_d,model=selection['model'],model_sha256=selection['model_sha256'],
                state=env.state(),memory=memory_state(policy),observation=obs[i].copy(),corner=c,
                stop_s_mm=c['reference_s_mm']+50.,reference=z['reference'].copy()))
            with (stage/'contexts.partial.pkl').open('wb') as f:pickle.dump(contexts,f,protocol=5)
            print('CACHED',c['id'],'decision',i,'observation_error',err,flush=True)
        latent=policy.act(obs[i],True)[0];decoded=env.decode_action(latent,env.fixed_d)
        err=float(np.max(np.abs(decoded-actions[i])));maximum=max(maximum,err)
        assert err<1e-7,(i,err)
        if time.perf_counter()-last>40:
            print('EXACT_REPLAY',i,'of',max(targets),'actions_max_error',maximum,'wall_s',time.perf_counter()-start,flush=True)
            last=time.perf_counter()
    assert len(contexts)==34
    with (stage/'contexts.pkl').open('wb') as f:pickle.dump(contexts,f,protocol=5)
    write_json(stage/'receipt.json',dict(status='all34_exact_causal_contexts_ready',count=len(contexts),
        identity_max_decoded_error=maximum,context_observation_max_error=obsmax,wall_s=time.perf_counter()-start,
        physics_ticks=0,ppo_updates=0,contexts_sha256=sha(stage/'contexts.pkl'),goal_complete=False))
    print('ALL34_CONTEXTS_READY',maximum,obsmax,time.perf_counter()-start,flush=True)

if __name__=='__main__':main()
