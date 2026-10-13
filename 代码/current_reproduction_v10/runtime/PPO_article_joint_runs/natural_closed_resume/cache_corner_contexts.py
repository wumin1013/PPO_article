"""Replay saved decisions exactly and retain causal pre-corner states; no physics."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time,pickle,argparse
import numpy as np
import torch
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from deep_joint_credit_policy import load_credit_policy
from learned_action_pulse_policy import environment_factory
from natural_case_metrics import ordered_projection
from study_io import sha,write_json
from arc_cruise_policy import build_policy


def memory_state(policy):
    out=[];obj=policy
    while obj is not None:
        if hasattr(obj,'memory'):
            out.append(dict(memory=obj.memory.copy(),previous_features=obj.previous_features.copy(),last_decision=obj.last_decision))
        obj=getattr(obj,'base',None)
    return out


def restore_memory(policy,state):
    index=0;obj=policy
    while obj is not None:
        if hasattr(obj,'memory'):
            for key,value in state[index].items():setattr(obj,key,value.copy() if isinstance(value,np.ndarray) else value)
            index+=1
        obj=getattr(obj,'base',None)
    assert index==len(state)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fixed',action='store_true');args=ap.parse_args()
    torch.set_num_threads(1);stage=OUT/('actual_fixed_corner_contexts' if args.fixed else 'actual_corner_contexts');stage.mkdir(exist_ok=False)
    geometry=json.loads((BASE/'paper_evidence_review/natural_closed_case/geometry.json').read_text(encoding='utf-8'))
    cases=[('successful',OUT/'joint_arc_specific_cruise_training',['corner_01','corner_02','corner_06']),
           ('failed_recovery',OUT/'joint_separate_recovery_training',['corner_01','corner_02'])]
    if args.fixed:cases=[('successful_fixed',OUT/'fixed_curve_context_cruise_training',['corner_01','corner_02','corner_06'])]
    write_json(stage/'registration.json',dict(kind='exact_saved_decision_replay',wall_budget_s=900,physics_ticks=0,training_updates=0,
        purpose='Retain actual complete-route dynamics, global observations, gate state and all causal memory before representative corners. No synthetic episode reset.'))
    contexts=[];start=time.perf_counter();maximum=0.;last=start
    for name,folder,ids in cases:
        selection=json.loads((folder/'selected_policy.json').read_text(encoding='utf-8'))
        z=np.load(folder/'full_trace.npz');rows=z['rows'];observations=z['observations'];actions=z['actions']
        progress=ordered_projection(z['reference'],rows[:,:2]);targets={}
        for cid in ids:
            corner=next(c for c in geometry['sharp_corners'] if c['id']==cid)
            index=int(np.searchsorted(progress,corner['reference_s_mm']-40.))//20
            targets[index]=corner
        policy=build_policy(load_credit_policy(torch.load(selection['model'],map_location='cpu',weights_only=False)),selection['spec'])
        env=environment_factory(z['reference'])
        if args.fixed:env.fixed_d=selection['spec']['fixed_d']
        policy.bind(env)
        for i in range(max(targets)+1):
            if time.perf_counter()>start+880:raise TimeoutError('Saved context replay budget')
            if i:
                for r in rows[(i-1)*20:i*20]:env.tracker.update(r[:2])
                r=rows[i*20-1];env.position=r[:2].copy();env.velocity=r[2:4].copy();env.acceleration=r[4:6].copy()
                env.heading=float(np.arctan2(r[9],r[8]));env.held=actions[i-1].copy()
            env.ticks=i*20;env.decisions=i
            if i in targets:
                corner=targets[i];record=dict(label=name+'_'+corner['id'],source_name=name,source_folder=str(folder),
                    source_trace_sha256=sha(folder/'full_trace.npz'),selection_sha256=sha(folder/'selected_policy.json'),
                    source_spec=selection['spec'],fixed_d=env.fixed_d,model=selection['model'],model_sha256=selection['model_sha256'],
                    state=env.state(),memory=memory_state(policy),observation=observations[i].copy(),
                    corner=corner,stop_s_mm=corner['reference_s_mm']+50.,reference=z['reference'].copy())
                contexts.append(record)
            latent=policy.act(observations[i],True)[0];decoded=env.decode_action(latent,env.fixed_d)
            maximum=max(maximum,float(np.max(np.abs(decoded-actions[i]))))
            if time.perf_counter()-last>40:
                print('CONTEXT_REPLAY',name,i,'max_error',maximum,flush=True);last=time.perf_counter()
    assert maximum<1e-7,maximum
    with (stage/'contexts.pkl').open('wb') as f:pickle.dump(contexts,f,protocol=5)
    write_json(stage/'receipt.json',dict(status='exact_saved_contexts_retained',contexts=[c['label'] for c in contexts],
        identity_max_decoded_error=maximum,wall_s=time.perf_counter()-start,physics_ticks=0,training_updates=0,
        contexts_sha256=sha(stage/'contexts.pkl'),goal_complete=False))
    print('CONTEXTS_READY',len(contexts),maximum,time.perf_counter()-start,flush=True)


if __name__=='__main__':main()
