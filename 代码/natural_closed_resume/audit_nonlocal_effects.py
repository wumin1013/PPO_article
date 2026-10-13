"""Exact saved-state replay with diagnostic-only input and memory interventions."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,time,copy
import numpy as np
import torch
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from deep_joint_credit_policy import load_credit_policy
from learned_action_pulse_policy import environment_factory
from natural_case_metrics import ordered_projection
from study_io import write_json,sha

def main():
    torch.set_num_threads(1); start=time.perf_counter()
    case=BASE/'paper_evidence_review/natural_closed_case'
    model=BASE/'canonical_gate_band_worst_guard_joint_learning_20260920/shadow_training_policy.pt'
    md=torch.load(model,map_location='cpu',weights_only=False);policy=load_credit_policy(md)
    z=np.load(case/'support_calibration/remaining_median.npz');rows=z['rows'];obs=z['observations'];actions=z['actions']
    env=policy.bind(environment_factory(z['reference']))
    geometry=json.loads((case/'geometry.json').read_text(encoding='utf-8'))
    progress=ordered_projection(z['reference'],rows[:,:2]);corner_s=np.array([c['reference_s_mm'] for c in geometry['sharp_corners']])
    upper=.1412694752216339
    chain=[];obj=policy
    while obj is not None:
        chain.append(dict(type=type(obj).__name__,memory_decays=getattr(obj,'metadata',{}).get('memory_decays',None)))
        obj=getattr(obj,'base',None)
    chain=json.loads(json.dumps(chain,default=lambda x:np.asarray(x).tolist()))
    write_json(OUT/'nonlocal_query_registration.json',dict(model=str(model),sha256=sha(model),trace_sha256=sha(case/'support_calibration/remaining_median.npz'),
        purpose='Same-state diagnostic only; every baseline decision and causal memory replayed. No physics, no fitting, no adoption.',
        wall_cap_s=1000,maximum_probe_states=240,chain=chain,
        execution_note='First replay stopped after prefix timing showed 480s cap insufficient; no simulation was run. This exact replay allows 1000s and saves incremental evidence.'))
    samples=[];maximum=0.;last=start
    for i,x0 in enumerate(obs):
        if time.perf_counter()>start+990:raise TimeoutError('Saved replay cap')
        if i:
            for r in rows[(i-1)*20:i*20]:env.tracker.update(r[:2])
            r=rows[i*20-1]
            env.position=r[:2].copy();env.velocity=r[2:4].copy();env.acceleration=r[4:6].copy()
            env.heading=float(np.arctan2(r[9],r[8]));env.held=actions[i-1].copy()
        env.ticks=i*20;env.decisions=i
        x=x0.copy();x[15]=min(x[15],upper)
        latent=policy.act(x,True)[0];decoded=env.decode_action(latent)
        maximum=max(maximum,float(np.max(abs(decoded-actions[i]))))
        if i%100==0:
            record=dict(decision=i,time_s=i*.02,s_mm=float(progress[min(i*20,len(rows)-1)]),
                        distance_to_corner_mm=float(np.min(abs(progress[min(i*20,len(rows)-1)]-corner_s))),
                        observation=x[[15,16,17,19]].tolist(),baseline=decoded.tolist(),queries={})
            for label,indices,value in [('progress_zero',[16],[0.]),('gate_zero',[17],[0.]),('time_zero',[19],[0.]),
                                         ('progress_gate_time_zero',[16,17,19],[0.,0.,0.]),
                                         ('nonlocal_median',[16,17,19],[.4548479617,.2399999946,.0868913829])]:
                q=x.copy();q[indices]=value
                record['queries'][label]=env.decode_action(policy.act(q,True)[0]).tolist()
            # Preserve accumulated causal memory while zeroing only its readout for a query.
            memories=[];obj=policy
            while obj is not None:
                if hasattr(obj,'memory'):
                    memories.append((obj,obj.memory.copy()));obj.memory[:]=0.
                obj=getattr(obj,'base',None)
            record['queries']['zero_memory']=env.decode_action(policy.act(x,True)[0]).tolist()
            for obj,memory in memories:obj.memory[:]=memory
            samples.append(record)
        if time.perf_counter()-last>40:
            print('SAVED_REPLAY',i,'/',len(obs),'identity_max',maximum,flush=True);last=time.perf_counter()
            write_json(OUT/'nonlocal_effects_progress.json',dict(decisions=i+1,identity_max_decoded_error=maximum,samples=samples,wall_s=last-start,complete=False))
    assert maximum<2e-4,maximum
    write_json(OUT/'nonlocal_effects.json',dict(identity_max_decoded_error=maximum,wall_s=time.perf_counter()-start,
        decisions=len(obs),samples=samples,chain=chain,new_simulation_ticks=0,training_updates=0,adopted=False))
    for quarter in range(4):
        part=[x for x in samples if quarter/4<=x['observation'][1]<(quarter+1)/4 and x['distance_to_corner_mm']>30]
        print('QUARTER',quarter+1,'n',len(part),'baseline',np.mean([x['baseline'][1] for x in part]),
              {label:np.mean([x['queries'][label][1] for x in part]) for label in samples[0]['queries']},flush=True)
    print('IDENTITY',maximum,'WALL',time.perf_counter()-start,flush=True)

if __name__=='__main__':main()
