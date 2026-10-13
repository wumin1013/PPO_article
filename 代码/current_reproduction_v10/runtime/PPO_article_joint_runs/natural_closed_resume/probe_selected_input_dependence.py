"""Saved-state action sensitivity; no physical simulation and no training."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time,pickle,argparse
import numpy as np
import torch
OUT=Path(__file__).resolve().parent
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
sys.path[:0]=[str(BASE),str(OUT)]
from deep_joint_credit_policy import load_credit_policy
from learned_action_pulse_policy import environment_factory
from arc_cruise_policy import build_policy
from cache_corner_contexts import restore_memory,memory_state
from study_io import write_json,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');args=ap.parse_args()
    torch.set_num_threads(1);start=time.perf_counter()
    stage=OUT/'all34_dominance_research';cache=stage/'selected_causal_contexts'/('contexts.partial.pkl' if args.partial else 'contexts.pkl')
    with cache.open('rb') as f:contexts=pickle.load(f)
    audit=json.loads((stage/'existing_evidence_audit.json').read_text(encoding='utf-8'))
    z=np.load(OUT/'joint_actual_context_feedback_training/exact_continuation/full_trace.npz')
    rows,observations,actions=z['rows'],z['observations'],z['actions']
    metadata=torch.load(contexts[0]['model'],map_location='cpu',weights_only=False)
    records=[];maximum=0.
    for c in contexts:
        if time.perf_counter()>start+800:break
        stats=next(r for r in audit['records'] if r['id']==c['corner']['id'])
        sc=c['corner']['reference_s_mm']
        # Find the minimum requested feed in the registered physical window.
        import csv
        table=list(csv.DictReader((BASE/'paper_evidence_review/verified_results/all34_corner_statistics.csv').open(encoding='utf-8-sig')))
        rr=next(r for r in table if r['method']=='Joint' and r['id']==c['corner']['id'])
        lo,hi=int(rr['first_index'])//20,int(rr['last_index'])//20+1
        target=lo+int(np.argmin(actions[lo:hi,1]))
        policy=build_policy(load_credit_policy(metadata),c['source_spec']);env=environment_factory(c['reference'])
        policy.bind(env);env.restore(c['state']);restore_memory(policy,c['memory'])
        for i in range(c['state']['decisions'],target):
            latent=policy.act(observations[i],True)[0]
            maximum=max(maximum,float(np.max(abs(env.decode_action(latent,None)-actions[i]))))
            for r in rows[i*20:(i+1)*20]:env.tracker.update(r[:2])
            r=rows[(i+1)*20-1];env.position=r[:2].copy();env.velocity=r[2:4].copy();env.acceleration=r[4:6].copy()
            env.heading=float(np.arctan2(r[9],r[8]));env.held=actions[i].copy();env.ticks=(i+1)*20;env.decisions=i+1
        memory=memory_state(policy);x=observations[target];variants={}
        for label,indices in [('original',[]),('progress_zero',[16]),('gate_fraction_zero',[17]),('elapsed_zero',[19]),('global_triplet_zero',[16,17,19]),('memory_zero',[])]:
            restore_memory(policy,memory)
            if label=='memory_zero':
                obj=policy
                while obj is not None:
                    if hasattr(obj,'memory'):obj.memory[:]=0.;obj.previous_features[:]=0.
                    obj=getattr(obj,'base',None)
            xx=x.copy();xx[indices]=0.
            variants[label]=env.decode_action(policy.act(xx,True)[0],None).tolist()
        maximum=max(maximum,float(np.max(abs(np.array(variants['original'])-actions[target]))))
        rec=dict(corner=c['corner']['id'],decision=target,angle=stats['angle'],global_triplet=x[[16,17,19]].tolist(),actions=variants)
        records.append(rec)
        print('INPUT_PROBE',rec['corner'],'feed', {k:round(v[1],4) for k,v in variants.items()},flush=True)
    assert maximum<1e-7,maximum
    write_json(stage/('input_sensitivity_partial.json' if args.partial else 'input_sensitivity_all34.json'),dict(
        kind='one_decision_saved_state_input_sensitivity',physics_ticks=0,ppo_updates=0,records=records,identity_error=maximum,
        cache_sha256=sha(cache),wall_s=time.perf_counter()-start,interpretation='Counterfactual input substitutions are diagnosis only, not trained lower-dimensional policies or closed-loop results.'))

if __name__=='__main__':main()
