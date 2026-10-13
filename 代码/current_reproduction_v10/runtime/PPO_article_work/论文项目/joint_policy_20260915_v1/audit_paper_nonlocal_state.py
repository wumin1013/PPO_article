"""Matched saved-state queries: isolate nonlocal input sensitivity without simulation."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import json
import time
import numpy as np
import torch
from deep_joint_credit_policy import load_credit_policy
from learned_action_pulse_policy import environment_factory
from study_io import BASE,register,receipt,sha,write_json


def main():
    torch.set_num_threads(1);review=BASE/'paper_evidence_review'
    frozen=json.loads((review/'frozen_evaluation_candidate.json').read_text(encoding='utf-8'))
    model=BASE/frozen['model']; assert sha(model)==frozen['sha256']
    audit=json.loads((review/'candidate_audit.json').read_text(encoding='utf-8'))
    candidate=next(c for c in audit['candidates'] if c['label']=='worst_guard')
    observations=[]
    for row in candidate['records']:
        z=np.load(BASE/row['trace'])
        if 'observations' in z:observations.append(z['observations'])
    X=np.vstack(observations);lo=X.min(0);hi=X.max(0);median=np.median(X,axis=0)
    sources=[review/'public_long/trace.npz',review/'remaining_public_long/dense_corners_large.npz',
             review/'remaining_public_long/closed_line_arc_large.npz']
    stage=register('paper_evidence_review/nonlocal_state_diagnostic',
        [__file__,review/'frozen_evaluation_candidate.json',review/'candidate_audit.json',*sources],
        dict(new_simulation_ticks=0,training_updates=0,maximum_saved_actor_queries=1200),
        'Replay only saved observations and exact recorded dynamic states. Compare single-decision '
        'queries with nonlocal scalar interventions while holding history fixed. Diagnostic only; '
        'not a deployable policy, successful completion, training or generalization result.')
    started=time.perf_counter();metadata=torch.load(model,map_location='cpu',weights_only=False)
    cases=[];queries=0
    for source in sources:
        z=np.load(source);ref=z['reference'];rows=z['rows'];obs=z['observations'];actions=z['actions']
        modes=['original','remaining_clip','nonlocal_median']
        actors=[load_credit_policy(metadata) for _ in modes]
        envs=[p.bind(environment_factory(ref)) for p in actors]
        samples=[];max_identity=0.
        for i,x in enumerate(obs):
            # Every actor receives the same recorded history. Only the final query differs.
            last=i==len(obs)-1
            result={}
            for mode,policy,env in zip(modes,actors,envs):
                if i:
                    for r in rows[(i-1)*20:i*20]:env.tracker.update(r[:2])
                    r=rows[i*20-1]
                    env.position=r[:2].copy();env.velocity=r[2:4].copy();env.acceleration=r[4:6].copy()
                    env.heading=float(np.arctan2(r[9],r[8]));env.held=actions[i-1].copy()
                env.ticks=i*20;env.decisions=i
                query=x.copy()
                if last and mode=='remaining_clip':query[15]=np.clip(query[15],lo[15],hi[15])
                if last and mode=='nonlocal_median':query[[15,16,17,19]]=median[[15,16,17,19]]
                latent=policy.act(query,True)[0];queries+=1
                decoded=env.decode_action(latent)
                if mode=='original':max_identity=max(max_identity,float(np.max(np.abs(decoded-actions[i]))))
                if last:result[mode]=decoded.tolist()
            if last:samples.append(result)
        assert max_identity<2e-4,max_identity
        local=list(range(8))+[11,12]+list(range(22,72))
        scale=np.maximum(X.std(0),.05)
        x=obs[-1]
        distances=np.sqrt(np.mean(((X[:,local]-x[local])/scale[local])**2,axis=1))
        cases.append(dict(source=str(source.relative_to(BASE)),identity_max_decoded_error=max_identity,
            final_nonlocal_state=x[[15,16,17,19]].tolist(),nearest_local_training_rms=float(distances.min()),
            last_query=samples[0],deployed=False))
    result=dict(training_paths_with_saved_observations=len(observations),training_states=len(X),
        indices=[15,16,17,19],training_min=lo[[15,16,17,19]].tolist(),
        training_max=hi[[15,16,17,19]].tolist(),training_median=median[[15,16,17,19]].tolist(),
        cases=cases,actor_queries=queries,new_simulation_ticks=0,training_updates=0)
    assert queries<=1200
    write_json(stage/'results.json',result)
    receipt(stage,dict(status='completed_saved_state_nonlocal_input_diagnostic',
        results_sha256=sha(stage/'results.json'),actor_queries=queries,simulation_ticks=0,
        wall_s=time.perf_counter()-started,policy_adopted=False,sealed_test_opened=False))
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
