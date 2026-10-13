"""Compare parent-state and actually visited-state residual magnitudes."""
import numpy as np
import torch
from observable_memory_joint_policy import actor_features,memory_correction
from head_counterfactual_rollouts import exploration_sigma
from joint_teacher_state_policy import OUTPUT_SCALE
from study_io import BASE,write_json
from search_complete_return import read


def main():
    stage=BASE/'canonical_gate_band_deterministic_complete_credit_20260920'
    history=read(stage/'state_joint/results.json')['candidates']
    results=[]
    for c in history[1:]:
        md=torch.load(stage/(c['context_arm']+'_initial_policy.pt'),weights_only=False)
        coefficients=np.array(c['coefficients']).reshape(-1,3)
        alpha=np.array(md['memory_decays'])[:,None,None]
        rows=[]
        for r in c['records']:
            with np.load(r['trace']) as z:observations=z['observations'].copy()
            memory=np.zeros((2,len(coefficients)//3,3));previous=np.zeros_like(memory[0])
            maximum=np.zeros(3);absolute=np.zeros(3)
            for x in observations:
                f=actor_features(md,x,np.zeros(3));memory=alpha*memory+(1-alpha)*previous;previous=f.copy()
                design=np.concatenate([f,*memory],axis=0)*OUTPUT_SCALE
                sigma=exploration_sigma(md,design)
                correction=memory_correction(f,memory,coefficients)
                ratio=np.divide(abs(correction),sigma,out=np.zeros(3),where=sigma>0.)
                assert not np.any((sigma==0.)&(correction!=0.))
                maximum=np.maximum(maximum,ratio);absolute=np.maximum(absolute,abs(correction))
            rows.append(dict(path=r['path'],max_standardized_change=maximum.tolist(),max_latent_change=absolute.tolist()))
        result=dict(tag=c['tag'],coefficient_norm=float(np.linalg.norm(coefficients)),
            box_bound_coefficients=int(np.count_nonzero(abs(coefficients)>=.05-1e-10)),
            max_visited_state_standardized_change=max(max(r['max_standardized_change']) for r in rows),paths=rows)
        results.append(result)
        print('FEEDBACK_TRUST',result['tag'],result['coefficient_norm'],result['box_bound_coefficients'],result['max_visited_state_standardized_change'],flush=True)
    write_json(stage/'closed_loop_trust_audit.json',dict(candidates=results,new_physical_runs=0,
        exact_outer_actor_features=True,not_a_bound_on_unseen_states=True))


if __name__=='__main__':main()
