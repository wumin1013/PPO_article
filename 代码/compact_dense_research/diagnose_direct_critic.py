"""Audit saved values/advantages; compare potential scales without any retraining."""
import json
import numpy as np
import torch
from evaluate_historical_dense import ROOT,BASE,sha,write
from compact_control import CompactPolicy
from direct_compact_runtime import DirectCompactRuntime
from learning_update_compact_mc import segmented_advantages

STAGE=ROOT/'direct14_multicase_ppo_seed922302_round1'


def main():
    torch.set_num_threads(1)
    refs={'original':BASE/'paper_evidence_review/natural_closed_case/reference.csv',
          'dense':ROOT/'dense_closed_case/reference.csv'}
    files=[__file__];records=[]
    for case,path in refs.items():
        files.append(path)
        reference=np.loadtxt(path,delimiter=',',skiprows=1)
        env=DirectCompactRuntime(reference);x=torch.tensor(env.observation());values={}
        for label in ['initial','checkpoint_0008','checkpoint_0016']:
            model=STAGE/(label+'.pt');files.append(model)
            policy=CompactPolicy();policy.load_state_dict(torch.load(model,weights_only=False)['policy'])
            with torch.no_grad():values[label]=float(policy.value(x))
        baseline={'original':445.072,'dense':15.254}[case]
        records.append(dict(case=case,initial_potential=env.potential(),
            exact_deterministic_initialization_return=-baseline-env.potential(),critic_start_predictions=values,
            same_complete_path_return_with_potential_speed100=float(env.corridor.arcs[-1]/100.-baseline)))
    batches=[]
    for i in [1,8,16]:
        path=STAGE/f'batch_{i:04d}.npz';files.append(path)
        with np.load(path) as batch:
            model=STAGE/f'checkpoint_{i:04d}.pt';files.append(model)
            policy=CompactPolicy();policy.load_state_dict(torch.load(model,weights_only=False)['policy'])
            with torch.no_grad():
                v=policy.value(torch.tensor(batch['observation'])).numpy()
                nv=policy.value(torch.tensor(batch['next_observation'])).numpy()
            advantage,_=segmented_advantages(batch['reward'],v,nv,batch['terminal'],batch['segment_end'])
            cases={}
            for case in refs:
                mask=batch['case']==case
                cases[case]=dict(refreshed_advantage_mean=float(advantage[mask].mean()),
                    refreshed_advantage_std=float(advantage[mask].std()),max_progress_mm=float(batch['progress_mm'][mask].max()),
                    terminals=int(batch['terminal'][mask].sum()))
            batches.append(dict(update=i,cases=cases))
    result=dict(scope='Read-only diagnostic; deterministic initialization return is not the stochastic on-policy expected value.',
        source_sha256={str(p):sha(p) for p in files},initial_state_values=records,batches=batches,
        potential_scale_note='Counterfactual algebra only: sum[-dt+Phi(next)-Phi(current)] = -T-Phi(start) for successful terminal Phi=0. Changing the scale changes a case-constant offset for complete successful returns. No reward change was used in this round.',
        goal_complete=False)
    write(STAGE/'critic_diagnostic.json',result)
    print(json.dumps(records,ensure_ascii=False))


if __name__=='__main__':main()
