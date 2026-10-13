"""Give the historical fixed model the same observation-support selection opportunity."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import time,json
import numpy as np
import torch
from study_io import BASE,sha,register,receipt,write_json
from joint_runtime_v5 import environment_factory
from smooth_policy_candidate import SmoothExplorationPolicy
from run_natural_support_search import LocalSupportPolicy
from paper_complete_evaluation import evaluate
from natural_case_audit import analytic_command_audit
from natural_case_metrics import metrics


class FixedSupportPolicy(LocalSupportPolicy):
    def bind(self,env):
        return env


def main():
    torch.set_num_threads(1);case=BASE/'paper_evidence_review/natural_closed_case'
    folder=BASE/'fixed3_gsde_refinement_v7';model=folder/'final.pt'
    old=json.loads((folder/'receipt.json').read_text(encoding='utf-8'));assert sha(model)==old['final_sha256']
    median=json.loads((BASE/'paper_evidence_review/nonlocal_state_diagnostic/results.json').read_text(encoding='utf-8'))['training_median']
    specs=[dict(label='remaining_median',upper=median[0],median=None),
           dict(label='remaining_midrange',upper=.20,median=None),
           dict(label='nonlocal_median',upper=median[0],median=median)]
    stage=register('paper_evidence_review/natural_closed_case/fixed_support_calibration',
        [__file__,BASE/'run_natural_support_search.py',model,folder/'receipt.json',case/'fixed_historical/receipt.json',case/'reference.csv'],
        dict(wall_limit_s=1800,max_evaluations=3,specs=specs,new_training_updates=0,fixed_d_mm=3.),
        'Give independently PPO-trained strong fixed3 the exact three support variants '
        'registered for main, with additional best-complete-time selection. Preserve failures '
        'and historic raw baseline. This is public-development calibration, not fresh PPO '
        'training or full-pipeline budget matching. Stop at three variants or wall limit.')
    start=time.perf_counter();records=[]
    parent=torch.load(model,map_location='cpu',weights_only=False)
    reference=np.loadtxt(case/'reference.csv',delimiter=',',skiprows=1)
    geometry=json.loads((case/'geometry.json').read_text(encoding='utf-8'))
    for spec in specs:
        if time.perf_counter()>start+1750:break
        base=SmoothExplorationPolicy(parent['observation_dim'],mode=parent['mode'])
        base.load_state_dict(parent['policy']);base.eval()
        policy=FixedSupportPolicy(base,spec['upper'],spec['median'])
        path=stage/(spec['label']+'.npz')
        result=evaluate(policy,environment_factory(reference,fixed_d=3.),path,start+1750)
        result['spec']=spec
        if result['success']:
            z=np.load(path)
            result['analytic']=analytic_command_audit(reference[0],z['rows'],geometry)
            result['success']=result['analytic']['analytic_corridor_pass']
            if not result['success']:result['complete_time_s']=None
            write_json(path.with_name(path.stem+'_metrics.json'),metrics(reference,z['rows'],geometry))
        records.append(result);write_json(stage/'results.json',records)
        print('FIXED_SUPPORT',spec['label'],result['success'],result['elapsed_prefix_or_complete_s'],flush=True)
    eligible=[r for r in records if r['success']]
    selected=min(eligible,key=lambda r:r['complete_time_s']) if eligible else None
    write_json(stage/'selection.json',selected)
    receipt(stage,dict(status='completed_fixed_support_calibration',wall_s=time.perf_counter()-start,
        variants=len(records),successes=len(eligible),ppo_updates=0,
        actor_calls=sum(r['actor_calls'] for r in records),physics_ticks=sum(r['physics_ticks'] for r in records),
        selected=selected['spec']['label'] if selected else None,results_sha256=sha(stage/'results.json'),goal_complete=False))


if __name__=='__main__':main()
