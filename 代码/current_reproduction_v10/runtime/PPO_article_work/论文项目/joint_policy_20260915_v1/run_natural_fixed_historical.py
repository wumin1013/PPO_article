"""Reuse the independently PPO-trained historical strong fixed-D checkpoint."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json
import time
import numpy as np
import torch
from study_io import BASE,sha,write_json,register,receipt
from joint_runtime_v5 import environment_factory
from smooth_policy_candidate import SmoothExplorationPolicy
from paper_complete_evaluation import evaluate
from natural_case_audit import analytic_command_audit


def main():
    torch.set_num_threads(1);case=BASE/'paper_evidence_review/natural_closed_case'
    folder=BASE/'fixed3_gsde_refinement_v7';model=folder/'final.pt'
    old=json.loads((folder/'receipt.json').read_text(encoding='utf-8'))
    assert sha(model)==old['final_sha256']
    stage=register('paper_evidence_review/natural_closed_case/fixed_historical',
        [__file__,model,folder/'receipt.json',folder/'selection.json',case/'reference.csv',
         BASE/'joint_runtime_v5.py',BASE/'smooth_policy_candidate.py'],
        dict(wall_limit_s=1800,complete_evaluations=1,new_training_updates=0,fixed_d_mm=3.),
        'Use the historical trained D=3 mm policy selected on its original validation set. '
        'No relabeling as a matched-backbone ablation; runtime/state differ from current '
        'candidate. Keep old training and selection budgets and all new-case failures.')
    started=time.perf_counter();parent=torch.load(model,map_location='cpu',weights_only=False)
    policy=SmoothExplorationPolicy(parent['observation_dim'],mode=parent['mode'])
    policy.load_state_dict(parent['policy']);policy.eval()
    reference=np.loadtxt(case/'reference.csv',delimiter=',',skiprows=1)
    result=evaluate(policy,environment_factory(reference,fixed_d=3.),stage/'trace.npz',started+1770)
    result.update(model_sha256=sha(model),prior_training=old['costs'],prior_refinement_wall_s=old['wall_s'])
    if result['success']:
        z=np.load(stage/'trace.npz')
        result['analytic']=analytic_command_audit(reference[0],z['rows'],json.loads((case/'geometry.json').read_text(encoding='utf-8')))
        if not result['analytic']['analytic_corridor_pass']:
            result['success']=False;result['complete_time_s']=None
    write_json(stage/'result.json',result)
    receipt(stage,dict(status='completed_historical_fixed_evaluation',success=result['success'],
        wall_s=time.perf_counter()-started,actor_calls=result['actor_calls'],physics_ticks=result['physics_ticks'],
        ppo_updates=0,goal_complete=False,result_sha256=sha(stage/'result.json')))
    print('HISTORICAL_FIXED',result['success'],result['reason'],result['elapsed_prefix_or_complete_s'],flush=True)


if __name__=='__main__':main()
