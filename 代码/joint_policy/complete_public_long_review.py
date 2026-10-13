"""Finish the two missing public development cases without tuning or retry."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import json
import time
import numpy as np
import torch
from deep_joint_credit_policy import load_credit_policy
from learned_action_pulse_policy import environment_factory
from paper_complete_evaluation import evaluate
from study_io import BASE,ROOT,register,receipt,sha,write_json


def main():
    torch.set_num_threads(1)
    review=BASE/'paper_evidence_review'
    frozen=json.loads((review/'frozen_evaluation_candidate.json').read_text(encoding='utf-8'))
    model=BASE/frozen['model'];assert sha(model)==frozen['sha256']
    names=['dense_corners_large','closed_line_arc_large']
    refs=[ROOT/f'论文项目/revision_20260913_preview_efficiency_v2/geometry_design_v1/references/{name}.csv' for name in names]
    stage=register('paper_evidence_review/remaining_public_long',
        [__file__,BASE/'paper_complete_evaluation.py',review/'followthrough/receipt.json',
         review/'frozen_evaluation_candidate.json',*refs],
        dict(cases=names,wall_s=600,models=1,training_updates=0,algorithm_rounds=0,sealed_paths=0),
        'Complete the remaining two pre-existing public long development cases once. '
        'No new weights, optimization, retries, or sealed tests. Keep failures and unexecuted cases explicit.')
    started=time.perf_counter();deadline=started+600
    policy=load_credit_policy(torch.load(model,map_location='cpu',weights_only=False))
    results=[]
    for name,ref in zip(names,refs):
        if time.perf_counter()>=deadline: break
        result=evaluate(policy,environment_factory(np.loadtxt(ref,delimiter=',',skiprows=1)),stage/f'{name}.npz',deadline)
        result.update(case=name,reference_sha256=sha(ref));results.append(result)
        print('PUBLIC_CASE',name,result['success'],result['reason'],result['elapsed_prefix_or_complete_s'],flush=True)
    write_json(stage/'results.json',dict(results=results,not_evaluated=names[len(results):],model_sha256=sha(model)))
    assert sha(model)==frozen['sha256']
    receipt(stage,dict(status='completed_remaining_public_long' if len(results)==2 else 'stopped_bounded_public_long',
        results_sha256=sha(stage/'results.json'),successes=sum(r['success'] for r in results),
        evaluated=len(results),actor_calls=sum(r['actor_calls'] for r in results),
        physics_ticks=sum(r['physics_ticks'] for r in results),wall_s=time.perf_counter()-started,
        algorithm_rounds=0,training_updates=0,sealed_test_opened=False))


if __name__=='__main__':main()
