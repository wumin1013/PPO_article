"""Finite observation-support calibration using the public composite development case."""
import os
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = '1'
import time
import json
import numpy as np
import torch
from study_io import BASE, sha, register, receipt, write_json
from paper_state_support_policy import RemainingSupportPolicy
from paper_complete_evaluation import evaluate
from learned_action_pulse_policy import environment_factory
from deep_joint_credit_policy import load_credit_policy


class LocalSupportPolicy(RemainingSupportPolicy):
    def __init__(self, base, upper, median=None):
        super().__init__(base, upper)
        self.median = median

    def transform(self, observation):
        x = super().transform(observation)
        if self.median is not None:
            # Fade to the unmodified terminal state within observed short-route support.
            w = np.clip((float(observation[15]) - .14) / .10, 0., 1.)
            x[[16, 17, 19]] = (1-w)*x[[16, 17, 19]] + w*np.asarray(self.median)[1:]
        return x


def main():
    torch.set_num_threads(1)
    case = BASE/'paper_evidence_review/natural_closed_case'
    model = BASE/'canonical_gate_band_worst_guard_joint_learning_20260920/shadow_training_policy.pt'
    data = json.loads((BASE/'paper_evidence_review/nonlocal_state_diagnostic/results.json').read_text(encoding='utf-8'))
    median = data['training_median']
    specs = [dict(label='remaining_median', upper=median[0], median=None),
             dict(label='remaining_midrange', upper=.20, median=None),
             dict(label='nonlocal_median', upper=median[0], median=median)]
    stage = register('paper_evidence_review/natural_closed_case/support_calibration',
        [__file__, case/'execution_diagnostic/receipt.json', model, case/'reference.csv',
         BASE/'paper_evidence_review/nonlocal_state_diagnostic/results.json'],
        dict(total_wall_s=1800, variants=specs, training_updates=0),
        'Test three globally applied observation-support variants in registered order. '
        'No angle/path-ID features, action table or deployed probe. Stop at first complete '
        'success; otherwise retain every failed prefix and end this repair route after three '
        'variants or the wall limit. Selection on this public development case is disclosed.')
    started=time.perf_counter();records=[]
    reference=np.loadtxt(case/'reference.csv',delimiter=',',skiprows=1)
    parent=torch.load(model,map_location='cpu',weights_only=False)
    for spec in specs:
        if time.perf_counter()>started+1750:break
        policy=LocalSupportPolicy(load_credit_policy(parent),spec['upper'],spec['median'])
        result=evaluate(policy,environment_factory(reference),stage/(spec['label']+'.npz'),started+1770)
        result.update(spec=spec)
        records.append(result);write_json(stage/'results.json',records)
        print('SUPPORT_CALIBRATION',spec['label'],result['success'],result['reason'],result['elapsed_prefix_or_complete_s'],flush=True)
        if result['success']:break
    receipt(stage,dict(status='completed_support_calibration',wall_s=time.perf_counter()-started,
        variants=len(records),successes=sum(x['success'] for x in records),
        actor_calls=sum(x['actor_calls'] for x in records),physics_ticks=sum(x['physics_ticks'] for x in records),
        ppo_updates=0,parameter_calibration=True,goal_complete=False,results_sha256=sha(stage/'results.json')))


if __name__=='__main__':main()
