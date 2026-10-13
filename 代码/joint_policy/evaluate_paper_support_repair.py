"""First finite algorithm round: training-support bound on remaining-distance input."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import json
import time
import numpy as np
import torch
from deep_joint_credit_policy import load_credit_policy
from learned_action_pulse_policy import environment_factory
from paper_state_support_policy import RemainingSupportPolicy
from paper_complete_evaluation import evaluate
from study_io import BASE,ROOT,register,receipt,sha,write_json


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    torch.set_num_threads(1);review=BASE/'paper_evidence_review'
    frozen=read(review/'frozen_evaluation_candidate.json');model=BASE/frozen['model']
    assert sha(model)==frozen['sha256']
    diagnostic=read(review/'nonlocal_state_diagnostic/results.json')
    upper=diagnostic['training_max'][0]
    names=['mixed_lengths_large','dense_corners_large','closed_line_arc_large']
    refs=[ROOT/f'论文项目/revision_20260913_preview_efficiency_v2/geometry_design_v1/references/{n}.csv' for n in names]
    stage=register('paper_evidence_review/support_repair',
        [__file__,BASE/'paper_state_support_policy.py',BASE/'paper_complete_evaluation.py',
         review/'frozen_evaluation_candidate.json',review/'nonlocal_state_diagnostic/receipt.json',
         review/'nonlocal_state_diagnostic/results.json',review/'candidate_audit.json',
         review/'development/results.json',*refs],
        dict(algorithm_round=1,total_wall_s=1800,maximum_long_case_wall_s=540,
             variants=1,remaining_encoding_upper=upper,training_updates=0,sealed_paths=0),
        'Bound only nonlocal remaining-distance input at the maximum observed over all88 training '
        'trajectories. Learned outputs, weights and physics stay unchanged. Reuse exactly identical '
        'training/development cases by verified decision-input identity; evaluate changed development '
        'cases and all3 public long cases once. Adopt for further evaluation only if no new development '
        'hard failure and at least one long case completes under the original physical contract. '
        'Keep failed or timeout prefixes. No retry or adaptive variant inside this round.')
    started=time.perf_counter();deadline=started+1800
    policy=RemainingSupportPolicy(load_credit_policy(torch.load(model,map_location='cpu',weights_only=False)),upper)
    candidate=next(c for c in read(review/'candidate_audit.json')['candidates'] if c['label']=='worst_guard')
    training_identity=[]
    for row in candidate['records']:
        z=np.load(BASE/row['trace']);x=z['observations']
        identity=all(np.array_equal(policy.transform(a),a) for a in x)
        assert identity
        training_identity.append(dict(path=row['path'],trace_sha256=sha(BASE/row['trace']),
                                      all_recorded_decision_inputs_identical=True,decisions=len(x)))
    development=[]
    for row in read(review/'development/results.json')['records']:
        z=np.load(BASE/row['trace']);identity=all(np.array_equal(policy.transform(a),a) for a in z['observations'])
        if identity:
            result=dict(path=row['path'],success=row['success'],complete_time_s=row['complete_time_s'],
                        reused_by_exact_input_identity=True,trace=row['trace'],trace_sha256=row['trace_sha256'])
        else:
            result=evaluate(policy,environment_factory(z['reference']),stage/f'development_{row["path"]}.npz',deadline)
            result.update(path=row['path'],reused_by_exact_input_identity=False)
        development.append(result)
    long=[]
    for name,source in zip(names,refs):
        if time.perf_counter()>=deadline: break
        result=evaluate(policy,environment_factory(np.loadtxt(source,delimiter=',',skiprows=1)),
                        stage/f'{name}.npz',min(deadline,time.perf_counter()+540))
        result.update(case=name,reference_sha256=sha(source));long.append(result)
        print('SUPPORT_REPAIR',name,result['success'],result['reason'],result['elapsed_prefix_or_complete_s'],flush=True)
        write_json(stage/'progress.json',dict(long_results=long,wall_s=time.perf_counter()-started))
    eligible=all(r['success'] for r in development) and any(r['success'] for r in long)
    results=dict(round=1,training_identity=training_identity,development=development,long=long,
        not_evaluated_long=names[len(long):],eligible_for_further_evaluation=eligible,
        formal_adoption=False,original_model_sha256=sha(model),
        complete_policy_spec=dict(backbone_model=frozen['model'],backbone_sha256=frozen['sha256'],
            observation_transform='remaining_distance_training_support_upper',upper=upper,
            runtime='learned_action_pulse_policy.environment_factory',optional_pulse=0))
    write_json(stage/'results.json',results)
    costs=[r for r in development+long if not r.get('reused_by_exact_input_identity',False)]
    receipt(stage,dict(status='completed_bounded_support_repair',algorithm_round=1,
        eligible_for_further_evaluation=eligible,formal_adoption=False,
        results_sha256=sha(stage/'results.json'),training_paths_reused=88,
        development_paths_reused=sum(r.get('reused_by_exact_input_identity',False) for r in development),
        long_successes=sum(r['success'] for r in long),long_evaluated=len(long),
        actor_calls=sum(r['actor_calls'] for r in costs),physics_ticks=sum(r['physics_ticks'] for r in costs),
        wall_s=time.perf_counter()-started,training_updates=0,sealed_test_opened=False))
    print('SUPPORT_REPAIR_COMPLETE',eligible,'long',sum(r['success'] for r in long),flush=True)


if __name__=='__main__':main()
