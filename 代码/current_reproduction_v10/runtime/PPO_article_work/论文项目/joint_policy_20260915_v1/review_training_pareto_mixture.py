"""Review a training-only Pareto improvement omitted by lexicographic top2."""
import time
import numpy as np
import torch
from learned_head_mixture_policy import LearnedHeadMixturePolicy
from probe_constrained_parameter_step import summarize
from train_pilot import evaluate,save_torch
from joint_runtime_v5 import environment_factory
from search_complete_return import read
from study_io import BASE,register,receipt,sha,write_json


def metrics(record):
    rows=record['paths'];corners=[r['shape'] for r in rows if r['shape']]
    return np.array([-record['rank'][1],-record['rank'][2],sum(r['elapsed_s'] for r in rows),sum(1.25*s['spatial']['normalized_bending_mm_inv'] for s in corners)])


def main():
    torch.set_num_threads(1)
    source=BASE/'learned_head_factorial_probe_20260916'
    a_path=BASE/'constrained_parameter_step_probe_20260916/candidate_1.pt';b_path=BASE/'heading_frame_feedback_learning_20260916/epoch_2000.pt'
    stage=register('learned_mixture_pareto_review_20260916',[__file__,BASE/'learned_head_mixture_policy.py',BASE/'frame_feedback_policy.py',
        source/'training_results.json',source/'receipt.json',a_path,b_path],dict(training_decisions=0,evaluation_decisions=1500,physics_ticks=40000,wall_s=120),
        'From saved factorial training data select a new complete-success candidate that Pareto-dominates the broad-turn endpoint separately in incoming violation, excess reversal, full time and normalized bending energy. Earlier top2 used lexicographic ranking and omitted this trade-space point. Selection is written before the one new development run. No parameter training or sealed test, and no claim that a combined score alone establishes shape.')
    candidates=read(source/'training_results.json')['candidates'];base=next(r for r in candidates if r['coefficients']==[0,0,0]);base_values=metrics(base)
    selected=[r for r in candidates if not r['reused'] and r['successes']==16 and np.all(metrics(r)<=base_values+1e-10) and np.any(metrics(r)<base_values-1e-4)]
    assert len(selected)==1,[(r['coefficients'],metrics(r)) for r in selected]
    row=selected[0];write_json(stage/'selection.json',dict(coefficients=row['coefficients'],training_only=True,metric_names=['incoming_violation','reverse_violation','full_time_s','normalized_energy'],baseline=base_values.tolist(),selected=metrics(row).tolist()))
    a=torch.load(a_path,map_location='cpu',weights_only=False);b=torch.load(b_path,map_location='cpu',weights_only=False)
    p=LearnedHeadMixturePolicy(a['policy'],b['policy'],row['coefficients']);model=stage/'candidate.pt'
    save_torch(model,dict(policy=p.state_dict(),observation_dim=72,mode='joint',architecture='learned_head_mixture',learning_method='training_Pareto_selection_of_learned_feedback_heads',
        coefficients=row['coefficients'],parent_sha256=[sha(a_path),sha(b_path)]))
    validation=[np.asarray(r) for r in read(BASE/'learning_preparation_v4/paths.json')['validation']]
    costs=dict(training_actor_calls=0,training_physics_ticks=0,evaluation_actor_calls=0,evaluation_physics_ticks=0);started=time.perf_counter()
    results=evaluate(p,validation,stage,'validation',None,costs,started+120,environment_factory)
    summary=summarize(stage,'validation',results,list(range(12)));summary.update(coefficients=row['coefficients'],model_sha256=sha(model),final_acceptance=False,goal_complete=False)
    write_json(stage/'results.json',dict(candidate=summary,costs=costs))
    assert costs['evaluation_actor_calls']<=1500 and costs['evaluation_physics_ticks']<=40000
    receipt(stage,dict(status='completed_training_Pareto_candidate_review',costs=costs,wall_s=time.perf_counter()-started))
    print('PARETO CANDIDATE',summary['successes'],summary['full_time_s'])
    for r in summary['paths']:
        s=r['shape']
        print(r['path'],'success',r['success'],'peak',round(r['peak_speed'],2),'cruise',r['cruise_ge95_s'],
            'shape',None if not s else (s['first_physical_turn'],round(s['outward_excursion_mm'],3),round(s['reverse_turning_total_deg'],2),round(s['spatial']['bending_energy_mm_inv'],2),round(s['spatial']['middle80_turn_arc_length_mm'],2)),flush=True)


if __name__=='__main__':main()
