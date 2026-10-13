"""Compare stale/current measured response and unadopted complete outcomes."""
import argparse
import numpy as np
from audit_coupled_trust_model import labels
from learn_full_pool_joint_credit import aggregate
from study_io import BASE,write_json,sha
from search_complete_return import read


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',default='canonical_gate_band_information_refreshed_joint_learning_20260920')
    args=parser.parse_args();stage=BASE/args.stage
    with np.load(stage/'current_joint_response.npz') as z:fresh,stale=z['fresh'],z['stale']
    a,b=fresh[:,-1],stale[:,-1]
    cosine=float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30))
    history=read(stage/'state_joint/results.json')['candidates'];updates=read(stage/'learning_updates.json')
    result=read(stage/'shadow_training_result.json');chosen=next(c for c in history if c['tag']==result['tag'])
    values=np.array(chosen['augmented_measured_inequalities']);names=labels(chosen['records'])
    assert len(names)==len(values)
    protected=aggregate(read(stage/'baseline.json')['baseline']['records'])
    source=BASE/'canonical_gate_band_coupled_early_late_joint_completion_20260920'
    old=read(source/'shadow_training_result.json')['summary']
    summary=result['summary']
    metrics=('complete_time_s','curved_cruise99_s','reverse_sum_deg','exit_sum_mm','energy_sum','reverse_lateral_sum_mm')
    report=dict(new_physical_runs=0,model_sha256=sha(stage/'shadow_training_policy.pt'),
        results_sha256=sha(stage/'state_joint/results.json'),current_vs_stale_cost_gradient_cosine=cosine,
        current_vs_stale_cost_gradient_norm_ratio=float(np.linalg.norm(a)/max(np.linalg.norm(b),1e-30)),
        response_directions=k if (k:=len(a)) else 0,actor_parameters=54,
        controls=[u for u in updates if u['arm'] in ('stale','fresh')],
        physically_valid_new_policies=sum(c['physical_pass'] for c in history[1:]),
        new_policies=len(history)-1,final_failed_guard_count=int(np.count_nonzero(values < -1e-8)),
        worst_final_guards=[dict(channel=names[i],margin=float(values[i])) for i in np.argsort(values)[:12]],
        final_summary=summary,changes_from_training_seed={key:summary[key]-old[key] for key in metrics},
        changes_from_protected_research={key:summary[key]-protected[key] for key in metrics},
        adopted=False,scope='All route times/cruise use complete ordered paths, with proven unchanged route reuse. '
        'Shape totals in aggregate describe isolated corners; multicorner31 is separately protected and raw-audited. '
        'Finite directional response agreement is local and does not establish PPO estimator bias or final superiority.')
    write_json(stage/'refreshed_response_outcome_audit.json',report)
    for key in ('current_vs_stale_cost_gradient_cosine','controls','final_failed_guard_count','final_summary',
                'changes_from_training_seed','changes_from_protected_research','worst_final_guards'):
        print('REFRESHED_AUDIT',key,report[key],flush=True)


if __name__=='__main__':main()
