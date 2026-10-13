"""Read saved evidence only; never infer full times from failed prefixes."""
import json
from pathlib import Path
import numpy as np
from evaluate_historical_dense import ROOT,BASE,HIST,sha,write

STAGE=ROOT/'direct14_multicase_ppo_seed922302_round1'
METRICS={
    'S2':BASE/'paper_evidence_review/natural_closed_case/s2_paper_metrics.json',
    'strong_historical_fixed':ROOT.parent/'all34_dominance_research/fixed_actual_failure_recovery_full/full_metrics.json',
    'retained_historical_main':HIST/'full_metrics.json',
    'direct14_initial':ROOT/'direct14_saved_motion_fit_seed922302_original_full/full_metrics.json',
}


def compare(m,b):
    assert m['all34_windows_complete'] and b['all34_windows_complete']
    a={r['id']:r for r in m['corners']};old={r['id']:r for r in b['corners']}
    assert set(a)==set(old) and len(a)==34
    delta=[dict(id=k,delta_deficit_mm=a[k]['speed_deficit_mm']-old[k]['speed_deficit_mm']) for k in a]
    seconds=m['complete_recorded_time_s']-b['complete_recorded_time_s']
    length=m['travel_distance_mm']-b['travel_distance_mm']
    corner=m['corner_total_speed_deficit_mm']-b['corner_total_speed_deficit_mm']
    return dict(new_minus_reference_time_s=seconds,path_length_contribution_s=length/100.,
        speed_deficit_contribution_s=seconds-length/100.,corner_deficit_contribution_s=corner/100.,
        outside_corner_deficit_contribution_s=seconds-(length+corner)/100.,
        corner_not_worse_count=sum(x['delta_deficit_mm']<=1e-7 for x in delta),
        corner_deficit_differences=delta,maximum_deficit_increase_mm=max(x['delta_deficit_mm'] for x in delta),
        cruise99_change_s=m['cruise']['99']['time_s']-b['cruise']['99']['time_s'])


def main():
    sources=dict(METRICS)
    for p in STAGE.glob('eval_*_original_metrics.json'):sources[p.stem]=p
    loaded={k:json.loads(p.read_text(encoding='utf-8')) for k,p in sources.items()}
    rows=[]
    for name,m in loaded.items():
        if name in ['S2','strong_historical_fixed','retained_historical_main']:continue
        rows.append(dict(name=name,complete_time_s=m['complete_recorded_time_s'],
            cruise99=m['cruise']['99'],comparisons={k:compare(m,loaded[k]) for k in METRICS if k!=name}))
    result=dict(goal_complete=False,source_sha256={str(p):sha(p) for p in sources.values()},
        original_comparisons=rows,time_decomposition='T=L_executed/100 + speed_deficit/100; fixed34corner windows and complementary samples.',
        final_acceptance_unchanged='Original34corner dominance plus paired-resolution gain screen remain required; round candidate screen is not final method acceptance.')
    if (STAGE/'updates.json').exists():
        updates=json.loads((STAGE/'updates.json').read_text(encoding='utf-8'))
        batches=[]
        for path in sorted(STAGE.glob('batch_*.npz')):
            with np.load(path) as b:
                by_case={}
                for case in ['original','dense']:
                    mask=b['case']==case
                    by_case[case]=dict(decisions=int(mask.sum()),progress_min_mm=float(b['progress_mm'][mask].min()),
                        progress_max_mm=float(b['progress_mm'][mask].max()),terminals=int(b['terminal'][mask].sum()),
                        mean_reward=float(b['reward'][mask].mean()),reward_std=float(b['reward'][mask].std()))
                batches.append(dict(batch=path.name,cases=by_case))
        result['training']=dict(updates=len(updates),new_decisions=updates[-1]['steps'],
            max_post_update_kl=max(max(x['post_update_exact_kl'],x['post_update_sampled_kl']) for x in updates),
            actor_minibatches=sum(x['actor_minibatches'] for x in updates),
            case_sampling=batches,final_actor_learning_rates=updates[-1]['actor_learning_rates'])
    if (STAGE/'result.json').exists():result['round_result']=json.loads((STAGE/'result.json').read_text(encoding='utf-8'))
    write(STAGE/'evidence_summary.json',result)
    for row in rows:
        compact={k:v for k,v in row.items() if k!='comparisons'}
        compact['comparisons']={k:{a:b for a,b in v.items() if a!='corner_deficit_differences'} for k,v in row['comparisons'].items()}
        print(json.dumps(compact,ensure_ascii=False))
    if 'training' in result:print(json.dumps({k:v for k,v in result['training'].items() if k!='case_sampling'},ensure_ascii=False))


if __name__=='__main__':main()
