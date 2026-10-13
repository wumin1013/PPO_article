"""Read-only full-return, paired-response and target-event audit of capacity learning."""
import numpy as np
from head_counterfactual_rollouts import cost
from audit_current_target_changes import target_changes
from learn_full_pool_joint_credit import aggregate
from study_io import BASE,write_json,sha
from search_complete_return import read


def main():
    stage=BASE/'canonical_gate_band_action_capacity_joint_training_20260920'
    data=read(stage/'baseline.json');baseline=data['baseline']['records']
    retained={r['path']:r['baseline_shape'] for r in data['retained'] if 'baseline_shape' in r}
    history=read(stage/'state_joint/results.json')['candidates'];zero=history[0]
    before={r['path']:r for r in zero['records']};index={c['tag']:c for c in history}
    outcomes=[];pairs=[]
    for arm in ('small','wide'):
        candidates=[c for c in history if c.get('arm')==arm]
        if not candidates:continue
        physically_safe=[c for c in candidates if c['physical_pass']]
        if not physically_safe:continue
        raw_best=min(physically_safe,key=lambda c:c['training_score'])
        by={r['path']:r for r in raw_best['records']}
        full=[by.get(r['path'],r) for r in baseline]
        changes=[]
        for r in raw_best['records']:
            b=before[r['path']];event=target_changes(r['trace']);old=target_changes(b['trace'])
            changes.append(dict(path=r['path'],cost_change=cost(r,b,retained[r['path']])-cost(b,b,retained[r['path']]),
                target_jump_before_deg=old['max_decision_jump_deg'],target_jump_after_deg=event['max_decision_jump_deg'],
                target_events=event['largest_changes'],
                shape_change={key:r['shape'][key]-b['shape'][key] for key in (
                    'out_edge_outward_mm','full_physical_reverse_deg','full_normalized_energy',
                    'full_spatial_peak_curvature_mm_inv','reverse_lateral_total_mm','reverse_lateral_significant_duration_s')},
                apex_speed_change=r['shape']['apex_speed_mm_s']-b['shape']['apex_speed_mm_s'],
                time_change_s=r['complete_time_s']-b['complete_time_s'],
                cruise_change_s=r['motion']['cruise_ge99_s']-b['motion']['cruise_ge99_s'],
                first_turn_inside=r['target_motion']['first_turn_inside']))
        outcomes.append(dict(arm=arm,policies=len(candidates),physical_passes=len(physically_safe),
            guard_passing=sum(min(c['augmented_measured_inequalities'])>=-1e-8 for c in physically_safe),
            raw_best=raw_best['tag'],raw_best_score=raw_best['training_score'],summary=aggregate(full),changes=changes))
    for pair in range(6):
        names=[f'{arm}_g0_p{pair}_s{sign}' for arm in ('small','wide') for sign in (-1,1)]
        if not all(name in index for name in names):continue
        a,b,c,d=[index[name] for name in names]
        assert np.allclose(np.array(a['coefficients'])*100.,c['coefficients'],rtol=0.,atol=1e-14)
        assert np.allclose(np.array(b['coefficients'])*100.,d['coefficients'],rtol=0.,atol=1e-14)
        small=(b['training_score']-a['training_score'])/2.
        wide=(d['training_score']-c['training_score'])/200.
        pairs.append(dict(pair=pair,small_directional_response=small,wide_response_divided_by100=wide,
            same_sign=bool(small*wide>0.),small_even_residual=(a['training_score']+b['training_score'])/2-zero['training_score'],
            wide_even_residual=(c['training_score']+d['training_score'])/2-zero['training_score']))
    report=dict(parent_score=zero['training_score'],parent_summary=aggregate(baseline),arms=outcomes,
        paired_first_generation=pairs,new_physical_runs=0,
        results_sha256=sha(stage/'state_joint/results.json'),
        scope='Actual full trajectories. Raw best is an unadopted training point, not evidence of acceptable superiority.')
    write_json(stage/'capacity_result_audit.json',report)
    for a in outcomes:
        print('CAPACITY_RESULT',a['arm'],a['policies'],a['physical_passes'],a['guard_passing'],a['raw_best'],
              a['raw_best_score'],a['summary'],flush=True)
        for c in sorted(a['changes'],key=lambda c:c['cost_change'])[:4]:print('CAPACITY_PATH',c,flush=True)
    print('CAPACITY_PAIRED_RESPONSES',pairs,flush=True)


if __name__=='__main__':main()
