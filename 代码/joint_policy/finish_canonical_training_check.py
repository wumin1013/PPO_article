"""Resume the failed empty-corner summary without repeating completed physics."""
import time
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy,environment_factory
from learn_canonical_credit_step import compare
from learn_single_decision_credit import review
from probe_direct_constraint_composition import metric_rows
from train_pilot import evaluate
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    torch.set_num_threads(1)
    original=BASE/'canonical_gate_waveform_learned_update_20260917'
    assert not (original/'receipt.json').exists()
    data=read(original/'results.json');names=data['metric_names'];baseline=data['baseline']
    selection=read(original/'screen_selection.json')
    chosen=next(r for r in data['candidates'] if r['model']==selection['model'])
    model=original/chosen['model'];assert chosen['checks']['eligible']
    ids=[r['original_path'] for r in baseline['results']]
    remaining=[i for i in range(64) if i not in ids]
    saved_parent=read(original/'parent_remaining/summary.json')['results']
    assert len(saved_parent)==len(remaining)==35 and all(r['success'] for r in saved_parent)
    stage=register('canonical_gate_full_training_recovery_20260917',
        [__file__,original/'registration.json',original/'results.json',
         original/'screen_selection.json',original/'parent_remaining/summary.json',model,
         BASE/'learn_exit_local_credit.py',BASE/'learn_canonical_credit_step.py'],
        dict(reused_parent_paths=35,new_candidate_paths=35,new_actor_calls=5000,
             new_physics_ticks=120000,wall_s=120),
        'The original run completed both29-path screens and all35 remaining parent '
        'paths, then its summary called max on an empty isolated-corner subset. '
        'Fix absent shape extrema to null. Reuse every completed trajectory and '
        'the prior training-only candidate selection. Run only the missing35 '
        'candidate routes under the unchanged canonical runtime, then compare '
        'full64 results. This bounded recovery adds no new candidate or fitting '
        'opportunity and accesses no development or sealed data.')
    start=time.perf_counter()
    def enrich(rs,ts,precomputed_metrics=None):
        summary=review(rs,ts)
        metrics=metric_rows(rs,ts,summary,names) if precomputed_metrics is None else precomputed_metrics
        return dict(summary=summary,metrics=metrics,
            results=rs,traces=[str(t.relative_to(BASE)) for t in ts])
    parent_remaining=enrich([dict(r,original_path=i) for r,i in zip(saved_parent,remaining)],
        [original/'parent_remaining'/f'{j}.npz' for j in range(len(remaining))])
    assert parent_remaining['summary']['median_exit_mm'] is None
    assert parent_remaining['summary']['max_entry_mm'] is None
    assert parent_remaining['summary']['shapes']==[]
    p=CanonicalGatePolicy(256)
    p.load_state_dict(torch.load(model,map_location='cpu',weights_only=False)['policy'])
    paths=read(BASE/'learning_preparation_v4/paths.json')['training']
    costs=dict(evaluation_actor_calls=0,evaluation_physics_ticks=0)
    rs=evaluate(p,[np.asarray(paths[i]) for i in remaining],stage,'candidate_remaining',None,
        costs,start+120,environment_factory)
    assert costs['evaluation_actor_calls']<=5000 and costs['evaluation_physics_ticks']<=120000
    candidate_remaining=enrich([dict(r,original_path=i) for r,i in zip(rs,remaining)],
        [stage/'candidate_remaining'/f'{j}.npz' for j in range(len(remaining))])
    def combine(a,b):
        pairs=sorted(zip(a['results']+b['results'],a['traces']+b['traces']),key=lambda r:r[0]['original_path'])
        assert [r['original_path'] for r,t in pairs]==list(range(64))
        metrics=sorted(a['metrics']+b['metrics'],key=lambda r:r['path'])
        return enrich([r for r,t in pairs],[BASE/t for r,t in pairs],metrics)
    full_parent=combine(baseline,parent_remaining);full_candidate=combine(chosen,candidate_remaining)
    full_candidate['checks']=compare(full_candidate,full_parent,names)
    cumulative=dict(data['costs'])
    cumulative['evaluation_actor_calls']+=sum(r['decisions'] for r in saved_parent)+costs['evaluation_actor_calls']
    cumulative['evaluation_physics_ticks']+=sum(round(r['elapsed_s']*1000) for r in saved_parent)+costs['evaluation_physics_ticks']
    assert cumulative['evaluation_actor_calls']<=16000 and cumulative['evaluation_physics_ticks']<=400000
    data.update(chosen=chosen,full_parent=full_parent,full_candidate=full_candidate,
        costs=cumulative,recovery=stage.name,working_model_pointer_changed=False,goal_complete=False)
    write_json(original/'results.json',data)
    write_json(original/'training_selection.json',dict(model=chosen['model'],parent=full_parent,
        training=full_candidate['summary'],results=full_candidate['results'],
        traces=full_candidate['traces'],checks=full_candidate['checks']))
    write_json(stage/'results.json',dict(costs=costs,combined_costs=cumulative,
        full_parent_successes=full_parent['summary']['successes'],
        full_candidate_successes=full_candidate['summary']['successes'],
        checks=full_candidate['checks'],repeated_physical_paths=0,goal_complete=False))
    receipt(stage,dict(status='completed_missing_candidate_training_routes',costs=costs,
        wall_s=time.perf_counter()-start,full_training_eligible=full_candidate['checks']['eligible']))
    receipt(original,dict(status='completed_canonical_learned_update_with_summary_recovery',
        costs=cumulative,screen_eligible=1,full_training_eligible=full_candidate['checks']['eligible'],
        recovery_receipt=str((stage/'receipt.json').relative_to(BASE)),
        recovery_receipt_sha256=sha(stage/'receipt.json'),goal_complete=False))
    print('CANONICAL_FULL_TRAINING',full_candidate['summary']['successes'],full_candidate['checks'],flush=True)


if __name__=='__main__':main()
