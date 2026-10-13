"""Complete the fixed shape candidate's missing routes without accepting it."""
import time
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy,checkpoint_environment_factory
from train_pilot import evaluate
from learn_single_decision_credit import review
from probe_direct_constraint_composition import metric_rows
from learn_canonical_credit_step import compare
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_smooth_shape_learning_20260917'
    data=read(source/'results.json');candidate=next(r for r in data['candidates'] if r['step']==1.)
    model=source/candidate['model'];assert not candidate['checks']['eligible']
    assert candidate['summary']['successes']==29
    parent=BASE/'canonical_gate_band_correct_physical_cruise_full_learning_20260917'
    baseline=read(parent/'results.json')['full_candidate']
    stage=register('canonical_gate_band_smooth_shape_full_diagnostic_20260917',
        [__file__,model,source/'results.json',source/'receipt.json',parent/'results.json',parent/'receipt.json',
         BASE/'learn_canonical_credit_step.py',BASE/'learning_preparation_v4/paths.json'],
        dict(missing_routes=35,new_actor_calls=3000,new_physics_ticks=70000,wall_s=90,new_parameter_updates=0),
        'The fixed full step already failed its original per-corner curvature '
        'and screen-time criteria, and remains rejected. Complete ONLY its35 '
        'missing training routes to resolve the whole64-route cruise/time '
        'tradeoff; the29-route screen excludes pure straights and cannot '
        'establish that result. Reuse all29 candidate and64 parent paths. '
        'This is diagnosis of an already learned model, not an extra scale '
        'search, renewed training, revised adoption threshold or promotion '
        'of the failed candidate. Failure-prefix times remain incomparable.')
    meta=torch.load(model,map_location='cpu',weights_only=False)
    p=CanonicalGatePolicy(256);p.load_state_dict(meta['policy'])
    factory=checkpoint_environment_factory(meta['architecture'])
    done={r['original_path'] for r in candidate['results']};ids=[i for i in range(64) if i not in done]
    paths=read(BASE/'learning_preparation_v4/paths.json')['training'];assert len(ids)==35
    start=time.perf_counter();costs=dict(evaluation_actor_calls=0,evaluation_physics_ticks=0)
    rs=evaluate(p,[np.asarray(paths[i]) for i in ids],stage,'remaining',None,costs,start+90,factory)
    assert costs['evaluation_actor_calls']<=3000 and costs['evaluation_physics_ticks']<=70000
    rs=[dict(r,original_path=i) for r,i in zip(rs,ids)]
    ts=[stage/'remaining'/f'{j}.npz' for j in range(len(ids))]
    extra=review(rs,ts);names=data['metric_names'];metrics=metric_rows(rs,ts,extra,names)
    pairs=sorted(zip(candidate['results']+rs,[BASE/t for t in candidate['traces']]+ts),key=lambda x:x[0]['original_path'])
    full=dict(results=[r for r,t in pairs],traces=[str(t.relative_to(BASE)) for r,t in pairs],
        summary=review([r for r,t in pairs],[t for r,t in pairs]),
        metrics=sorted(candidate['metrics']+metrics,key=lambda r:r['path']))
    full['checks']=compare(full,baseline,names,energy_scope='aggregate',preserve_actual_cruise=True)
    write_json(stage/'results.json',dict(model=str(model.relative_to(BASE)),model_sha256=sha(model),
        full_parent=baseline,full_candidate=full,metric_names=names,costs=costs,
        original_screen_failure_preserved=True,adopted=False,goal_complete=False))
    receipt(stage,dict(status='completed_fixed_shape_candidate_full_training_diagnostic',costs=costs,
        complete_paths=full['summary']['successes'],wall_s=time.perf_counter()-start,adopted=False,goal_complete=False))
    print('FULL_SHAPE_DIAGNOSTIC',{k:full['summary'][k] for k in ('successes','full_time_s','complete_cruise_s','exit_sum_mm','full_reverse_sum_deg')},flush=True)


if __name__=='__main__':main()
