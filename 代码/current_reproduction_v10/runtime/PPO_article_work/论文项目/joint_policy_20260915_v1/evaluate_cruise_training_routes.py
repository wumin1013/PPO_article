"""Complete straight training-route diagnostics; never override failed shape checks."""
import argparse
import time
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy,checkpoint_environment_factory
from learn_single_decision_credit import review
from search_complete_return import read
from train_pilot import evaluate
from study_io import BASE,register,receipt,write_json,sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--parent',default='canonical_gate_band_bend_interpolated_learning_20260917')
    parser.add_argument('--stage',required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    model=BASE/args.model;parent=BASE/args.parent;data=read(parent/'results.json')
    parent_model=parent/data['chosen']['model'];saved=torch.load(model,map_location='cpu',weights_only=False)
    assert saved['parent_sha256']==sha(parent_model)
    paths=[np.asarray(r) for r in read(BASE/'learning_preparation_v4/paths.json')['training']]
    ids=[i for i,r in enumerate(paths) if len(r)==2]
    full=data['full_candidate'];assert full['summary']['successes']==64 and len(ids)==16
    indices=[next(j for j,r in enumerate(full['results']) if r['original_path']==i) for i in ids]
    baseline=review([full['results'][j] for j in indices],[BASE/full['traces'][j] for j in indices])
    stage=register(args.stage,[__file__,model,parent_model,parent/'results.json',parent/'receipt.json',
        BASE/'learning_preparation_v4/paths.json',BASE/'canonical_gate_state.py',BASE/'train_pilot.py',
        BASE/'learn_single_decision_credit.py',BASE/'complete_motion_metrics.py'],
        dict(paths=ids,evaluation_actor_calls=4000,evaluation_physics_ticks=100000,wall_s=120),
        'The existing29-route corner/mixed screen has no pure straight routes. '
        'Evaluate this single saved learned model on ALL16 already prepared straight TRAINING routes '
        'to measure the actual effect of its cruise request objective, including complete braking '
        'and rest. Reuse the exact parent64-route traces. This is a diagnostic supplement: '
        'it cannot reverse a failed shape screen or constitute full64 acceptance. No new model '
        'selection, physical interventions, development or sealed tests.')
    start=time.perf_counter();costs=dict(evaluation_actor_calls=0,evaluation_physics_ticks=0)
    p=CanonicalGatePolicy(256);p.load_state_dict(saved['policy'])
    rs=evaluate(p,[paths[i] for i in ids],stage,'straight',None,costs,start+120,
        checkpoint_environment_factory(saved['architecture']))
    rs=[dict(r,original_path=i) for i,r in zip(ids,rs)]
    ts=[stage/'straight'/f'{j}.npz' for j in range(len(ids))]
    actual=review(rs,ts)
    assert costs['evaluation_actor_calls']<=4000 and costs['evaluation_physics_ticks']<=100000
    comparable=actual['successes']==len(ids)
    result=dict(parent_sha256=sha(parent_model),model_sha256=sha(model),baseline=baseline,
        actual=actual,results=rs,traces=[str(t.relative_to(BASE)) for t in ts],
        complete_time_gain_s=baseline['full_time_s']-actual['full_time_s'] if comparable else None,
        cruise99_gain_s=actual['complete_cruise_s']['99']-baseline['complete_cruise_s']['99'] if comparable else None,
        diagnostic_only=True,failed_shape_adoption_unchanged=True,goal_complete=False)
    write_json(stage/'results.json',result)
    receipt(stage,dict(status='completed_straight_training_cruise_diagnostic',costs=costs,
        successes=actual['successes'],wall_s=time.perf_counter()-start,goal_complete=False))
    print('STRAIGHT_CRUISE',actual['successes'],'time_gain',result['complete_time_gain_s'],
        'cruise99_gain',result['cruise99_gain_s'],flush=True)


if __name__=='__main__':main()
