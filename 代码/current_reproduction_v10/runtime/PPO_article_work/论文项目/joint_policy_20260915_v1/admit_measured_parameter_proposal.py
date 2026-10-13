"""Admit bounded learned steps after a successful empirical-guard challenge."""
import argparse
import shutil
import numpy as np
from search_complete_return import read
from study_io import BASE,ROOT,register,receipt,write_json,sha


def main():
    parser=argparse.ArgumentParser()
    for name in ('source','probe','solve','stage'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    source,probe,solve=BASE/args.source,BASE/args.probe,BASE/args.solve
    opt=read(source/'optimizer.json');comparison=read(probe/'comparison.json')
    assert opt['diagnostic_only'] and opt['purpose']=='empirical_margin_challenge'
    assert comparison['passed_composition'] and not comparison['regressions']
    model=BASE/opt['model'];assert sha(model)==opt['parent_sha256']
    assert read(probe/'registration.json')['source_sha256'][model.relative_to(ROOT).as_posix()]==sha(model)
    assert read(solve/'optimizer.json')['parent_sha256']==sha(model)
    assert read(solve/'progress.json')['rounds'][-1]['linear_violation']<=1e-6
    assert opt['projection']['metric_norm']<=.15
    with np.load(source/'direction.npz') as z:
        names=z['metric_names'].tolist();changes=z['predicted_changes'].copy()
    assert np.max(changes[:,:names.index('segment_max_distance_mm')])<=1e-8
    stage=register(args.stage,[__file__,source/'direction.npz',source/'optimizer.json',source/'receipt.json',
        probe/'comparison.json',probe/'results.json',probe/'receipt.json',model,
        solve/'optimizer.json',solve/'progress.json',solve/'receipt.json'],
        dict(new_actor_decisions=0,new_physics_ticks=0,allowed_further_steps=[.5,1.],
             nominal_metric_radius=.15,physical_shape_adoption_reduction=.1),
        'The bounded parameter solve failed its empirical inflated-error model, '
        'but its original nominal waveform/energy and remaining linear constraints '
        'pass. A separately registered both-sign29-route challenge now passes '
        'actual full completion, every original per-corner shape check, aggregate '
        'shape improvement and time allowance. Admit only the two predeclared '
        'larger learned-parameter steps for physical verification. This changes '
        'the model-admission rule, NOT physical limits, shape adoption criteria '
        'or final research acceptance. Do not report the empirical model as '
        'converged or certified, and do not extrapolate small-step success as '
        'larger-step safety. The working model remains unchanged.')
    shutil.copy2(source/'direction.npz',stage/'direction.npz')
    result=dict(opt,diagnostic_only=False,metric_radius=.15,
        purpose='bounded_parameter_proposal_admitted_by_actual_small_composition',
        empirical_inflated_model_feasible=False,nominal_shape_model_feasible=True,
        actual_small_composition_passed=True,source_diagnostic=args.source,
        source_physical_probe=args.probe,source_bounded_solve=args.solve,
        admission_is_final_goal_acceptance=False)
    write_json(stage/'optimizer.json',result)
    receipt(stage,dict(status='admitted_bounded_proposal_by_actual_small_checks',
        new_actor_decisions=0,new_physics_ticks=0,goal_complete=False))


if __name__=='__main__':main()
