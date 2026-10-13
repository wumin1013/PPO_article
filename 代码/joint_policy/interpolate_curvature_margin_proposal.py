"""A feasible line interpolation within the declared empirical metric radius."""
import argparse
import numpy as np
from search_complete_return import read
from study_io import BASE, register, receipt, write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--lower',default='empirical_curvature_margin_zero_gain_feasibility_20260917')
    parser.add_argument('--upper',default='empirical_curvature_margin_solve_20260917')
    parser.add_argument('--stage',default='empirical_curvature_margin_feasible_proposal_20260917')
    args=parser.parse_args()
    low=BASE/args.lower;high=BASE/args.upper
    prepared=BASE/'direct_constraint_parameter_preparation_20260917'
    stage=register(args.stage,
        [__file__,low/'direction.npz',low/'optimizer.json',low/'receipt.json',
         high/'direction.npz',high/'optimizer.json',high/'receipt.json',prepared/'support.npz'],
        dict(new_physics_ticks=0,new_actor_decisions=0,wall_s=10,maximum_metric_step_norm=1.),
        'The20% improvement proposal exceeds the empirical-margin radius; '
        'the same constraints at zero required shape gain are within that radius. Both are '
        'feasible for every shared linear constraint at their respective gain targets. '
        'Choose the largest convex interpolation coefficient whose actual metric norm '
        'is1. Linear feasibility and interpolated gain follow by convexity. This is '
        'optimal only on this explicit line, not a global conic optimum or physical '
        'certificate. No further physical work if the predicted gain is below10%. '
        'Otherwise require small actual full-policy checks before candidate learning.')
    lower,upper=read(low/'optimizer.json'),read(high/'optimizer.json')
    if not lower['empirical_margin_radius_pass']:
        receipt(stage,dict(status='no_radius_feasible_lower_endpoint',new_physics_ticks=0,
            new_actor_decisions=0,lower_metric_norm=lower['projection']['metric_norm'],goal_complete=False))
        print('NO_INTERPOLATION: even the lower endpoint exceeds the declared radius',flush=True)
        return
    assert not upper['empirical_margin_radius_pass']
    assert lower['parent_sha256']==upper['parent_sha256']
    assert lower.get('each_corner_exit_and_reverse_nonincrease')==upper.get('each_corner_exit_and_reverse_nonincrease')
    with np.load(low/'direction.npz') as z:a={k:z[k].copy() for k in z.files}
    with np.load(high/'direction.npz') as z:b={k:z[k].copy() for k in z.files}
    with np.load(prepared/'support.npz') as z:metric=z['metric'].copy()
    start=a['reduced_direction'];delta=b['reduced_direction']-start
    aa=delta@metric@delta;bb=2*start@metric@delta;cc=start@metric@start-1
    coefficient=float((-bb+np.sqrt(bb**2-4*aa*cc))/(2*aa))
    assert 0<coefficient<1
    out=dict(a)
    for key in ('reduced_direction','direction','predicted_changes','variant_predicted_changes'):
        out[key]=a[key]+coefficient*(b[key]-a[key])
    norm=float(np.sqrt(out['reduced_direction']@metric@out['reduced_direction']))
    eta=(1-coefficient)*lower['predicted_joint_fraction']+coefficient*upper['predicted_joint_fraction']
    assert norm<=1+1e-10
    np.savez_compressed(stage/'direction.npz',**out)
    result=dict(upper,projection=dict(metric_norm=norm),predicted_joint_fraction=eta,
        interpolation_coefficient=coefficient,empirical_margin_radius_pass=True,
        interpolation_preserves_linear_feasibility=True,
        suitable_for_small_physical_probe=eta>=.1,goal_complete=False)
    write_json(stage/'optimizer.json',result)
    receipt(stage,dict(status='completed_feasible_empirical_margin_interpolation',
        new_physics_ticks=0,new_actor_decisions=0,predicted_joint_fraction=eta))
    print('FEASIBLE_MARGIN_PROPOSAL',eta,norm,coefficient,flush=True)


if __name__=='__main__':main()
