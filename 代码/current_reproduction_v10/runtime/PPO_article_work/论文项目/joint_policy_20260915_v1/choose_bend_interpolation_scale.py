"""One evidence-based scalar interpolation, without another parameter direction."""
import time
import numpy as np
from solve_continuous_sensor_credit import IMPROVEMENT_NAMES
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    source=BASE/'canonical_gate_band_capsule_measured_margin_solve_20260917'
    probe=BASE/'canonical_gate_band_capsule_measured_margin_composition_20260917'
    learned=BASE/'canonical_gate_band_capsule_measured_energy_learning_20260917'
    stage=register('canonical_gate_band_bend_interpolation_selection_20260917',
        [__file__,source/'direction.npz',source/'optimizer.json',probe/'results.json',
         probe/'receipt.json',learned/'results.json',learned/'receipt.json'],
        dict(new_actor_decisions=0,new_physics_ticks=0,wall_s=30,maximum_new_scalar_checks=1),
        'The one permitted parameter-direction correction is finished. Both '
        'larger steps completed29/29: half-step preserves all metrics but misses '
        'the required gain, full-step achieves3.87% energy gain with just one '
        'path14 exit regression of0.00005765mm. Before changing mechanisms, '
        'use the already measured-.125,+.125,.5,1 responses to test whether '
        'ONE intermediate scalar step is justified. Fit each complete metric '
        'change as a*scale+b*scale^2 with exact zero intercept. Add its maximum '
        'absolute fit residual across ALL four measured scales as empirical '
        'upper allowance. Require every predicted corner quantity nonworse, '
        'five shape totals down>=10%, both total energies down>=2.5%, corridor '
        '<=1.25mm and full time<=102%. Search only the finite0.025 grid strictly '
        'between.5 and1, then select the middle of the feasible grid. This '
        'fit is not a new-direction bound or physical success proof. If no '
        'grid point qualifies, do not execute another rollout. If one does, '
        'register exactly one actual learned scalar check plus missing35 '
        'routes only after full29-route acceptance. No new direction, extra '
        'error fit, physical-limit change, development or sealed access.')
    start=time.perf_counter();p=read(probe/'results.json');l=read(learned/'results.json')
    names=l['metric_names'];base=np.array([r['values'] for r in l['baseline']['metrics']])
    candidates=p['candidates']+l['candidates'];scales=np.array([c['step'] for c in candidates])
    assert scales.tolist()==[-.125,.125,.5,1.] and all(c['summary']['successes']==29 for c in candidates)
    changes=np.array([[r['values'] for r in c['metrics']] for c in candidates])-base
    design=np.c_[scales,scales**2]
    coefficients=np.linalg.lstsq(design,changes.reshape(4,-1),rcond=None)[0].reshape(2,*base.shape)
    fitted=np.einsum('sj,jpm->spm',design,coefficients)
    allowance=np.max(abs(changes-fitted),axis=0)
    records=[];eligible=[];shape_end=names.index('segment_max_distance_mm')
    for scale in np.arange(21,40)*.025:
        predicted=base+scale*coefficients[0]+scale**2*coefficients[1]+allowance
        nonworse=bool(np.all(predicted[:,:shape_end]<=base[:,:shape_end]+1e-6))
        shape_checks=[]
        for name in IMPROVEMENT_NAMES:
            m=names.index(name);old=base[:,m].sum();new=predicted[:,m].sum()
            shape_checks.append(bool(new<=.9*old if old>1e-8 else new<=1e-8))
        energy_checks=[]
        for name in ('full_xy_normalized_energy','full_physical_normalized_energy'):
            m=names.index(name);energy_checks.append(bool(predicted[:,m].sum()<=.975*base[:,m].sum()))
        corridor=bool(np.all(predicted[:,shape_end]<=1.25))
        time_ok=bool(predicted[:,-1].sum()<=1.02*base[:,-1].sum())
        passed=nonworse and all(shape_checks+energy_checks) and corridor and time_ok
        records.append(dict(scale=float(scale),passed=passed,per_corner_nonworse=nonworse,
            aggregate_shape_passes=shape_checks,energy_passes=energy_checks,corridor_passes=corridor,time_passes=time_ok))
        if passed:eligible.append(float(scale))
    chosen=eligible[(len(eligible)-1)//2] if eligible else None
    np.savez_compressed(stage/'scalar_response_model.npz',scales=scales,actual_changes=changes,
        quadratic_coefficients=coefficients,absolute_residual_allowance=allowance,metric_names=np.array(names))
    write_json(stage/'selection.json',dict(parent_sha256=read(source/'optimizer.json')['parent_sha256'],
        direction_sha256=sha(source/'direction.npz'),source_proposal=source.name,eligible_grid=eligible,
        selected_steps=[chosen] if chosen is not None else [],records=records,
        required_actual_energy_reduction=.025,actual_energy_scope='aggregate',
        guaranteed_physical_result=False,maximum_new_scalar_checks=1,goal_complete=False))
    receipt(stage,dict(status='completed_saved_bend_scale_interpolation',selected_scale=chosen,
        new_actor_decisions=0,new_physics_ticks=0,wall_s=time.perf_counter()-start))
    print('BEND_INTERPOLATION_GRID',eligible,'chosen',chosen,flush=True)


if __name__=='__main__':main()
