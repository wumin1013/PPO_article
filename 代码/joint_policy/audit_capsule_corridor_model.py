"""Validate affine corridor enclosures without executing new trajectories."""
import argparse
import time
import numpy as np
from capsule_corridor_model import CapsuleCorridorModel
from audit_function_constraint_interference import command_segment_margin,distances_to_segments
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--waveforms',default='canonical_gate_band_bend_current_waveform_model_20260917')
    parser.add_argument('--reference',default='canonical_gate_band_bend_learning_reference_20260917')
    parser.add_argument('--stage',default='canonical_gate_band_refined_capsule_facet_audit_20260917')
    args=parser.parse_args()
    source=BASE/args.waveforms
    reference=BASE/args.reference
    conflict=BASE/'canonical_gate_band_bend_constraint_conflict_20260917'
    stage=register(args.stage,
        [__file__,BASE/'capsule_corridor_model.py',source/'results.json',source/'receipt.json',
         reference/'results.json',reference/'basis.npz',reference/'receipt.json',
         conflict/'results.json',conflict/'receipt.json',BASE/'audit_function_constraint_interference.py'],
        dict(new_actor_decisions=0,new_physics_ticks=0,wall_s=90,paths=29,existing_directions=4),
        'Replace only the isotropic affine displacement restriction in a future '
        'learning proposal. Bind each nominal1ms command interval to the reference '
        'segment minimizing the maximum of its two endpoint distances. Enforce '
        'both affine endpoints within that same convex segment capsule; convexity '
        'then encloses the entire affine interval, including switches between '
        'capsules. Where one1ms interval crosses a capsule intersection, refine '
        'ONLY its affine geometry into subintervals until the assigned-capsule '
        'nominal bound is no greater than the independently certified nominal '
        'union-distance upper bound. This repairs the initial coarse-capsule '
        'overrestriction; it does not change the runtime sampling. Keep the same HALF nominal certified margin reserved below '
        'the original1.25mm Euclidean limit. No change to runtime acceptance, '
        'gate order, braking, dynamics or any actual trajectory. Verify nominal '
        'admissibility, finite-difference directional derivatives of the fixed '
        'active capsule face (the maximum envelope may be nondifferentiable), supporting '
        'inequalities, an independent continuous-union distance enclosure and '
        'containment of rescaled old-displacement-feasible examples. Reuse four '
        'exact saved policy parameter differences. Validity concerns the affine '
        'XY model; actual full-policy checks remain mandatory.')
    start=time.perf_counter();data=read(source/'results.json');physical=read(reference/'results.json')
    assert data['parent_sha256']==physical['parent_sha256']
    with np.load(reference/'basis.npz') as z:directions=z['basis']*z['amplitudes'][:,None]
    records=[];zero=np.zeros(768)
    for j,record in enumerate(data['records']):
        with np.load(source/record['xy_file']) as z:nominal=z['nominal'].copy();gradient=z['gradient'].copy()
        with np.load(BASE/physical['baseline']['traces'][j]) as z:ref=z['reference'].copy()
        model=CapsuleCorridorModel(ref,nominal,gradient,record['segment_max_distance_mm'])
        margin=.5*(1.25-record['segment_max_distance_mm']);bound=1.25-margin
        value,row,active=model.evaluate(zero)
        assert value<bound,(record['path'],value,bound)
        checks=[]
        for k,direction in enumerate(directions):
            displacement=np.einsum('tcd,d->tc',gradient,direction)
            isotropic=float(np.linalg.norm(displacement,axis=1).max())
            new,_,_=model.evaluate(direction)
            rows=np.zeros((len(nominal),2));rows[:,:2]=nominal+displacement
            union=command_segment_margin(ref,rows)['max_distance_upper_mm']
            assert union<=new+1.1e-6
            assert new>=value+row@direction-1e-10
            safe_scale=min(1.,margin/max(isotropic,1e-30))
            safe,_,_=model.evaluate(safe_scale*direction)
            assert safe<=bound+1e-8
            eps=1e-4
            fd=(model.evaluate(eps*direction)[0]-model.evaluate(-eps*direction)[0])/(2*eps)
            coordinate=active['sample_coordinate'];left=int(np.floor(coordinate))
            alpha=coordinate-left;right=min(left+1,len(nominal))
            point=(1-alpha)*model.nominal[left]+alpha*model.nominal[right]
            delta=np.zeros(2)
            if left:delta+=(1-alpha)*(gradient[left-1]@direction)
            if alpha:delta+=alpha*(gradient[right-1]@direction)
            segment=active['reference_segment']
            face_values=distances_to_segments(ref,np.vstack([point-eps*delta,point+eps*delta]))[:,segment]
            face_fd=(face_values[1]-face_values[0])/(2*eps)
            error=abs(float(face_fd-row@direction))
            assert error<1e-5,(record['path'],k,error)
            checks.append(dict(direction=k,old_displacement_mm=isotropic,
                old_bound_passed=isotropic<=margin,capsule_distance_mm=new,
                capsule_bound_passed=new<=bound,continuous_union_upper_mm=union,
                active_face_directional_derivative_error=error,
                max_envelope_central_difference_discrepancy=abs(float(fd-row@direction)),old_safe_scale=safe_scale,
                old_safe_example_passed=safe<=bound+1e-8))
        records.append(dict(path=record['path'],nominal_capsule_distance_mm=value,
            refined_subintervals=len(model.intervals),refinement_depth=model.refinement_depth,
            reserved_distance_mm=margin,capsule_distance_bound_mm=bound,active_nominal=active,checks=checks))
        assert time.perf_counter()-start<85
    write_json(stage/'results.json',dict(parent_sha256=data['parent_sha256'],records=records,
        affine_full_interval_enclosure_verified=True,actual_new_policy_guaranteed=False,goal_complete=False))
    receipt(stage,dict(status='verified_affine_capsule_corridor_enclosures',paths=len(records),
        new_actor_decisions=0,new_physics_ticks=0,wall_s=time.perf_counter()-start))
    print('CAPSULE_CORRIDOR_AUDIT',len(records),
        'old_rejected_new_allowed',sum(not c['old_bound_passed'] and c['capsule_bound_passed']
            for r in records for c in r['checks']),flush=True)


if __name__=='__main__':main()
