"""Signed corridor-distance errors of complete saved affine predictions."""
import time
import argparse
import numpy as np
from capsule_corridor_model import CapsuleCorridorModel
from audit_function_constraint_interference import distances_to_segments
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def main():
    parser=argparse.ArgumentParser()
    for name,suffix in [('waveforms','waveforms'),('reference','reference'),('coverage','coverage'),('events','events'),('capsules','capsules')]:
        parser.add_argument('--'+name,default='canonical_gate_band_refined_energy_current_'+suffix+'_20260917')
    parser.add_argument('--stage',default='canonical_gate_band_current_capsule_error_audit_20260917')
    args=parser.parse_args()
    waves=BASE/args.waveforms;reference=BASE/args.reference;coverage=BASE/args.coverage
    events=BASE/args.events;capsules=BASE/args.capsules
    stage=register(args.stage,
        [__file__,BASE/'capsule_corridor_model.py',waves/'results.json',waves/'receipt.json',
         reference/'results.json',reference/'basis.npz',reference/'receipt.json',
         coverage/'results.json',coverage/'receipt.json',events/'results.json',events/'receipt.json',
         capsules/'results.json',capsules/'receipt.json'],
        dict(new_actor_decisions=0,new_physics_ticks=0,wall_s=60,paths=29,existing_variants=4),
        'Preserve the separate current unconditional XY-norm audit and its actual result. '
        'The learning proposal constrains actual Euclidean '
        'distance to fixed segment capsules, not XY displacement. Audit the '
        'relevant signed distance UNDERprediction at every endpoint of every '
        'refined command subinterval, using exactly that bound capsule '
        'assignment and the SAME global half-margin reserve. If each actual '
        'endpoint distance <= its predicted distance + reserve, then any '
        'candidate whose predicted endpoints satisfy the reserved capsule '
        'bound has actual endpoints within1.25mm. Convexity of each capsule '
        'extends endpoint inclusion across the entire actual affine1ms '
        'subinterval. Measured distance overprediction consumes no safety '
        'reserve. Check this on all116 existing complete trajectories and '
        'preserve separate vector errors and signed errors. This is empirical '
        'coverage, not a guarantee on an unseen direction. Keep the original '
        'terminal-branch exclusion for any observed brake switch; no runtime '
        'or physical-limit modification and no relabeling of failed XY predictions.')
    start=time.perf_counter();data=read(waves/'results.json');physical=read(reference/'results.json')
    event_data=read(events/'results.json');old=read(coverage/'results.json')
    assert data['parent_sha256']==physical['parent_sha256']==event_data['parent_sha256']
    with np.load(reference/'basis.npz') as z:directions=z['basis']*z['amplitudes'][:,None]
    records=[]
    for j,record in enumerate(data['records']):
        path=record['path'];assert physical['baseline']['results'][j]['original_path']==path
        with np.load(waves/record['xy_file']) as z:nominal=z['nominal'].copy();gradient=z['gradient'].copy()
        with np.load(BASE/physical['baseline']['traces'][j]) as z:ref=z['reference'].copy()
        model=CapsuleCorridorModel(ref,nominal,gradient,record['segment_max_distance_mm'])
        def sample(x):return (1.-model.alpha[:,None])*x[model.point_index]+model.alpha[:,None]*x[model.right_index]
        def distance(x):
            points=sample(x)
            return distances_to_segments(ref,points)[np.arange(len(points)),model.segment_index]
        reserve=.5*(1.25-record['segment_max_distance_mm']);checks=[]
        for c in physical['candidates']:
            with np.load(BASE/c['traces'][j]) as z:rows=z['rows'].copy()
            if len(rows)<len(nominal):
                assert np.linalg.norm(rows[-1,2:4])<.001
                rows=np.vstack([rows,np.repeat(rows[-1:],len(nominal)-len(rows),axis=0)])
            actual=np.vstack([ref[0],rows[:len(nominal),:2]])
            direction=c['sign']*directions[c['direction']]
            predicted=np.vstack([ref[0],nominal+np.einsum('tcd,d->tc',gradient,direction)])
            actual_distance=distance(actual);predicted_distance=distance(predicted)
            error=actual_distance-predicted_distance;k=int(error.argmax())
            event=next(e for r in event_data['records'] if r['path']==path for e in r['checks']
                       if e['direction']==c['direction'] and e['sign']==c['sign'])
            distance_pass=bool(max(0.,error.max())<reserve)
            branch_changed=event['initiation_changed'] or event.get('duration_changed',False)
            eligible=(event['event_constraint_rejects'] if branch_changed else distance_pass)
            checks.append(dict(direction=c['direction'],sign=c['sign'],
                maximum_signed_distance_error_mm=float(error.max()),positive_error_mm=float(max(0.,error.max())),
                reserved_distance_mm=reserve,underprediction_within_reserve=distance_pass,
                worst_sample_coordinate=float(model.sample_coordinate[k]),
                maximum_actual_capsule_distance_mm=float(actual_distance.max()),
                maximum_predicted_capsule_distance_mm=float(predicted_distance.max()),
                brake_initiation_changed=event['initiation_changed'],
                brake_duration_changed=event.get('duration_changed',False),conditional_check_passed=bool(eligible)))
        records.append(dict(path=path,checks=checks))
        assert time.perf_counter()-start<55
    passed=all(c['conditional_check_passed'] for r in records for c in r['checks'])
    write_json(stage/'results.json',dict(parent_sha256=data['parent_sha256'],records=records,
        conditional_distance_coverage_passed=passed,old_vector_coverage_passed=old['measured_coverage_passed'],
        new_direction_guaranteed=False,requires_same_capsule_half_margin_constraint=True,goal_complete=False))
    receipt(stage,dict(status='completed_signed_capsule_prediction_error_audit',
        conditional_distance_coverage_passed=passed,new_actor_decisions=0,new_physics_ticks=0,
        wall_s=time.perf_counter()-start,goal_complete=False))
    print('CAPSULE_DISTANCE_ERROR_COVERAGE',passed,'worst_fraction',
        max(c['positive_error_mm']/c['reserved_distance_mm'] for r in records for c in r['checks']),flush=True)


if __name__=='__main__':main()
