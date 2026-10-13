"""Finite shared-parameter locality contrast at the retained policy center."""
import time
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy,checkpoint_environment_factory
from capsule_corridor_model import CapsuleCorridorModel
from audit_function_constraint_interference import distances_to_segments
from train_pilot import evaluate,save_torch
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    torch.set_num_threads(1)
    reference=BASE/'canonical_gate_band_retained_center_reference_20260918'
    waves=BASE/'canonical_gate_band_retained_center_waveforms_20260918'
    coverage=BASE/'canonical_gate_band_retained_center_distance_error_20260918'
    data=read(reference/'results.json');model=BASE/data['parent_model']
    stage=register('canonical_gate_band_retained_center_locality_20260918',
        [__file__,model,reference/'results.json',reference/'basis.npz',reference/'receipt.json',
         waves/'results.json',waves/'receipt.json',coverage/'results.json',coverage/'receipt.json',
         BASE/'canonical_gate_band_retained_center_events_20260918/results.json',
         BASE/'capsule_corridor_model.py',BASE/'train_pilot.py'],
        dict(paths=[31,41],source_direction=1,scales=[-.5,-.125,.125,.5],
             new_full_paths=8,evaluation_actor_calls=3000,evaluation_physics_ticks=70000,wall_s=90),
        'Fresh4067 credit fails finite coverage, with path31 the sole signed-capsule failure '
        'and path41 the largest XY failure; no brake-initiation changes explain the saved errors. '
        'Test a single EXACT shared-network direction from4067 to the saved joint-regret full '
        'candidate at signed0.125 and0.5 amplitudes, reusing its already measured full amplitude. '
        'Hold state, rewards, all three-head parameter direction, model precision, guide and '
        'physical limits fixed; only finite amplitude changes. Complete every path including '
        'coverage and stop. Compare fixed-capsule signed error, full XY error, and brake initiation '
        'with the same fresh model. Small-locality evidence requires both signs at0.125 complete, '
        'unchanged brake initiation and error inside the unchanged half nominal capsule reserve '
        'on both paths. A positive result is limited directional evidence for a smaller local '
        'model domain, not a new training policy, global radius guarantee or lowered acceptance '
        'criterion. No reward/noise sweep, empirical-error inflation, deployment or sealed access.')
    start=time.perf_counter();costs=dict(evaluation_actor_calls=0,evaluation_physics_ticks=0)
    md=torch.load(model,map_location='cpu',weights_only=False);policy=CanonicalGatePolicy(256)
    with np.load(reference/'basis.npz') as z:direction=(z['basis'][1]*z['amplitudes'][1]).reshape(3,256)
    all_paths=[np.asarray(p) for p in read(BASE/'learning_preparation_v4/paths.json')['training']]
    wave_records=read(waves/'results.json')['records'];models={};baseline_rows={}
    for path in (31,41):
        j=next(j for j,r in enumerate(data['baseline']['results']) if r['original_path']==path)
        with np.load(BASE/data['baseline']['traces'][j]) as z:baseline_rows[path]=z['rows'].copy()
        r=next(r for r in wave_records if r['path']==path)
        with np.load(waves/r['xy_file']) as z:nominal=z['nominal'].copy();gradient=z['gradient'].copy()
        models[path]=(CapsuleCorridorModel(all_paths[path],nominal,gradient,r['segment_max_distance_mm']),
                      .5*(1.25-r['segment_max_distance_mm']))
    records=[]
    for scale in (-.5,-.125,.125,.5):
        policy.load_state_dict(md['policy'])
        with torch.no_grad():policy.residual_coefficients.add_(torch.tensor(scale*direction,dtype=torch.float32))
        tag=f'scale{scale:g}'
        save_torch(stage/f'{tag}.pt',dict(md,policy=policy.state_dict(),diagnostic_only=True,
            source_model_sha256=sha(model),source_direction=1,scale=scale))
        results=evaluate(policy,[all_paths[p] for p in (31,41)],stage,tag,None,costs,start+90,
                         checkpoint_environment_factory(md['architecture']))
        for j,(path,result) in enumerate(zip((31,41),results)):
            rec=dict(path=path,scale=scale,result=result,trace=f'{tag}/{j}.npz')
            if result['success']:
                with np.load(stage/rec['trace']) as z:rows=z['rows'].copy()
                cm,reserve=models[path];nominal=baseline_rows[path];brake=lambda x:int(np.flatnonzero(x[:,12])[0])
                changed=brake(rows)!=brake(nominal)
                if len(rows)<len(nominal):
                    assert np.linalg.norm(rows[-1,2:4])<.001 and np.linalg.norm(rows[-1,4:6])<.01
                    rows=np.vstack([rows,np.repeat(rows[-1:],len(nominal)-len(rows),axis=0)])
                actual=np.vstack([all_paths[path][0],rows[:len(nominal),:2]])
                predicted=cm.nominal.copy();predicted[1:]+=np.einsum('tcd,d->tc',cm.gradient,(scale*direction).ravel())
                def distance(x):
                    points=(1-cm.alpha[:,None])*x[cm.point_index]+cm.alpha[:,None]*x[cm.right_index]
                    return distances_to_segments(all_paths[path],points)[np.arange(len(points)),cm.segment_index]
                error=distance(actual)-distance(predicted);positive=float(max(0.,error.max()))
                rec.update(brake_changed=changed,maximum_xy_error_mm=float(np.linalg.norm(actual-predicted,axis=1).max()),
                    positive_capsule_error_mm=positive,reserve_mm=reserve,error_reserve_ratio=positive/reserve,
                    small_model_domain_pass=bool(not changed and positive<reserve))
            records.append(rec)
            print('CURRENT_RESPONSE_LOCALITY',{k:v for k,v in rec.items() if k!='result'},flush=True)
        assert costs['evaluation_actor_calls']<=3000 and costs['evaluation_physics_ticks']<=70000
        write_json(stage/'results.json',dict(parent_sha256=sha(model),records=records,costs=costs,
            diagnostic_only=True,goal_complete=False))
    passed=all(r['result']['success'] and r.get('small_model_domain_pass',False) for r in records if abs(r['scale'])==.125)
    receipt(stage,dict(status='completed_current_finite_response_locality_probe',
        small_directional_domain_passed=passed,costs=costs,wall_s=time.perf_counter()-start,
        diagnostic_only=True,goal_complete=False))


if __name__=='__main__':main()
