"""Check complete XY response coverage using existing policy rollouts only."""
import argparse
import time
import numpy as np
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--waveforms',default='canonical_gate_band_transition_waveform_model_20260917')
    parser.add_argument('--reference',default='canonical_gate_band_coherent_parameter_credit_20260917')
    parser.add_argument('--stage',default='canonical_gate_band_transition_response_coverage_20260917')
    args=parser.parse_args()
    source=BASE/args.waveforms
    probes=BASE/args.reference
    data=read(source/'results.json');physical=read(probes/'results.json')
    assert read(source/'receipt.json')['status']=='compiled_complete_transition_waveform_model'
    stage=register(args.stage,
        [__file__,source/'results.json',source/'receipt.json',probes/'results.json',
         probes/'basis.npz',probes/'receipt.json'],
        dict(new_actor_decisions=0,new_physics_ticks=0,wall_s=60,paths=len(data['records']),variants=len(physical['candidates'])),
        'Audit full XY prediction on ALL29 screen paths, including mixed and long '
        'paths with one-sided local derivatives. Reuse the bound complete policy traces. '
        'Compare each finite response with the chained affine prediction and '
        'require its worst XY residual below half that parent path\'s certified '
        'corridor margin before a shared-parameter proposal. This is measured '
        'coverage at existing directions, not a bound for unmeasured updates. '
        'The stored shape-error audit remains separate; no optimistic time score '
        'or actual policy acceptance comes from this check.')
    start=time.perf_counter();records=[]
    with np.load(probes/'basis.npz') as z:directions=z['basis']*z['amplitudes'][:,None]
    for local in data['records']:
        p=local['path'];j=next(j for j,r in enumerate(physical['baseline']['results']) if r['original_path']==p)
        with np.load(source/local['xy_file']) as z:nominal=z['nominal'].copy();g=z['gradient'].copy()
        projected=np.einsum('tcd,jd->tcj',g,directions);margin=1.25-local['segment_max_distance_mm']
        errors=[]
        for c in physical['candidates']:
            with np.load(BASE/c['traces'][j]) as z:rows=z['rows'].copy()
            if len(rows)<len(nominal):
                assert np.linalg.norm(rows[-1,2:4])<.001
                rows=np.vstack([rows,np.repeat(rows[-1:],len(nominal)-len(rows),axis=0)])
            actual=rows[:len(nominal),:2]
            predicted=nominal+c['sign']*projected[:,:,c['direction']]
            error=np.linalg.norm(actual-predicted,axis=1)
            errors.append(dict(direction=c['direction'],sign=c['sign'],max_error_mm=float(error.max()),
                rms_error_mm=float(np.sqrt(np.mean(error**2))),maximum_error_tick=int(error.argmax()),
                max_measured_change_mm=float(np.linalg.norm(actual-nominal,axis=1).max())))
        worst=max(r['max_error_mm'] for r in errors)
        records.append(dict(path=p,errors=errors,corridor_margin_mm=margin,maximum_xy_error_mm=worst,
                            half_margin_check_passed=bool(worst<.5*margin)))
    passed=all(r['half_margin_check_passed'] for r in records)
    write_json(stage/'results.json',dict(parent_sha256=data['parent_sha256'],records=records,
        measured_coverage_passed=passed,new_direction_guaranteed=False,goal_complete=False))
    receipt(stage,dict(status='completed_full_xy_response_coverage',measured_coverage_passed=passed,
        new_actor_decisions=0,new_physics_ticks=0,wall_s=time.perf_counter()-start))
    print('TRANSITION_XY_COVERAGE',passed,[(r['path'],r['maximum_xy_error_mm'],r['corridor_margin_mm'])
        for r in sorted(records,key=lambda x:x['maximum_xy_error_mm'],reverse=True)[:5]],flush=True)


if __name__=='__main__':main()
