"""Failure-inclusive validation and complete deterministic baseline runs."""
from pathlib import Path
import argparse
import json
import shutil
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'literature/ruckig_vendor'))
from common import sha,write_json,utc
from metrics import metrics


def verify_upstream():
    from ruckig import Ruckig,InputParameter,Trajectory,Result
    from baselines.ruckig_s2 import straight,transition_duration
    out=ROOT/'literature/ruckig_validation';out.mkdir(parents=True,exist_ok=True)
    inp=InputParameter(3)
    values=dict(current_position=[0.,0.,.5],current_velocity=[0.,-2.2,-.5],current_acceleration=[0.,2.5,-.5],
                target_position=[5.,-2.,-3.5],target_velocity=[0.,-.5,-2.],target_acceleration=[0.,0.,.5],
                max_velocity=[3.,1.,3.],max_acceleration=[3.,2.,1.],max_jerk=[4.,3.,2.],
                min_velocity=[-1.,-.5,-3.],min_acceleration=[-2.,-1.,-2.])
    for k,v in values.items():setattr(inp,k,v)
    tr=Trajectory(3);result=Ruckig(3).calculate(inp,tr)
    end=tr.at_time(tr.duration)
    errors=[float(np.max(np.abs(np.array(end[i])-values[name]))) for i,name in enumerate(('target_position','target_velocity','target_acceleration'))]
    records=[dict(test='unchanged_offline_example_inputs',source='https://github.com/pantor/ruckig/blob/v0.12.2/examples/02_position_offline.py',
                  inputs=values,result=str(result),duration_s=tr.duration,target_errors=errors,
                  initial_state_already_exceeds_limits=True,pass_check=max(errors)<1e-10)]
    samples=np.array([sum((list(x) for x in tr.at_time(t)),[]) for t in np.linspace(0,tr.duration,1001)])
    np.savez_compressed(out/'upstream_example.npz',time=np.linspace(0,tr.duration,1001),states=samples)
    # Independent exact single-axis rest-to-rest duration across all S-curve cases.
    for length in (.001,.01,.1,1.,10.,100.,1000.):
        V,A,J=100.,2000.,62500.
        lo,hi=0.,V
        for _ in range(80):
            v=.5*(lo+hi)
            if v*transition_duration(0,v,A,J)<=length:lo=v
            else:hi=v
        peak=lo;tacc=transition_duration(0,peak,A,J)
        expected=2*tacc+max(0.,length-peak*tacc)/peak
        native=straight(length,0.,0.,(V,A,J))
        error=abs(native.duration-expected)
        records.append(dict(test='independent_rest_to_rest_scurve',length_mm=length,duration_s=native.duration,
                            exact_duration_s=expected,error_s=error,pass_check=error<1e-10))
    write_json(out/'property_checks.json',dict(created_utc=utc(),upstream_version='0.12.2',checks=records,
               passed=sum(x['pass_check'] for x in records),attempted=len(records)))
    if not all(x['pass_check'] for x in records):raise AssertionError('Upstream validation failed')
    print('UPSTREAM_CHECKS',len(records),flush=True)


def run_one(reference,path_id,method,scale,dt,folder,tolerance=1.25,source_geometry=None):
    key=f'{method}_{path_id}_dt{round(dt*1e6)}us_scale{round(scale*1000)}'
    out=folder/key;folder.mkdir(parents=True,exist_ok=True)
    if out.with_suffix('.json').exists():return json.loads(out.with_suffix('.json').read_text('utf-8'))
    started=time.perf_counter();planner=None;error=None
    try:
        if method=='S2R':from baselines.ruckig_s2 import plan
        elif method=='S2':from baselines.legacy_s2 import plan
        elif method=='SCMO':
            from baselines.motion_overlap import plan as overlap
            tr=overlap(reference,dt=dt,tolerance=tolerance,limits=dict(v=100.,a=2000.,j=62500.))
            positions=tr['positions'];planner=tr['metrics']
        if method!='SCMO':positions,planner=plan(reference,dt=dt,tolerance=tolerance,corner_scale=scale,source_geometry=source_geometry)
        np.savez_compressed(out.with_suffix('.npz'),positions=positions,reference=reference,time=np.arange(len(positions))*dt,dt_s=dt)
        record=metrics(reference,positions,dt=dt,tolerance=tolerance)
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc)
        record=dict(success=False,complete_time_s=None,reason='planner_or_audit_exception',error=error)
    record.update(path_id=path_id,method=method,corner_scale=scale,wall_s=time.perf_counter()-started,
                  dt_s=dt,tolerance_mm=tolerance,planner=planner,trace=str(out.with_suffix('.npz').relative_to(ROOT)),
                  trace_sha256=sha(out.with_suffix('.npz')) if out.with_suffix('.npz').exists() else None)
    write_json(out.with_suffix('.json'),record)
    print('BASELINE_V3',key,record['success'],record['complete_time_s'],error,flush=True)
    return record


def validation():
    verify_upstream()
    folder=ROOT/'results/baseline_v3/validation_final'
    folder.mkdir(parents=True,exist_ok=True)
    source=json.loads((ROOT/'data/manifest.json').read_text('utf-8'))
    paths=[x for x in source['paths'] if x['split']=='validation']
    write_json(folder/'registration.json',dict(created_utc=utc(),paths=paths,scales=[1.,.75,.5],
        methods=['S2','S2R'],limits=[100.,2000.,62500.],tolerance_mm=1.25,dt_s=.001,
        selection='Within each method maximize full completions, then minimize total complete execution time; no final-test data'))
    rows=[];selected={}
    for method in ('S2','S2R'):
        scores=[]
        for scale in (1.,.75,.5):
            trial=[]
            for row in paths:
                ref=np.loadtxt(ROOT/'data'/(row['id']+'.csv'),delimiter=',',skiprows=1)
                record=run_one(ref,row['id'],method,scale,.001,folder);rows.append(record);trial.append(record)
            scores.append(dict(corner_scale=scale,completed=sum(r['success'] for r in trial),
                               complete_sum_s=sum(r['complete_time_s'] for r in trial if r['success'])))
        chosen=min(scores,key=lambda s:(-s['completed'],s['complete_sum_s']))
        selected[method]=dict(selected=chosen,candidates=scores)
    selection_file=ROOT/'results/baseline_v3/selection.json'
    if selection_file.exists():
        previous=ROOT/'results/baseline_v3/selection_before_long_path_numerics.json'
        if not previous.exists():shutil.copy2(selection_file,previous)
    write_json(selection_file,dict(created_utc=utc(),selection=selected,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'src/baselines/legacy_s2.py',ROOT/'src/baselines/ruckig_s2.py']},
        runtime_scope='All final trajectory acceptance uses unchanged nominal limits, full ordered traversal and terminal rest.'))


def development():
    selection=json.loads((ROOT/'results/baseline_v3/selection.json').read_text('utf-8'))['selection']
    folder=ROOT/'results/baseline_v3/development_v2'
    for case in ('original','dense'):
        ref=np.loadtxt(ROOT/'baseline_records/supplementary_reproduction/cases'/case/'reference.csv',delimiter=',',skiprows=1)
        geometry=json.loads((ROOT/'baseline_records/supplementary_reproduction/cases'/case/'geometry.json').read_text('utf-8'))
        for dt in (.001,.0005):
            for method in ('S2','S2R','SCMO'):
                scale=selection[method]['selected']['corner_scale'] if method in selection else 1.
                run_one(ref,case,method,scale,dt,folder,source_geometry=geometry)


def final():
    from runtime_bootstrap import setup
    setup()
    from runtime_configuration import configure_runtime
    selection=json.loads((ROOT/'results/baseline_v3/selection.json').read_text('utf-8'))['selection']
    registry=json.loads((ROOT/'prospective/final_v3/registration.json').read_text('utf-8'))
    frozen=ROOT/'results/baseline_v3/final_freeze_v3b.json'
    code=[ROOT/'src/baselines'/p for p in ('legacy_s2.py','ruckig_s2.py','motion_overlap.py','study_v3.py')]
    code += [ROOT/'prospective/final_v3/registration.json',ROOT/'results/baseline_v3/selection.json']
    # Bind the complete imported execution and native-library dependencies
    # before evaluating the replacement registry; v3a artifacts stay untouched.
    dependencies=json.loads((ROOT/'results/baseline_v3/dependency_provenance_receipt.json').read_text('utf-8'))['sha256']
    code += [ROOT/p for p in dependencies]
    code += [ROOT/'src/common.py',ROOT/'literature/ruckig_validation/sampler_regression_v11.json']
    code += [ROOT/'prospective/final_v3'/row['file'] for row in registry['paths']]
    digests={str(p.relative_to(ROOT)):sha(p) for p in code}
    if frozen.exists():
        if json.loads(frozen.read_text('utf-8'))['sha256']!=digests:raise ValueError('Final baseline code/selection changed after freeze')
    else:write_json(frozen,dict(created_utc=utc(),sha256=digests,methods=['S2','S2R','SCMO'],dt_s=[.001,.0005],
        revision='v3b', prior_results='results/baseline_v3/final', sampler_version='1.1-empty-phase-safe',
        amendment='The retired open-hook path exposed an empty sampled phase; the fix was independently regressed before this freeze.'))
    folder=ROOT/'results/baseline_v3/final_v3b'
    for row in registry['paths']:
        source=ROOT/'prospective/final_v3'/row['file']
        if sha(source)!=row['sha256']:raise ValueError('Final reference hash mismatch')
        reference=np.loadtxt(source,delimiter=',',skiprows=1)
        tolerance=row['evaluation_tolerance_mm']
        configure_runtime('v3',tolerance=tolerance)
        for dt in (.001,.0005):
            for method in ('S2','S2R','SCMO'):
                scale=selection[method]['selected']['corner_scale'] if method in selection else 1.
                record=run_one(reference,row['id'],method,scale,dt,folder,tolerance=tolerance)
                record.update(method=method,seed=None,path=row['id'],dt_s=dt,
                              original_tolerance_mm=row['original_tolerance_mm'],arc_chord_error_bound_mm=row['arc_chord_error_bound_mm'])
                target=ROOT/'results/final_v3b'/f'{method}_None_{row["id"]}_{dt}'
                target.parent.mkdir(parents=True,exist_ok=True)
                if record['trace_sha256']:
                    shutil.copy2(ROOT/record['trace'],Path(str(target)+'.npz'))
                write_json(Path(str(target)+'.json'),record)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['verify','validation','development','final']);args=p.parse_args()
    dict(verify=verify_upstream,validation=validation,development=development,final=final)[args.mode]()
