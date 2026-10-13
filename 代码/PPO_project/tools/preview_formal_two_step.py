"""Registered unchanged two-step baseline with the formal full-rest audit.

Prepare reads configuration/source metadata only. New holdout coordinates are
accessible only after the complete ten-model gate passes. No output samples,
rest suffix, velocity, acceleration or planner parameters are repaired here.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

import preview_formal_continuation as formal
import preview_formal_holdout as gate
import preview_two_step_eval as legacy
from terminal_metrics import DEFAULT_REST_POLICY, evaluate_terminal_trajectory

ROOT = Path(__file__).resolve().parents[2]
FORMAL = ROOT/'论文项目/revision_20260910_continued/formal_v1'
OUT = FORMAL/'two_step_original_v1'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def _json_value(value):
    """Keep nonfinite original diagnostics explicit without invalid JSON."""
    if isinstance(value, dict): return {str(k):_json_value(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray): return _json_value(value.tolist())
    if isinstance(value, np.generic): return _json_value(value.item())
    if isinstance(value,float) and not np.isfinite(value):
        return 'NaN' if np.isnan(value) else '+Infinity' if value>0 else '-Infinity'
    return value


def write_new(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(_json_value(value),stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def configuration():
    # Sensor configuration reads only old train/validation metadata. It does
    # not load weights or any preview_continuation_v1 holdout coordinates.
    cfg=formal.sensor.configuration('preview_time','fixed3',11)
    return dict(dt_s=float(cfg['environment']['interpolation_period']),
        one_sided_tolerance_mm=float(cfg['environment']['epsilon'])/2,
        tolerance_bandwidth_mm=float(cfg['environment']['epsilon']),
        max_steps=int(cfg['environment']['max_steps']),
        limits={key:float(cfg['kinematic_constraints'][key]) for key in
                ('MAX_VEL','MAX_ACC','MAX_JERK','MAX_ANG_VEL','MAX_ANG_ACC','MAX_ANG_JERK')},
        smoothing=asdict(legacy.SmoothingParams()),scheduler=asdict(legacy.SchedulerParams()),
        rest_policy=dict(DEFAULT_REST_POLICY),witness_spacing_mm=.01)


def source_state(protocol):
    files=dict(protocol['sources']['continuation']['source_files'])
    # Include the imported legacy wrapper and the fixed ten-model access gate.
    for name in ('preview_two_step_eval.py','preview_formal_holdout.py','preview_formal_two_step.py'):
        path=Path(__file__).with_name(name)
        files[path.relative_to(ROOT).as_posix()]=sha(path)
    for name,digest in files.items():
        if sha(ROOT/name)!=digest: raise ValueError(f'Frozen numerical source changed: {name}')
    return dict(source_files=files,source_sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest())


def prepare():
    formal_protocol=read(FORMAL/'protocol.json')
    cfg=configuration()
    if (cfg['dt_s'],cfg['one_sided_tolerance_mm'],cfg['max_steps']) != (.001,1.25,60000):
        raise ValueError('Baseline registration requires the common 1ms/1.25mm/60s configuration')
    value=dict(version='formal-original-two-step-v1',configuration=cfg,source=source_state(formal_protocol),
        formal_protocol_file=(FORMAL/'protocol.json').relative_to(ROOT).as_posix(),
        formal_protocol_sha256=sha(FORMAL/'protocol.json'),
        holdout_manifest=formal_protocol['holdout_manifest'],
        algorithm='Unchanged preview_two_step_eval.run_two_step_baseline; fixed geometric smoothing followed by original feed scheduling',
        algorithm_file='PPO_project/src/algorithms/two_step_baseline.py',
        algorithm_sha256=sha(ROOT/'PPO_project/src/algorithms/two_step_baseline.py'),
        model_gate='preview_formal_holdout.validate_freeze: all ten independent models and all registered validation candidates before manifest/coordinate access',
        defaults='Default SmoothingParams and SchedulerParams, no parameter tuning on new holdout',
        initial_state='Original smoothed path start at t=0, tangent heading, zero feed/acceleration/angular rates',
        output_policy='Preserve every original command; prepend only the declared initial t=0 point; never append a fabricated rest tail or clip/repair output',
        rest_rule='Require the same full20ms position-derived rest window; lack of rest is a retained failure',
        constraint_scope='Independently audit the same six scalar feed/turn limits. The original scheduler does not guarantee angular or Cartesian axis limits; internal scalar a/j are not final-output guarantees',
        comparison_rule='No efficiency superiority comparison without common complete successes including all constraints and rest',
        budget='60000 original1ms samples, at most60s, no added terminal commands',
        statistical_unit='Deterministic baseline once per path; never count the same trajectory once per PPO seed as independent baseline replications',
        nonfinite_json_policy='Original nonfinite diagnostic numbers represented explicitly as NaN/+Infinity/-Infinity strings; raw NPZ numeric arrays remain original',
        preparation_reads_holdout_coordinates=False,trained=False,machine_execution_authorized=False)
    path=OUT/'protocol.json'
    if path.exists():
        previous=read(path)
        value['registered_utc']=previous['registered_utc']
        if previous!=value: raise ValueError('Preserve registered original baseline protocol')
    else:
        value['registered_utc']=datetime.now(timezone.utc).isoformat()
        write_new(path,value)
    source_zip=OUT/'source.zip'
    if not source_zip.exists():
        with zipfile.ZipFile(source_zip,'x',zipfile.ZIP_DEFLATED) as archive:
            for name in value['source']['source_files']: archive.write(ROOT/name,name)
    with zipfile.ZipFile(source_zip) as archive:
        for name,digest in value['source']['source_files'].items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=digest:
                raise ValueError('Registered original baseline source archive changed')
    return value


def command_arrays(result):
    smoothed=np.asarray(result.smoothed.smoothed_path,dtype=float)
    rows=result.trace_rows
    positions=np.vstack([smoothed[0],np.asarray([[r['x'],r['y']] for r in rows],dtype=float).reshape(-1,2)])
    timestamps=np.r_[0.,[r['time'] for r in rows]]
    tangent=smoothed[1]-smoothed[0]
    return positions,timestamps,float(np.arctan2(tangent[1],tangent[0]))


def _csv(path, rows, fields=None):
    fields=fields or list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


def verify_result(run, binding):
    manifest=read(run/'delivery_manifest.json')
    if manifest['binding']!=binding: raise ValueError('Baseline result belongs to another source/protocol/freeze')
    for name,digest in manifest['files'].items():
        if sha(run/name)!=digest: raise ValueError(f'Baseline result changed: {name}')
    return read(run/'result.json')


def run_one(path_id, reference, family, run, protocol, binding, role):
    run=Path(run)
    if (run/'delivery_manifest.json').exists(): return verify_result(run,binding)
    if run.exists() and any(run.iterdir()):
        raise FileExistsError('Preserve incomplete original baseline output; resume with a new output directory')
    run.mkdir(parents=True,exist_ok=True);began=time.perf_counter()
    cfg=protocol['configuration'];limits=cfg['limits']
    constraints=legacy.TwoStepConstraints(*(limits[key] for key in
        ('MAX_VEL','MAX_ACC','MAX_JERK','MAX_ANG_VEL','MAX_ANG_ACC','MAX_ANG_JERK')))
    reference=np.asarray(reference,dtype=float)
    planner_started=time.perf_counter()
    try:
        result=legacy.run_two_step_baseline(path_name=path_id,reference_path=reference,
            constraints=constraints,dt=cfg['dt_s'],half_epsilon=cfg['one_sided_tolerance_mm'],
            closed=bool(np.linalg.norm(reference[0]-reference[-1])<1e-9),max_steps=cfg['max_steps'],
            smoothing_params=legacy.SmoothingParams(**cfg['smoothing']),scheduler_params=legacy.SchedulerParams(**cfg['scheduler']))
    except Exception as error:
        np.savez_compressed(run/'commands.npz',reference=reference,
            command_positions=np.empty((0,2)),timestamp_s=np.empty(0))
        _csv(run/'commands.csv',[],['timestamp_s','x_mm','y_mm'])
        _csv(run/'raw_trace.csv',[],['time','x','y'])
        write_new(run/'internal_metrics.json',dict(available=False,error=repr(error)))
        record=dict(success=False,failure_reasons=['planner_exception'],error=repr(error),
            valid_trace=False,elapsed_time_s=None,planner_internal_metrics_available=False)
        planning_wall=time.perf_counter()-planner_started;audit_wall=0.
    else:
        planning_wall=time.perf_counter()-planner_started
        positions,timestamps,heading=command_arrays(result)
        audit_started=time.perf_counter()
        reason=result.metrics['termination_status']
        record=evaluate_terminal_trajectory(reference,positions,timestamps,
            tolerance_mm=cfg['one_sided_tolerance_mm'],limits=limits,initial_heading=heading,
            initial_velocity=[0.,0.],initial_acceleration=[0.,0.],
            initial_angular_velocity=0.,initial_angular_acceleration=0.,rest_policy=cfg['rest_policy'],
            termination_reason=reason,claimed_success=reason=='success',witness_spacing_mm=cfg['witness_spacing_mm'])
        audit_wall=time.perf_counter()-audit_started
        fields=list(dict.fromkeys(key for row in result.trace_rows for key in row))
        np.savez_compressed(run/'commands.npz',reference=reference,
            smoothed_path=result.smoothed.smoothed_path,command_positions=positions,timestamp_s=timestamps,
            **{f'raw_trace__{key}':np.asarray([row.get(key,np.nan) for row in result.trace_rows]) for key in fields})
        _csv(run/'raw_trace.csv',result.trace_rows,fields or ['time','x','y'])
        _csv(run/'commands.csv',[dict(timestamp_s=float(t),x_mm=float(p[0]),y_mm=float(p[1])) for p,t in zip(positions,timestamps)])
        _csv(run/'smoothed_path.csv',[dict(x_mm=float(p[0]),y_mm=float(p[1])) for p in result.smoothed.smoothed_path])
        write_new(run/'internal_metrics.json',result.metrics)
        record.update(initial_heading_rad=heading,initial_position_mm=positions[0].tolist(),
            initial_time_s=0.,raw_command_count=len(result.trace_rows),output_sample_count=len(positions),
            original_termination_status=reason,planner_internal_metrics_available=True,
            planner_terminal_internal_feed_mm_s=result.trace_rows[-1]['v'] if result.trace_rows else None,
            planner_stop_velocity_threshold_mm_s=max(limits['MAX_VEL']*cfg['scheduler']['feedrate_safety_ratio']*cfg['scheduler']['stop_velocity_ratio'],1e-5))
    record.update(method='two_step_original',seed=None,independent_training=False,path_id=path_id,family=family,
        data_role=role,source_sha256=protocol['source']['source_sha256'],binding=binding,
        planning_wall_s=planning_wall,independent_terminal_audit_wall_s=audit_wall,
        recorded_path_wall_s=time.perf_counter()-began,command='python '+' '.join(sys.argv),
        appended_rest_samples=0,altered_original_command_samples=0,
        angular_constraints_enforced_by_original_planner=False,cartesian_axis_constraints_enforced_by_original_planner=False,
        efficiency_comparison='Only common complete successes with PPO may enter time comparison; no automatic superiority claim')
    write_new(run/'result.json',record)
    files={path.name:sha(path) for path in run.iterdir() if path.is_file()}
    write_new(run/'delivery_manifest.json',dict(binding=binding,files=files))
    print(json.dumps(dict(path_id=path_id,success=record['success'],failure_reasons=record['failure_reasons']),ensure_ascii=False),flush=True)
    return record


def evaluate(freeze_path):
    protocol=prepare()
    # No held-out manifest or coordinate access is allowed before this gate.
    checked=gate.validate_freeze(freeze_path,ROOT)
    signature=checked['common_configuration']
    cfg=protocol['configuration']
    if (signature['environment']['interpolation_period']!=cfg['dt_s'] or
        signature['environment']['epsilon']/2!=cfg['one_sided_tolerance_mm'] or
        signature['constraints']!=cfg['limits'] or signature['environment']['max_steps']!=cfg['max_steps']):
        raise ValueError('Frozen learning methods and original baseline must share evaluation constraints')
    if checked['freeze']['protocol']['sha256']!=protocol['formal_protocol_sha256']:
        raise ValueError('Baseline was registered against another formal protocol')
    manifest,data=gate.frozen_manifest(checked,ROOT)
    from generate_preview_continuation_paths import load_path
    binding=dict(protocol_sha256=sha(OUT/'protocol.json'),freeze_sha256=checked['freeze_sha256'],
        holdout_manifest_sha256=checked['freeze']['holdout_manifest']['sha256'])
    rows=[]
    for item in manifest['paths']:
        reference=load_path(item['id'],data)
        rows.append(run_one(item['id'],reference,item['family'],OUT/'test'/item['id'],
            protocol,binding,'Original deterministic baseline on independently reserved paths'))
    target=OUT/'test/summary.json'
    value=dict(binding=binding,rows=rows,count=len(rows),successes=sum(row['success'] for row in rows),
        deterministic_runs_per_path=1,all_failures_retained=True,
        comparison_rule=protocol['comparison_rule'])
    if target.exists():
        if read(target)!=_json_value(value): raise ValueError('Preserve completed original baseline summary')
    else: write_new(target,value)
    return value


def smoke(path_id='mixed_lengths_validation'):
    protocol=prepare()
    entries=[p for p in formal.old.manifest()['paths'] if p['split']=='validation']
    entry=next((p for p in entries if p['id']==path_id),None)
    if entry is None: raise ValueError('Smoke permits existing validation paths only')
    reference=formal.old.points(path_id)
    binding=dict(protocol_sha256=sha(OUT/'protocol.json'),split='existing_validation_smoke')
    return run_one(path_id,reference,entry['family'],OUT/'smoke'/path_id,protocol,binding,
        'Existing validation-only workflow smoke; not new holdout evidence')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('prepare')
    p=commands.add_parser('evaluate');p.add_argument('--freeze',type=Path,default=FORMAL/'freeze.json')
    p=commands.add_parser('smoke');p.add_argument('--path-id',default='mixed_lengths_validation')
    args=parser.parse_args()
    result=prepare() if args.command=='prepare' else evaluate(args.freeze) if args.command=='evaluate' else smoke(args.path_id)
    print(json.dumps(_json_value(result),ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__': main()
