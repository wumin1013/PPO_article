"""Reaudit archived rollouts plus a common finite brake; no policy retraining.

All previously exposed splits are DEVELOPMENT diagnostics in this study.
The original planner prefix, failures and frozen result files remain unchanged.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import yaml

import preview_event_study as old
from terminal_braking import BrakingPolicy, plan_terminal_brake
from terminal_metrics import evaluate_terminal_trajectory

OUT=old.ROOT/'论文项目/revision_20260909_terminal'


def numerical_sources():
    files=old.source_state()['source_files'].copy()
    for name in ('preview_terminal_compare.py','terminal_braking.py','terminal_metrics.py'):
        path=Path(__file__).with_name(name)
        files[path.relative_to(old.ROOT).as_posix()]=old.sha(path)
    return files


def prepare():
    # Verify the immutable predecessor before accepting its saved policies.
    old.check_protocol()
    value=dict(version='terminal-comparison-v2',sources=numerical_sources(),
        prior_manifest_sha256=old.sha(old.DATA/'manifest.json'),
        prior_freeze_sha256=old.sha(old.OUT/'freeze.json'),
        brake=asdict(BrakingPolicy()),dt_s=.001,one_sided_tolerance_mm=1.25,
        geometric_boundary='Original ordered arrival plus same tolerance during entire stop; not exact endpoint positioning',
        evaluation='Independent differences of all final positions including splice and 20ms stationary window',
        methods=['selected_fixed','selected_hold','original_two_step'],seeds=list(old.SEEDS),
        training='Frozen predecessor models; no new training. Shared terminal adapter is not a learned-preview contribution.',
        data_status='All old splits exposed; terminal results are development diagnostics, not new independent test evidence.',
        two_step='Original native output fully audited; no new smoothing or speed retuning; infeasible state is not patched with zeros.',
        machine_execution_authorized=False)
    path=OUT/'comparison_v2_protocol.json'
    if path.exists() and old.read(path)!=value:
        raise ValueError('Preserve previous terminal protocol; use a separate version')
    if not path.exists():old.write(path,value)
    return value


def evaluate(split):
    protocol=prepare()
    root=OUT/'comparison_v2'/split
    if root.exists() and (root/'summary.json').exists():
        raise FileExistsError('Preserve completed terminal comparison')
    rows=[]
    for variant in ('selected_fixed','selected_hold'):
        for seed in old.SEEDS:
            model=old.OUT/'formal'/f'{variant}_seed{seed}'
            cfg=yaml.safe_load((model/'config.yaml').read_text(encoding='utf-8'))
            old.verify_training(model,old.FORMAL_STEPS)
            source=model/f'evaluation_{split}_nominal'
            for record in old.read(source/'summary.json')['rows']:
                path_id=record['path_id'];old.validate_result(record,model,path_id,split)
                file=source/f'{path_id}.npz'
                with np.load(file) as z: c={str(k):z['data'][:,i] for i,k in enumerate(z['columns'])}
                positions=np.c_[c['x_mm'],c['y_mm']];times=c['timestamp_s']
                reference=old.points(path_id)
                failure=None;tail=None
                if record['path_completed'] and record['termination_reason']=='success':
                    try:
                        tail=plan_terminal_brake(positions[-1],c['heading_rad'][-1],c['v_mm_s'][-1],
                            c['a_mm_s2'][-1],c['omega_rad_s'][-1],c['alpha_rad_s2'][-1],
                            dt=.001,limits=cfg['kinematic_constraints'])
                    except ValueError as exc:failure=str(exc)
                if tail:
                    d=tail['columns']
                    positions=np.vstack([positions,np.c_[d['x_mm'],d['y_mm']][1:]])
                    times=np.r_[times,times[-1]+d['time_s'][1:]]
                initial=dict(initial_velocity=[0.,0.],initial_acceleration=[0.,0.],
                    initial_heading=float(c['heading_rad'][0]),initial_angular_velocity=0.,initial_angular_acceleration=0.)
                result=evaluate_terminal_trajectory(reference,positions,times,tolerance_mm=1.25,
                    limits=cfg['kinematic_constraints'],witness_spacing_mm=.01,
                    termination_reason='completed' if tail else record['termination_reason'],
                    claimed_success=bool(tail),**initial)
                dest=root/f'{variant}_seed{seed}'/path_id
                dest.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest/'command.npz',positions=positions,timestamps=times)
                if tail:np.savez_compressed(dest/'brake.npz',**tail['columns'])
                result.update(variant=variant,seed=seed,path_id=path_id,split=split,
                    original_arrival_time_s=record['elapsed_time_s'],
                    source_trace_sha256=old.sha(file),checkpoint_sha256=old.sha(model/'checkpoint.pt'),
                    source_config_sha256=old.sha(model/'config.yaml'),
                    protocol_sha256=old.sha(OUT/'comparison_v2_protocol.json'),
                    command_sha256=old.sha(dest/'command.npz'),source_trace_path=file.relative_to(old.ROOT).as_posix(),
                    initial_state=initial,brake_failure=failure,
                    brake=None if tail is None else {k:v for k,v in tail.items() if k!='columns'},
                    role='Frozen-policy transfer with common terminal brake; development only')
                old.write(dest/'result.json',result);rows.append(result)
                print(f'{variant} {seed} {path_id}: {result["success"]}, {result["failure_reasons"]}',flush=True)
    for path in sorted((old.OUT/'two_step_event_v1'/f'{split}_nominal').glob('*/result.json')):
        saved=old.read(path);folder=path.parent
        command=np.loadtxt(folder/'command_with_initial.csv',delimiter=',',skiprows=1)
        reference=np.loadtxt(folder/'reference.csv',delimiter=',',skiprows=1)
        cfg=yaml.safe_load((folder.parent/'common_config.yaml').read_text(encoding='utf-8'))
        initial={k:v for k,v in saved['initial_state'].items() if k.startswith('initial_')}
        native=saved['native_terminal_rest_audit']
        result=evaluate_terminal_trajectory(reference,command[:,1:],command[:,0],tolerance_mm=1.25,
            limits=cfg['kinematic_constraints'],witness_spacing_mm=.01,
            termination_reason=native['termination_reason'],claimed_success=native['claimed_success'],**initial)
        result.update(variant='original_two_step',seed=None,path_id=saved['path_id'],split=split,
            source_trace_path=(folder/'command_with_initial.csv').relative_to(old.ROOT).as_posix(),
            source_trace_sha256=old.sha(folder/'command_with_initial.csv'),
            role='Original deterministic native-rest output; no tail fabrication',
            protocol_sha256=old.sha(OUT/'comparison_v2_protocol.json'))
        old.write(root/'original_two_step'/saved['path_id']/'result.json',result);rows.append(result)
    summaries={v:old.backend.summarize_runs([r for r in rows if r['variant']==v])
               for v in ('selected_fixed','selected_hold','original_two_step')}
    old.write(root/'summary.json',dict(rows=rows,summaries=summaries,command='python '+' '.join(sys.argv)))
    if protocol['sources']!=numerical_sources():raise ValueError('Sources changed during comparison')
    return summaries


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',choices=('validation','test'),required=True)
    print(json.dumps(evaluate(p.parse_args().split),indent=2))
