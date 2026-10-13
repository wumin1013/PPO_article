"""Run the unchanged traditional planner on frozen-study reference paths."""
import argparse
import csv
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.algorithms.two_step_baseline import (TwoStepConstraints, SmoothingParams,
                                              SchedulerParams, run_two_step_baseline)


def load_study(name):
    if name in ('direct-gain','direct-fine','direct-explore'):
        return importlib.import_module('preview_'+name.replace('-','_')).install()
    modules=dict(direct='preview_direct_research',gate='preview_gate_research',v2='preview_research_v2')
    return importlib.import_module(modules[name])


def require_test_freeze(study, name):
    """Validate only freeze metadata; never load or execute a held-out path."""
    freeze_path=study.OUT/'freeze.json'
    if not freeze_path.exists():
        raise ValueError(f'Freeze the final {name} study before evaluating test baselines')
    if name in ('direct-fine','direct-explore'):
        from preview_unified_baselines import validate_final
        validate_final(study)
        return
    frozen=json.loads(freeze_path.read_text(encoding='utf-8'))
    if (frozen['path_manifest_sha256']!=study.backend.sha(study.DATA/'manifest.json')
            or frozen['source_sha256']!=study.source_state()['source_sha256']):
        raise ValueError(f'Frozen {name} source or path manifest changed; test baselines refused')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--study', choices=['direct','direct-gain','direct-fine','direct-explore','gate','v2'], default='direct',
                        help='Final study whose frozen models and path manifest authorize test evaluation')
    parser.add_argument('--split',choices=['validation','test','regression'],required=True)
    parser.add_argument('--regime',choices=['nominal','tight','low_jerk'],default='nominal')
    args=parser.parse_args()
    study=load_study(args.study)
    if args.split=='test' or args.study in ('direct-fine','direct-explore'):
        require_test_freeze(study,args.study)
    shared=study.config_for('pretrain_mid' if args.study=='v2' else 'fixed_mid',101)
    if args.regime=='tight': shared['environment']['epsilon']=1.25
    if args.regime=='low_jerk': shared['kinematic_constraints']['MAX_JERK']=31250.
    if args.split=='regression':
        import yaml
        cfg=yaml.safe_load((study.ROOT/'论文项目/generated/configs/full_method_snapshot.yaml').read_text(encoding='utf-8'))
        paths=[(p['name'],study.backend._build_path(p)) for p in cfg['training']['path_curriculum']['paths']]
    else:
        paths=[(p['id'],study.path_points(p['id'])) for p in study.manifest()['paths'] if p['split']==args.split]
    limits=shared['kinematic_constraints']
    constraints=TwoStepConstraints(*(float(limits[key]) for key in
        ['MAX_VEL','MAX_ACC','MAX_JERK','MAX_ANG_VEL','MAX_ANG_ACC','MAX_ANG_JERK']))
    dt=float(shared['environment']['interpolation_period'])
    tolerance=float(shared['environment']['epsilon'])/2
    max_steps=int(shared['revision'].get('episode_max_steps',12000))
    destination=study.OUT/'two_step'/f'{args.split}_{args.regime}'
    protocol=dict(source=study.source_state(),script_sha256=study.backend.sha(__file__),
        study=args.study,manifest_sha256=study.backend.sha(study.DATA/'manifest.json'),dt_s=dt,max_steps=max_steps,
        one_sided_tolerance_mm=tolerance,constraints=asdict(constraints),
        smoothing=asdict(SmoothingParams()),scheduler=asdict(SchedulerParams()),
        shared_parameters_source='study.config_for(fixed_mid or V2 pretrain_mid, seed=101), same regime overrides as PPO evaluation',
        initial_state='Reference start, smoothed initial tangent heading, zero feed/acceleration and angular rates; t=0 supplied to independent audit separately from trace.csv',
        constraint_scope='Same scalar feed and angular limits are audited. The unchanged scheduler does not enforce angular or Cartesian axis limits.',
        boundary_policy='Full retained trace is audited against one-sided tolerance; baseline does not stop early at a boundary failure as PPO does.',
        trained=False,terminal_note='Planner requests rest; PPO endpoint arrival does not guarantee rest. No same-terminal-state time superiority claim.')
    if (destination/'protocol.json').exists():
        if json.loads((destination/'protocol.json').read_text(encoding='utf-8'))!=protocol:
            raise ValueError('Preserve existing baseline protocol')
    else:
        study.backend.write_json(destination/'protocol.json',protocol)
        (destination/'preview_two_step_eval_snapshot.py').write_bytes(Path(__file__).read_bytes())
    for path_id,reference in paths:
        run=destination/path_id
        if (run/'result.json').exists(): continue
        run.mkdir(parents=True,exist_ok=True)
        began=time.perf_counter()
        result=run_two_step_baseline(path_name=path_id,reference_path=reference,constraints=constraints,
            dt=dt,half_epsilon=tolerance,closed=bool(np.linalg.norm(reference[0]-reference[-1])<1e-9),max_steps=max_steps)
        with (run/'trace.csv').open('w',encoding='utf-8',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(result.trace_rows[0]))
            writer.writeheader();writer.writerows(result.trace_rows)
        metrics=result.metrics['output_command_audit']
        smoothed_path=run/'smoothed_path.csv'
        np.savetxt(smoothed_path,result.smoothed.smoothed_path,delimiter=',',header='x_mm,y_mm',comments='')
        tangent=result.smoothed.smoothed_path[1]-result.smoothed.smoothed_path[0]
        stop_tolerance=max(constraints.max_vel*SchedulerParams().feedrate_safety_ratio*SchedulerParams().stop_velocity_ratio,1e-5)
        record=dict(metrics=metrics,path_id=path_id,split=args.split,regime=args.regime,
            protocol_sha256=study.backend.sha(destination/'protocol.json'),
            raw_trace_sha256=study.backend.sha(run/'trace.csv'),
            initial_time_s=0.,initial_position_mm=result.smoothed.smoothed_path[0].tolist(),
            initial_velocity_mm_s=0.,initial_acceleration_mm_s2=0.,
            initial_heading_rad=float(np.arctan2(tangent[1],tangent[0])),
            initial_angular_velocity_rad_s=0.,initial_angular_acceleration_rad_s2=0.,
            terminal_feed_tolerance_mm_s=stop_tolerance,
            terminal_feed_mm_s=metrics['terminal_feed_mm_s'],
            planner_internal_metrics={key:value for key,value in result.metrics.items() if key!='output_command_audit'},
            smoothed_path_sha256=study.backend.sha(smoothed_path),
            wall_time_s=time.perf_counter()-began,command='python '+' '.join(sys.argv))
        study.backend.write_json(run/'result.json',record)
        print(path_id,metrics['success'],metrics['failure_reasons'],flush=True)


if __name__=='__main__': main()
