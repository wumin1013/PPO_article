"""Transfer frozen V2/NNC models to a final shared evaluation certificate.

recertify-validation reads saved validation commands only. evaluate executes
old policies only after every final-study checkpoint/configuration is frozen
and verified. Old guidance and policy parameters are retained; no new training
is claimed and no source result is overwritten. Import has no activation effect.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

import preview_direct_fine as fine
import preview_research_v2 as v2
from src.utils import revision_metrics as metrics

ROOT=v2.ROOT
SEEDS=(11,22,33,44,55)
OUTPUT_VERSION='final-matrix-v3'
LEGACY_V2_ENV=v2.make_env


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):v2.backend.write_json(path,value)


def final_study(name):
    module={'direct-fine':'preview_direct_fine','direct-explore':'preview_direct_explore'}[name]
    try:return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name==module:raise ValueError(f'Final study module is not available: {module}.py') from exc
        raise


def verify_freeze(base,required_keys,*,source_sha256=None,verify_protocol=False):
    """Verify every listed model/config before any held-out path or policy access."""
    freeze_path=base/'freeze.json'
    if not freeze_path.exists():raise ValueError(f'Final/baseline freeze is required before policy evaluation: {freeze_path}')
    value=read(freeze_path)
    if set(value['models'])!=set(required_keys) or set(value['configs'])!=set(required_keys):
        raise ValueError('Freeze must contain exactly the complete registered model/configuration set')
    for key in sorted(required_keys):
        if sha(base/key/'checkpoint.pt')!=value['models'][key] or sha(base/key/'config.yaml')!=value['configs'][key]:
            raise ValueError(f'Frozen checkpoint/configuration hash mismatch: {key}')
    if source_sha256 is not None and value['source_sha256']!=source_sha256:
        raise ValueError('Frozen source hash mismatch')
    if value['path_manifest_sha256']!=sha(v2.DATA/'manifest.json'):
        raise ValueError('Frozen path manifest mismatch')
    if verify_protocol:
        if value['protocol_sha256']!=sha(base/'protocol.json') or value['selection_sha256']!=sha(base/'selection.json'):
            raise ValueError('Frozen protocol/selection hash mismatch')
    return value


def validate_final(candidate):
    if not (candidate.OUT/'freeze.json').exists():
        raise ValueError(f'Final/baseline freeze is required before policy evaluation: {candidate.OUT/"freeze.json"}')
    protocol=read(candidate.OUT/'protocol.json')
    variants,seeds=protocol['formal_variants'],protocol['formal_seeds']
    if not variants or not seeds or len(set(variants))!=len(variants) or len(set(seeds))!=len(seeds):
        raise ValueError('Formal protocol requires nonempty, unique variants and seeds')
    required_keys={f'formal/{variant}_seed{seed}' for variant in variants for seed in seeds}
    return verify_freeze(candidate.OUT,required_keys,source_sha256=candidate.source_state()['source_sha256'],verify_protocol=True)


def baseline_runs():
    v2_keys={f'finetune/{variant}_seed{seed}' for variant in ['fixed','residual','absolute'] for seed in SEEDS}
    locations=[('absolute',v2.OUT,'finetune',v2_keys),
               ('nnc',v2.OUT/'nnc_auxiliary','runs',{f'runs/nnc_seed{s}' for s in SEEDS})]
    result=[]
    for baseline,base,stage,keys in locations:
        frozen=verify_freeze(base,keys)
        for seed in SEEDS:
            key=f'{stage}/{baseline}_seed{seed}';run=base/key
            provenance=read(run/'provenance.json');training=read(run/'training_summary.json')
            if provenance['source_sha256']!=frozen['source_sha256'] or provenance['config_sha256']!=frozen['configs'][key]:
                raise ValueError(f'Legacy training provenance mismatch: {key}')
            for name,expected in provenance['source_files'].items():
                if sha(ROOT/name)!=expected:raise ValueError(f'Legacy source changed; use its archived runtime before transfer: {name}')
            if training['checkpoint_sha256']!=frozen['models'][key]:raise ValueError('Training summary/checkpoint mismatch')
            result.append(dict(baseline=baseline,seed=seed,run=run,training=training,provenance=provenance,
                checkpoint_sha256=frozen['models'][key],config_sha256=frozen['configs'][key],
                source_archive_sha256=sha(run/'source.zip'),baseline_freeze_sha256=sha(base/'freeze.json')))
    return result


def reference_specs(split):
    if split=='regression':
        cfg=yaml.safe_load((ROOT/'论文项目/generated/configs/full_method_snapshot.yaml').read_text(encoding='utf-8'))
        return [dict(id=path['name'],historical_spec=path) for path in cfg['training']['path_curriculum']['paths']]
    return [path for path in v2.manifest()['paths'] if path['split']==split]


def reference_points(spec):
    return v2.backend._build_path(spec['historical_spec']) if 'historical_spec' in spec else v2.path_points(spec['id'])


def effective_config(original,candidate,regime):
    cfg=copy.deepcopy(original)
    common=candidate.config_for('fixed_mid',101)
    cfg['environment']['epsilon']=1.25 if regime=='tight' else common['environment']['epsilon']
    cfg['environment']['interpolation_period']=common['environment']['interpolation_period']
    cfg['environment']['max_steps']=common['revision']['episode_max_steps']
    cfg['kinematic_constraints']=copy.deepcopy(common['kinematic_constraints'])
    if regime=='low_jerk':cfg['kinematic_constraints']['MAX_JERK']=31250.
    cfg['revision'].update(episode_max_steps=common['revision']['episode_max_steps'],ordered_witness_spacing_mm=.01,
        evaluation_role='Frozen old policy transferred to common fine certificate; original policy/control parameters preserved; no new training')
    return cfg


@contextmanager
def fine_metric_scope():
    previous=metrics.ordered_projection
    metrics.ordered_projection=fine.fine_ordered_projection
    try:yield
    finally:metrics.ordered_projection=previous


def evaluate_trace(reference,rows,cfg,reason,claimed):
    with fine_metric_scope():
        return metrics.evaluate_trajectory(reference,[[r['x_mm'],r['y_mm']] for r in rows],
            [r['timestamp_s'] for r in rows],tolerance_mm=cfg['environment']['epsilon']/2,
            limits=cfg['kinematic_constraints'],termination_reason=reason,claimed_success=claimed,
            initial_velocity=[0.,0.],initial_acceleration=[0.,0.],initial_heading=rows[0]['heading_rad'],
            initial_angular_velocity=0.,initial_angular_acceleration=0.)


def transferred_rollout(item,reference,path_id,cfg):
    old_env,observation=LEGACY_V2_ENV(cfg,path_id,reference=reference)
    env=fine.FineTimeObjective(old_env.env,old_env.cfg,reference)
    agent=v2.BASE_AGENT(cfg,env)
    checkpoint=torch.load(item['run']/'checkpoint.pt',map_location='cpu',weights_only=False)
    agent.actor.load_state_dict(checkpoint['actor']);agent.actor.eval()
    rows=[v2.backend.sample_row(env,{},0.,[0.,0.,.5])]
    latencies=[];done=False;info={};began=time.perf_counter()
    with torch.no_grad():
        while not done:
            start=time.perf_counter_ns()
            latent=agent.actor(torch.as_tensor(observation,dtype=torch.float32).unsqueeze(0))[0][0].numpy()
            latencies.append((time.perf_counter_ns()-start)/1000.)
            action=np.clip(latent[:env.action_space_dim],env.action_space.low,env.action_space.high)
            observation,reward,done,info=env.step(action)
            rows.append(v2.backend.sample_row(env,info,reward,latent))
    return rows,info['done_reason'],dict(inference_median_us=float(np.median(latencies)),
        inference_p95_us=float(np.percentile(latencies,95)),evaluation_wall_s=time.perf_counter()-began)


def metadata(item,candidate,study_name,mode,split,regime,original,cfg,final_freeze):
    adapter_files=[Path(__file__),Path(fine.__file__),Path(fine.__file__).with_name('preview_direct_fine_batch.py')]
    return dict(adapter_version=OUTPUT_VERSION,mode=mode,study=study_name,split=split,regime=regime,baseline=item['baseline'],seed=item['seed'],
        comparison_role='Frozen legacy policy transferred to common evaluation, not independently retrained for the final method',
        training_run=item['run'].relative_to(ROOT).as_posix(),original_training_summary=item['training'],
        original_training_provenance=item['provenance'],original_config=original,effective_config=cfg,
        checkpoint_sha256=item['checkpoint_sha256'],original_config_sha256=item['config_sha256'],
        original_source_archive_sha256=item['source_archive_sha256'],baseline_freeze_sha256=item['baseline_freeze_sha256'],
        final_source_sha256=candidate.source_state()['source_sha256'],
        final_freeze_sha256=None if final_freeze is None else sha(candidate.OUT/'freeze.json'),
        final_protocol_sha256=sha(candidate.OUT/'protocol.json'),
        evaluation_adapter_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in adapter_files},
        path_manifest_sha256=sha(v2.DATA/'manifest.json'),witness_spacing_mm=.01,
        historical_path_config_sha256=sha(ROOT/'论文项目/generated/configs/full_method_snapshot.yaml') if split=='regression' else None,
        one_sided_tolerance_mm=cfg['environment']['epsilon']/2,
        original_guidance=original['reward_weights']['control_authority'],
        control_parameter_policy='Keep original control-authority/preview/state/KCM settings. The final gain15 is not injected into older policies.',
        common_differences='Common physical limits, interpolation period, horizon, fine coverage and initial-state convention; old training/guidance may differ from final policy. This does not isolate learned preview contribution.',
        initial_state='Reference start and tangent; zero feed/acceleration/angular rates',
        terminal_state='Ordered endpoint arrival and scalar constraints; no terminal-rest requirement',
        no_new_training=True,extra_training_steps=0,
        command='python '+' '.join(sys.argv))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',choices=['direct-fine','direct-explore'],default='direct-fine')
    commands=parser.add_subparsers(dest='mode',required=True)
    commands.add_parser('recertify-validation',help='Offline existing validation traces only; frozen labels remain')
    evaluate=commands.add_parser('evaluate',help='Run old frozen policies only after the final complete freeze verifies')
    evaluate.add_argument('--split',choices=['validation','test','regression'],required=True)
    evaluate.add_argument('--regime',choices=['nominal','tight','low_jerk'],default='nominal')
    args=parser.parse_args()
    candidate=final_study(args.study)
    final_freeze=validate_final(candidate) if args.mode=='evaluate' else None
    split=args.split if args.mode=='evaluate' else 'validation'
    regime=args.regime if args.mode=='evaluate' else 'nominal'
    selected=reference_specs(split)
    items=baseline_runs()
    output=candidate.OUT/'unified_baselines'/OUTPUT_VERSION/f'{args.mode}_{split}_{regime}'
    torch.set_num_threads(1)
    all_rows=[]
    for item in items:
        run=output/f"{item['baseline']}_seed{item['seed']}"
        original=yaml.safe_load((item['run']/'config.yaml').read_text(encoding='utf-8'))
        cfg=effective_config(original,candidate,regime)
        protocol=metadata(item,candidate,args.study,args.mode,split,regime,original,cfg,final_freeze)
        if (run/'protocol.json').exists() and read(run/'protocol.json')!=protocol:
            raise ValueError('Preserve earlier transfer output: protocol/adapter/source changed')
        write(run/'protocol.json',protocol)
        (run/'original_config.yaml').write_bytes((item['run']/'config.yaml').read_bytes())
        (run/'effective_config.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
        (run/'evaluation_adapter.py').write_bytes(Path(__file__).read_bytes())
        run_rows=[]
        for path in selected:
            path_id=path['id'];target=run/f'{path_id}.json'
            if target.exists():
                saved=read(target)
                if saved['protocol_sha256']!=sha(run/'protocol.json'):raise ValueError('Cached protocol hash mismatch')
                source=item['run']/'evaluation_validation_nominal' if args.mode=='recertify-validation' else run
                if saved['raw_trace_sha256']!=sha(source/f'{path_id}.npz'):
                    raise ValueError('Cached command trace hash mismatch')
                if args.mode=='recertify-validation' and saved['original_result_sha256']!=sha(source/f'{path_id}.json'):
                    raise ValueError('Cached original-result hash mismatch')
                run_rows.append(saved);all_rows.append(saved);continue
            reference=reference_points(path)
            original_result=None;trace_hash=None
            if args.mode=='recertify-validation':
                source=item['run']/'evaluation_validation_nominal'
                original_result=read(source/f'{path_id}.json')
                if (original_result['checkpoint_sha256']!=item['checkpoint_sha256']
                        or original_result['config_sha256']!=item['config_sha256']
                        or original_result['code_sha256']!=item['provenance']['source_sha256']
                        or original_result['path_id']!=path_id or original_result['seed']!=item['seed']
                        or original_result['split']!='validation'
                        or original_result['effective_config_sha256']!=sha(source/'effective_config.yaml')):
                    raise ValueError('Saved validation belongs to another legacy model/configuration')
                old_effective=yaml.safe_load((source/'effective_config.yaml').read_text(encoding='utf-8'))
                if any(old_effective['environment'][k]!=cfg['environment'][k] for k in ['epsilon','interpolation_period','max_steps']) or old_effective['kinematic_constraints']!=cfg['kinematic_constraints']:
                    raise ValueError('Offline recertification requires identical original physical protocol')
                z=np.load(source/f'{path_id}.npz');rows=[dict(zip(z['columns'].tolist(),values)) for values in z['data']]
                trace_hash=sha(source/f'{path_id}.npz');reason=original_result['termination_reason'];runtime={}
                claimed=original_result['claimed_success']
            else:
                rows,reason,runtime=transferred_rollout(item,reference,path_id,cfg)
                claimed=reason=='success'
                v2.backend.save_trace(run/f'{path_id}.npz',rows);trace_hash=sha(run/f'{path_id}.npz')
            result=evaluate_trace(reference,rows,cfg,reason,claimed)
            record=dict(metrics=result,baseline=item['baseline'],seed=item['seed'],path_id=path_id,split=split,regime=regime,
                mode=args.mode,original_result=original_result,protocol_sha256=sha(run/'protocol.json'),
                original_result_sha256=None if original_result is None else sha(source/f'{path_id}.json'),
                effective_config_sha256=sha(run/'effective_config.yaml'),checkpoint_sha256=item['checkpoint_sha256'],
                raw_trace_sha256=trace_hash,original_labels_mutated=False,
                diagnostic_fine_coverage_and_physical_feasibility=bool(result['path_completed'] and result['geometry_feasible'] and all(c['satisfied'] is not False for c in result['constraints'].values())),
                offline_note='Original termination/claimed-success retained; certificate-only feasibility is not a newly executed or trained result' if original_result is not None else None,
                comparison_role=protocol['comparison_role'],**runtime)
            write(target,record);run_rows.append(record);all_rows.append(record)
            print(item['baseline'],item['seed'],path_id,'strict',result['success'],'coverage',result['path_completed'],flush=True)
        write(run/'summary.json',dict(rows=run_rows))
    write(output/'summary.json',dict(mode=args.mode,study=args.study,split=split,regime=regime,rows=all_rows,
        summary=metrics.summarize_runs([r['metrics'] for r in all_rows]),
        original_training_seeds=list(SEEDS),new_training_steps=0,
        aggregate_warning='Do not merge legacy transferred models into the main equally trained fixed-versus-direct effect estimate'))
    print(output/'summary.json')


if __name__=='__main__':main()
