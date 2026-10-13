"""Evaluate preserved NNC policies in their byte-verified archived runtime.

The parent independently audits raw commands with the event-study certificate.
No NNC training or source normalization is performed. Test evaluation requires
the complete new study freeze; transferred models remain auxiliary evidence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'PPO_project/src').is_dir())
REPLAY=ROOT/'论文项目/revision_20260908/frozen_replay_explore'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def archive_worker(job):
    # This child has imported only the standard library before selecting the
    # archived package paths. Never mix live and archived src modules.
    job=Path(job).resolve();request=read(job)
    if request['split']=='test':
        # The gate runs in another process, leaving this worker free of live
        # src imports before it selects the byte-preserved archived runtime.
        gate=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--verify-freeze-only'],cwd=ROOT,check=False)
        if gate.returncode:raise ValueError('Event test freeze verification failed')
        event=ROOT/'论文项目/revision_20260909_event_preview/study'
        if request['event_freeze_sha256']!=sha(event/'freeze.json'):raise ValueError('Request freeze binding changed')
        if request['event_source_sha256']!=read(event/'freeze.json')['source_sha256']:raise ValueError('Request source binding changed')
    if request['script_sha256']!=sha(__file__):raise ValueError('Worker script differs from request snapshot')
    event_manifest=ROOT/'PPO_project/data/preview_event_v1/manifest.json'
    if request['manifest_sha256']!=sha(event_manifest):raise ValueError('Request path manifest changed')
    if request['split'] in ('validation','test'):
        allowed={p['id']:p for p in read(event_manifest)['paths'] if p['split']==request['split']}
        if len(request['paths'])!=len(allowed) or {p['id'] for p in request['paths']}!=set(allowed):raise ValueError('Request split/path assignment mismatch')
    for spec in request['paths']:
        for suffix in ('.npz','.worker.json'):
            if (job.parent/(spec['id']+suffix)).exists():raise FileExistsError('Preserve existing archived-worker output')
    sys.path[:0]=[str(REPLAY/'PPO_project/tools'),str(REPLAY/'PPO_project')]
    import numpy as np
    import torch
    import yaml
    import preview_unified_baselines as old
    if Path(old.ROOT).resolve()!=REPLAY.resolve():raise ValueError('Wrong archived import root')
    item=next(i for i in old.baseline_runs() if i['baseline']=='nnc' and i['seed']==request['seed'])
    cfg=yaml.safe_load((item['run']/'config.yaml').read_text(encoding='utf-8'))
    original=__import__('copy').deepcopy(cfg)
    common=request['common']
    cfg['environment'].update(epsilon=common['epsilon'],interpolation_period=common['dt_s'],max_steps=common['max_steps'])
    cfg['kinematic_constraints']=common['limits']
    cfg['revision'].update(episode_max_steps=common['max_steps'],ordered_witness_spacing_mm=.01)
    torch.set_num_threads(1)
    for spec in request['paths']:
        reference=np.loadtxt(spec['reference_file'],delimiter=',',skiprows=1)
        if sha(spec['reference_file'])!=spec['reference_sha256']:raise ValueError('Reference changed')
        if request['split'] in ('validation','test'):
            registered=event_manifest.parent/allowed[spec['id']]['reference_file']
            if sha(registered)!=allowed[spec['id']]['reference_sha256'] or not np.array_equal(reference,np.loadtxt(registered,delimiter=',',skiprows=1)):raise ValueError('Reference differs from registered split geometry')
        rows,reason,runtime=old.transferred_rollout(item,reference,spec['id'],cfg)
        dest=job.parent/spec['id'];old.v2.backend.save_trace(dest.with_suffix('.npz'),rows)
        write(dest.with_suffix('.worker.json'),dict(reason=reason,runtime=runtime,reference_sha256=spec['reference_sha256'],
            original_config=original,effective_config=cfg,original_training_summary=item['training'],
            original_training_provenance=item['provenance'],checkpoint_sha256=item['checkpoint_sha256'],
            original_config_sha256=item['config_sha256'],source_archive_sha256=item['source_archive_sha256'],
            old_freeze_sha256=item['baseline_freeze_sha256'],request_sha256=sha(job),
            trace_sha256=sha(dest.with_suffix('.npz')),script_sha256=sha(__file__)))


def require_freeze(study):
    study.check_protocol();frozen=read(study.OUT/'freeze.json')
    expected={f'formal/{v}_seed{s}' for v in study.FORMAL for s in study.SEEDS}
    if set(frozen['models'])!=expected or set(frozen['configs'])!=expected:raise ValueError('Incomplete freeze')
    for key in expected:
        if frozen['models'][key]!=sha(study.OUT/key/'checkpoint.pt') or frozen['configs'][key]!=sha(study.OUT/key/'config.yaml'):raise ValueError('Frozen model/config changed')
        study.verify_training(study.OUT/key,study.FORMAL_STEPS)
        study.validation_rows(study.OUT/key)
    for key,path in [('protocol_sha256',study.OUT/'protocol.json'),('selection_sha256',study.OUT/'selection.json'),('manifest_sha256',study.DATA/'manifest.json')]:
        if frozen[key]!=sha(path):raise ValueError('Frozen registration changed')
    if frozen['source_sha256']!=study.source_state()['source_sha256']:raise ValueError('Frozen source changed')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--archived-worker',type=Path)
    parser.add_argument('--verify-freeze-only',action='store_true')
    parser.add_argument('--split',choices=['validation','test','regression'],default='validation')
    parser.add_argument('--seed',type=int,choices=[11,22,33,44,55],default=11)
    parser.add_argument('--output',type=Path,help='New destination for an additional reproducibility run')
    args=parser.parse_args()
    if args.verify_freeze_only:
        sys.path.insert(0,str(ROOT/'PPO_project/tools'))
        import preview_event_study as study
        require_freeze(study);return
    if args.archived_worker:archive_worker(args.archived_worker);return
    import numpy as np
    import yaml
    import preview_event_study as study
    from src.utils.revision_metrics import evaluate_trajectory
    import prepare_auxiliary_replay
    # Validate every archived source/model before starting a policy process.
    archive_verification=prepare_auxiliary_replay.verify(REPLAY)
    study.check_protocol()
    if args.split=='test':require_freeze(study)
    cfg=study.config_for('selected_fixed',args.seed)
    if args.split=='regression':
        legacy=yaml.safe_load((study.ROOT/'论文项目/generated/configs/full_method_snapshot.yaml').read_text(encoding='utf-8'))
        specs=[(p['name'],study.backend._build_path(p)) for p in legacy['training']['path_curriculum']['paths']]
    else:specs=[(p['id'],study.points(p['id'])) for p in study.manifest()['paths'] if p['split']==args.split]
    dest=(args.output or study.OUT/'nnc_transfer'/args.split/f'nnc_seed{args.seed}').resolve()
    if not dest.is_relative_to((study.OUT/'nnc_transfer').resolve()):raise ValueError('Keep transfers under this study nnc_transfer directory')
    if dest.exists():raise FileExistsError('Preserve previous transfer; use its read-only audit output')
    dest.mkdir(parents=True)
    paths=[]
    for path_id,reference in specs:
        file=dest/f'{path_id}.reference.csv';np.savetxt(file,reference,delimiter=',',header='x_mm,y_mm',comments='')
        paths.append(dict(id=path_id,reference_file=str(file),reference_sha256=sha(file)))
    common=dict(epsilon=cfg['environment']['epsilon'],dt_s=cfg['environment']['interpolation_period'],
        max_steps=cfg['revision']['episode_max_steps'],limits=cfg['kinematic_constraints'])
    request=dict(seed=args.seed,split=args.split,common=common,paths=paths,archive_verification=archive_verification,
        event_source_sha256=study.source_state()['source_sha256'],manifest_sha256=sha(study.DATA/'manifest.json'),
        event_freeze_sha256=sha(study.OUT/'freeze.json') if (study.OUT/'freeze.json').exists() else None,
        script_sha256=sha(__file__),command='python '+' '.join(sys.argv),
        role='Preserved 98,304-step NNC policies, no additional training; auxiliary transfer, not equal-budget preview ablation')
    job=dest/'request.json';write(job,request)
    (dest/'evaluation_adapter.py').write_bytes(Path(__file__).read_bytes())
    with (dest/'worker.log').open('w',encoding='utf-8') as stream:
        result=subprocess.run([sys.executable,str(dest/'evaluation_adapter.py'),'--archived-worker',str(job)],cwd=REPLAY,stdout=stream,stderr=subprocess.STDOUT,check=False)
    if result.returncode:raise RuntimeError(f'Archived worker failed; retain {dest}/worker.log')
    results=[]
    for path_id,reference in specs:
        worker=read(dest/f'{path_id}.worker.json');trace=dest/f'{path_id}.npz'
        if worker['request_sha256']!=sha(job) or worker['trace_sha256']!=sha(trace) or worker['script_sha256']!=request['script_sha256'] or worker['script_sha256']!=sha(dest/'evaluation_adapter.py'):raise ValueError('Worker provenance changed')
        z=np.load(trace);columns=z['columns'].tolist();a=z['data'];column=lambda k:a[:,columns.index(k)]
        metric=evaluate_trajectory(reference,np.column_stack([column('x_mm'),column('y_mm')]),column('timestamp_s'),
            tolerance_mm=common['epsilon']/2,limits=common['limits'],termination_reason=worker['reason'],claimed_success=worker['reason']=='success',
            initial_velocity=[0.,0.],initial_acceleration=[0.,0.],initial_heading=float(column('heading_rad')[0]),
            initial_angular_velocity=0.,initial_angular_acceleration=0.,witness_spacing_mm=.01)
        row=dict(path_id=path_id,seed=args.seed,metrics=metric,worker_sha256=sha(dest/f'{path_id}.worker.json'),trace_sha256=sha(trace),
            request_sha256=sha(job),comparison_role=request['role'],new_training_steps=0)
        write(dest/f'{path_id}.json',row);results.append(row)
        print('NNC archived',args.seed,path_id,metric['success'],metric['failure_reasons'],flush=True)
    write(dest/'summary.json',dict(rows=results,request_sha256=sha(job),script_sha256=sha(__file__)))


if __name__=='__main__':main()
