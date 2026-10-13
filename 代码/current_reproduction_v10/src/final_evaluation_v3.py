"""Freeze and run the unified final five-seed study; retain every outcome."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse
import json
from pathlib import Path
import numpy as np
from common import ROOT,sha,utc,write_json

OUT=ROOT/'results'/'final_v3b'
STUDY=ROOT/'results'/'unified_v3'
RESERVE=ROOT/'prospective'/'final_v3'
FREEZE=ROOT/'audit'/'final_v3b_freeze.json'
SEEDS=[101,211,307,401,503]


def read(path):return json.loads(Path(path).read_text('utf-8'))


def freeze():
    if FREEZE.exists():raise FileExistsError(FREEZE)
    selection=read(STUDY/'teacher_selection.json')
    source=read(STUDY/'registration.json')
    if not selection['gate_pass']:raise ValueError('Teacher gate not complete')
    if sha(ROOT/'src/unified_training_v3.py')!=source['source_sha256']:
        raise ValueError('Trainer changed since registration')
    for name,digest in source['protected_source_sha256'].items():
        if sha(ROOT/'src'/name)!=digest:raise ValueError('Protected training dependency changed '+name)
    if sha(STUDY/'shared_teacher_records.npz')!=selection['dataset_sha256']:
        raise ValueError('Teacher records changed')
    for row in source['paths']:
        if sha(ROOT/'data'/(row['id']+'.csv'))!=row['sha256']:
            raise ValueError('Training/validation reference changed '+row['id'])
    models=[]
    for seed in SEEDS:
        for arm in ('fixed','adaptive'):
            stage=STUDY/'paired'/f'seed_{seed}'/arm
            receipt=read(stage/'receipt.json');reg=read(stage/'registration.json')
            if not receipt['complete'] or not receipt['source_unchanged'] or not receipt['protected_sources_unchanged']:
                raise ValueError(f'Incomplete or changed training {seed} {arm}')
            if receipt['cost']['training_interactions']!=8192 or receipt['cost']['supervised_optimizer_steps']!=2000:
                raise ValueError('Unequal training budgets')
            if reg['dataset_sha256']!=selection['dataset_sha256']:
                raise ValueError('Unequal paired teacher data')
            file=stage/'final.pt'
            if sha(file)!=receipt['model_sha256']:raise ValueError('Changed model')
            if (reg['seed'],reg['arm'])!=(seed,arm):raise ValueError('Model identity mismatch')
            models.append(dict(method=arm,seed=seed,file=file.relative_to(ROOT).as_posix(),sha256=sha(file)))
    for arm in ('fixed','adaptive'):
        gate=read(STUDY/'paired'/'seed_101'/arm/'validation_08192.json')
        if not all(r['success'] for r in gate['records']):raise ValueError('Seed-101 pipeline gate failed')
    names=['final_evaluation_v3.py','unified_training_v3.py','paired_learning.py','common.py','evaluate.py','metrics.py','model_io.py','runtime_configuration.py','runtime_bootstrap.py','execution_settings.py','corridor_v3.py','register_final_v3.py','register_final_v3b.py']
    files={'src/'+name:sha(ROOT/'src'/name) for name in names}
    for row in read(ROOT/'runtime/source_manifest.json'):
        file=ROOT/'runtime'/row['relative']
        if sha(file)!=row['sha256']:raise ValueError('Historical runtime snapshot changed '+row['relative'])
        files[file.relative_to(ROOT).as_posix()]=row['sha256']
    files['runtime/source_manifest.json']=sha(ROOT/'runtime/source_manifest.json')
    files['data/manifest.json']=sha(ROOT/'data/manifest.json')
    for row in source['paths']:
        files['data/'+row['id']+'.csv']=row['sha256']
    for file in STUDY.rglob('*'):
        if file.is_file() and file.suffix in ('.json','.npz','.pt') and file.name!='models.json':
            files[file.relative_to(ROOT).as_posix()]=sha(file)
    reg=read(RESERVE/'registration.json')
    for row in reg['paths']:
        path=RESERVE/row['file']
        if sha(path)!=row['sha256']:raise ValueError('Final reference changed')
        files[path.relative_to(ROOT).as_posix()]=sha(path)
    files['prospective/final_v3/registration.json']=sha(RESERVE/'registration.json')
    files['prospective/final_v3/registration_v3a_retired.json']=sha(RESERVE/'registration_v3a_retired.json')
    files['audit/final_v3_registry_transition.json']=sha(ROOT/'audit/final_v3_registry_transition.json')
    write_json(STUDY/'models.json',dict(created_utc=utc(),scope='UNIFIED_V3',models=models))
    files['results/unified_v3/models.json']=sha(STUDY/'models.json')
    grid=[dict(path=p['id'],dt_s=dt,tolerance_mm=p['evaluation_tolerance_mm'],arc_error_bound_mm=p['arc_chord_error_bound_mm'],**m)
        for p in reg['paths'] for dt in (.001,.0005) for m in models if dt==.001 or m['seed']==101]
    write_json(FREEZE,dict(created_utc=utc(),models=models,grid=grid,sha256=files,
        training_registration_sha256=sha(STUDY/'registration.json'),teacher_sha256=sha(STUDY/'shared_teacher_records.npz'),
        selection='Final 8192 checkpoint of all five registered seeds; no seed/checkpoint/test-outcome selection.',
        scope='Five unseen whole synthetic programs; the analytic line/arc program reserves chord error from nominal tolerance.'))
    print('FROZEN',len(grid))


def run(worker=0,workers=1):
    import torch
    from evaluate import rollout
    from model_io import load_model
    torch.set_num_threads(1)
    frozen=read(FREEZE)
    for file,digest in frozen['sha256'].items():
        if sha(ROOT/file)!=digest:raise ValueError('Frozen input changed '+file)
    models={}
    if not 0<=worker<workers:raise ValueError('Invalid worker allocation')
    for n,cell in enumerate(frozen['grid']):
        if n%workers!=worker:continue
        stem=f"{cell['method']}_{cell['seed']}_{cell['path']}_{cell['dt_s']}"
        output=OUT/(stem+'.npz')
        if output.with_suffix('.json').exists():continue
        if cell['file'] not in models:
            if sha(ROOT/cell['file'])!=cell['sha256']:raise ValueError('Actual model differs from frozen cell')
            models[cell['file']]=load_model(ROOT/cell['file'])
        policy,saved=models[cell['file']]
        if (saved['seed'],saved['arm'])!=(cell['seed'],cell['method']):raise ValueError('Loaded checkpoint identity differs')
        ref=np.loadtxt(RESERVE/(cell['path']+'.csv'),delimiter=',',skiprows=1)
        result=rollout(policy,ref,output,dt=cell['dt_s'],tolerance=cell['tolerance_mm'],
            fixed_d=saved['d'] if cell['method']=='fixed' else None,runtime_version='v3')
        result.update(path=cell['path'],method=cell['method'],seed=cell['seed'],dt_s=cell['dt_s'],
            original_tolerance_mm=1.25,arc_chord_error_bound_mm=cell['arc_error_bound_mm'],
            model_sha256=cell['sha256'],freeze_sha256=sha(FREEZE),worker=worker,workers=workers)
        write_json(output.with_suffix('.json'),result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','run']);p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--output',type=Path,help='New output directory for an independent evaluation rerun; no change to frozen inputs.')
    args=p.parse_args()
    if args.output:
        OUT=args.output.resolve()
        if not OUT.is_relative_to(ROOT/'results'):raise ValueError('Output must stay under results')
    freeze() if args.mode=='freeze' else run(args.worker,args.workers)
