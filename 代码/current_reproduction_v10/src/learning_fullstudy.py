"""Separately register four further seeds using the unchanged development pilot."""
import argparse
import json
import shutil
import sys
from pathlib import Path
import learning_pilot as pilot
from common import ROOT, sha, utc, write_json

PILOT=pilot.OUT
OUT=ROOT/'results'/'development_v2'/'learning_002'
SEEDS=[211,307,401,503]


def read(path):return json.loads(path.read_text('utf-8'))


def register():
    file=OUT/'registration.json'
    if file.exists():return read(file)
    evidence=[]
    for arm in ('fixed','adaptive'):
        stage=PILOT/'paired'/'seed_101'/arm
        receipt=read(stage/'receipt.json');validation=read(stage/'validation_08192.json')
        if not receipt['complete'] or len(validation['records'])!=2 or not all(r['success'] for r in validation['records']):
            raise RuntimeError('Both seed 101 arms must finish 8192 decisions and pass both validation programs')
        evidence.append(dict(arm=arm,receipt_sha256=sha(stage/'receipt.json'),validation_sha256=sha(stage/'validation_08192.json'),
            final_model_sha256=sha(stage/'final.pt'),validation=validation['records']))
    original=read(PILOT/'registration.json')
    if sha(ROOT/'src'/'learning_pilot.py')!=original['source_sha256']:raise ValueError('Reused source changed')
    OUT.mkdir(parents=True,exist_ok=False)
    for name in ('shared_teacher_records.npz','teacher_selection.json'):
        shutil.copy2(PILOT/name,OUT/name)
    registration=dict(original)
    registration.update(created_utc=utc(),scope='DEVELOPMENT EXTENSION: remaining four independently initialized paired seeds; updated common initialization and training protocol',
        parent_pilot_registration_sha256=sha(PILOT/'registration.json'),
        extension_source_sha256=sha(__file__),extension_source='src/learning_fullstudy.py',
        seeds=SEEDS,arms=['fixed','adaptive'],pilot_gate=evidence,
        execution_environment=dict(python=sys.executable,python_version=sys.version),
        execution_schedule='Up to two separate one-thread training processes concurrently; wall times include shared-host contention. Online latency is measured separately.',
        development_selection='No additional parameter search. Same selected teacher bytes, feature construction, fit/PPO budgets and fixed final 8192 checkpoint. Record every validation outcome and retain all failure prefixes.',
        random_streams=dict(actor_initialization='seed',critic_initialization='seed+100000',
            action_sampling='Continues the torch stream set by critic initialization seed+100000 after critic tensor construction; checkpoint torch_rng_state is authoritative.',
            feature_and_minibatch_sampling='NumPy default_rng(seed), shared sequential stream',path_sampling='NumPy default_rng(seed+10000)'),
        attribution='Teacher quality and PPO budget both change from v1; any combined performance change belongs to the updated shared protocol, not necessarily PPO or preview.',
        final_test_executions_permitted=0)
    registration['paired_training']=dict(original['paired_training'])
    registration['paired_training'].pop('seed',None)
    registration['paired_training']['seeds']=SEEDS
    registration['paired_training']['selection']='All four registered seed pairs run to the same 8192 decision budget. No best-seed or checkpoint selection.'
    write_json(file,registration)
    return registration


def train(seed,arm):
    registration=register()
    if seed not in SEEDS:raise ValueError('Unregistered extension seed')
    if sha(__file__)!=registration['extension_source_sha256']:raise ValueError('Extension source changed')
    pilot.OUT=OUT
    pilot.train(arm,seed)
    stage=OUT/'paired'/f'seed_{seed}'/arm
    raw=stage/'registration.json'
    shutil.copy2(raw,stage/'registration_emitted_by_pilot.json')
    record=read(raw)
    record.update(scope=registration['scope'],extension_registration_sha256=sha(OUT/'registration.json'),
        extension_source_sha256=sha(__file__),random_streams=registration['random_streams'],
        metadata_correction='The reused pilot emits a nominal action-sampling seed label; this independent extension replaces it with the exact stream ancestry. Its original emitted registration is retained byte for byte.')
    write_json(raw,record)


def index():
    models=[]
    for seed in (101,*SEEDS):
        for arm in ('fixed','adaptive'):
            folder=(PILOT if seed==101 else OUT)/'paired'/f'seed_{seed}'/arm
            receipt=read(folder/'receipt.json')
            if not receipt['complete']:raise RuntimeError(f'Incomplete model: {seed} {arm}')
            file=folder/'final.pt'
            if sha(file)!=receipt['model_sha256']:raise ValueError('Final model changed')
            models.append(dict(method=arm,seed=seed,file=str(file.relative_to(ROOT)),sha256=sha(file)))
    write_json(OUT/'models.json',dict(scope='Updated shared protocol; final checkpoint, all registered seeds',models=models))
    print('WROTE',OUT/'models.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['register','train','index'])
    parser.add_argument('--seed',type=int);parser.add_argument('--arm',choices=['fixed','adaptive'])
    args=parser.parse_args()
    if args.mode=='register':register()
    elif args.mode=='train':train(args.seed,args.arm)
    else:index()
