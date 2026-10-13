"""Bounded two-worker queues with explicit completion and failure logs."""
import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from common import ROOT, sha, write_json, utc
from paired_learning import SEEDS


def command(args,key,timeout=1800):
    folder=ROOT/'results'/'queue';folder.mkdir(parents=True,exist_ok=True)
    began=time.perf_counter()
    with (folder/(key+'.stdout.txt')).open('w',encoding='utf-8') as out, (folder/(key+'.stderr.txt')).open('w',encoding='utf-8') as err:
        try:
            result=subprocess.run([sys.executable,'-B',*args],cwd=ROOT,stdout=out,stderr=err,timeout=timeout)
            record=dict(key=key,args=args,returncode=result.returncode,wall_s=time.perf_counter()-began)
        except subprocess.TimeoutExpired:record=dict(key=key,args=args,returncode=None,status='timeout',wall_s=time.perf_counter()-began)
    write_json(folder/(key+'.json'),record);print('QUEUE',key,record['returncode'],round(record['wall_s'],2),flush=True)
    return record


def train_job(job):
    seed,arm,tag,clip=job
    key=f'train_{tag}_{seed}_{arm}'
    final=ROOT/'results'/'paired'/tag/f'seed_{seed}'/arm/'receipt.json'
    if final.exists():
        if not json.loads(final.read_text('utf-8'))['complete']:return dict(key=key,status='previous_failed_run_retained')
    else:
        result=command(['src/paired_learning.py','train','--seed',str(seed),'--arm',arm,'--tag',tag,'--clip',str(clip)],key)
        if result['returncode']!=0:return result
    return command(['src/run_evaluations.py','matrix','--split','validation','--seed',str(seed),'--method',arm,'--tag',tag],key+'_validation')


def train():
    for arm in ['adaptive','fixed']:
        rec=json.loads((ROOT/'results'/'paired'/'fullrange'/'seed_101'/arm/'receipt.json').read_text('utf-8'))
        assert rec['complete']
        for name in ['validation_trapezoid','validation_zigzag']:
            result=json.loads((ROOT/'results'/'validation'/f'fullrange_{arm}_101_{name}_0.001.json').read_text('utf-8'))
            assert result['success'],'Single-seed validation gate failed'
    jobs=[(s,a,'fullrange',.2) for s in SEEDS[1:] for a in ['adaptive','fixed']]
    jobs += [(101,a,'fullrange_clip_'+str(c),c) for c in [.1,.3] for a in ['adaptive','fixed']]
    write_json(ROOT/'audit'/'fullrange_training_registration.json',dict(created_utc=utc(),jobs=jobs,
        main_clip=.2,clip_comparison='Predetermined validation-only sensitivity; 0.2 retained for final test irrespective of validation ranking',
        main_independent_seeds=SEEDS,source_sha256=sha(ROOT/'src'/'paired_learning.py'),
        fair_budget='Each arm: 2000 Adam fit iterations, 4096 PPO interactions, 16 x 4 x 4 PPO minibatches; shared teacher selection cost separately reported.',
        teachers='Same deterministic teacher records; no historical model inheritance.',worker_count=2))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:records=list(pool.map(train_job,jobs))
    write_json(ROOT/'results'/'queue'/'fullrange_training_summary.json',records)


def freeze():
    freeze_file=ROOT/'audit'/'test_freeze.json'
    if freeze_file.exists():raise FileExistsError(freeze_file)
    files=[ROOT/'data'/'manifest.json',ROOT/'data'/'teacher_records_v3.npz',ROOT/'data'/'test_line_arc_primitives.json',ROOT/'runtime'/'source_manifest.json',ROOT/'baseline_records'/'supplementary_reproduction'/'frozen_policy.py']
    files += [ROOT/'src'/n for n in ['common.py','paired_learning.py','model_io.py','evaluate.py','metrics.py','execution_settings.py','runtime_configuration.py','runtime_bootstrap.py','run_evaluations.py','baselines/motion_overlap.py','baselines/legacy_s2.py']]
    for record in json.loads((ROOT/'runtime'/'source_manifest.json').read_text('utf-8')):
        p=ROOT/'runtime'/record['relative']
        if sha(p)!=record['sha256']:raise ValueError('Runtime source drift: '+str(p))
        files.append(p)
    for seed in SEEDS:
        for arm in ['adaptive','fixed']:
            stage=ROOT/'results'/'paired'/'fullrange'/f'seed_{seed}'/arm
            assert json.loads((stage/'receipt.json').read_text('utf-8'))['complete']
            files.extend(stage/n for n in ['final.pt','registration.json','receipt.json'])
    files+=list((ROOT/'data').glob('*.csv'))
    files+=list((ROOT/'baseline_records'/'supplementary_reproduction'/'models').glob('*.pt'))
    write_json(freeze_file,dict(created_utc=utc(),sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files},
        acceptance='All initial motion, ordered complete coverage, continuous command-segment feasibility, vector finite-difference limits, braking, >=20 ms rest. Failures have no completion time.',
        test_scope='Five whole synthetic programs, untouched by all parameter/model selection; common v2 observation-support correction. Frozen legacy weights are transfer tests of v2 executor, not new legacy training.',
        seeds=SEEDS,primary_dt=.001,resolution_check_dt=.0005,
        parameters='No test-driven selection or changes permitted; later corrections require retiring these tests as development.',
        confidence='Per-path seed mean/SD for successes, success counts/failure reasons for every seed; paired seed intervals only for paired complete successes, descriptive if n<3. No cross-path pseudo-replication.'))


def test():
    assert (ROOT/'audit'/'test_freeze.json').exists()
    jobs=[]
    for seed in SEEDS:
        for arm in ['adaptive','fixed']:
            jobs.append((['src/run_evaluations.py','matrix','--method',arm,'--seed',str(seed)],f'test_{seed}_{arm}'))
    for arm in ['legacy_adaptive','legacy_fixed']:
        jobs.append((['src/run_evaluations.py','matrix','--method',arm],f'test_{arm}'))
    jobs.append((['src/run_evaluations.py','baseline'], 'test_scmo'))
    jobs.append((['src/run_evaluations.py','baseline','--method','S2'], 'test_s2'))
    for arm in ['adaptive','fixed','legacy_adaptive','legacy_fixed']:
        jobs.append((['src/run_evaluations.py','matrix','--method',arm,'--seed','101','--dt','.0005'],f'resolution_{arm}'))
    jobs.append((['src/run_evaluations.py','baseline','--dt','.0005'],'resolution_scmo'))
    jobs.append((['src/run_evaluations.py','baseline','--method','S2','--dt','.0005'],'resolution_s2'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(lambda j:command(*j),jobs))
    write_json(ROOT/'results'/'queue'/'test_summary.json',results)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','freeze','test']);args=p.parse_args()
    globals()[args.mode]()
