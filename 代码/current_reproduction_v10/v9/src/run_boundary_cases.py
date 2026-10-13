"""Execute a registered offline boundary-refinement grid on complete planning traces."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=Path(r'D:\PPO_article_work\JNN_dominance_20261011_v5')
sys.path.insert(0,str(BASE/'src'))
from common import sha,write_json,utc
from refine_boundaries import refine
OUT=ROOT/'boundary_refinement'


def read(p):return json.loads(Path(p).read_text('utf-8'))


def register(group):
    regfile=OUT/(group+'_registration.json');assert not regfile.exists()
    assert read(ROOT/'boundary_validation/summary.json')['adoptable']
    if group=='scaled':
        inventory=read(ROOT/'audit/baseline_inventory.json')
        inputs=[r for r in inventory['rows'] if r['case'].endswith('_scale10') and r['method'] in ('JNN','FIXED')]
    else:
        inventory=read(ROOT/'unified_v5_inputs/trace_summary.json')
        inputs=inventory['rows']
    assert len(inputs)==8
    sources=[Path(__file__),ROOT/'src/refine_boundaries.py',ROOT/'src/boundary_timewarp.py',
             ROOT/'terminal_research/terminal_candidate.py']
    specs=[]
    for item in inputs:
        record=Path(item['record']);trace=Path(item['trace'])
        assert sha(record)==item['record_sha256'] and sha(trace)==item['trace_sha256']
        specs.append(dict(case=item['case'],method=item['method'],dt_s=item['dt_s'],
            record=str(record),record_sha256=sha(record),trace=str(trace),trace_sha256=sha(trace)))
    reg=dict(created_utc=utc(),stage='offline complete command refinement',group=group,inputs=specs,
        source_sha256={str(p):sha(p) for p in sources},
        validation_summary_sha256=sha(ROOT/'boundary_validation/summary.json'),
        no_parameter_selection=True,model_parameters_unchanged=True,
        command_provenance='Frozen actor generates the original plan; refine only its rest boundaries. Retimed physical commands carry separate arrays from original planning observations and actions.',
        failed_original_plans_retained=True,whole_plan_fallback_on_any_new_audit_failure=True,
        same_complete_geometry_vector_rest_contract=True,shared_by=['JNN','FIXED'],
        main_model_sha256='c16d45e38bab022ce8f289148aac4f3dcb3559f98b18bdfa580e23aea4a99764',
        fixed_model_sha256='40af7ea30bec92d9cd42e060b2d71ee72919c4cc50c3b28281ff036a0790f469')
    write_json(regfile,reg);print('REGISTERED',sha(regfile),flush=True)


def worker(group,index):
    regfile=OUT/(group+'_registration.json');reg=read(regfile);spec=reg['inputs'][index]
    for p,value in reg['source_sha256'].items():assert sha(p)==value,p
    assert sha(spec['record'])==spec['record_sha256'] and sha(spec['trace'])==spec['trace_sha256']
    record=read(spec['record'])
    assert record['model_sha256']==reg['main_model_sha256' if spec['method']=='JNN' else 'fixed_model_sha256']
    with np.load(spec['trace']) as z:
        reference=z['reference'];positions=z['positions'];rows=z['rows']
        observations=z['observations'];actions=z['actions']
    target=OUT/'results'/group/f"{spec['case']}_{spec['method']}_dt{round(spec['dt_s']*1e6)}us.npz"
    target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists()
    cpu=time.process_time()
    new,audit,evidence,prefixes=refine(reference,positions,rows,dt=spec['dt_s'],
        tolerance=record['tolerance_mm'],completed=record['success'])
    refine_cpu=time.process_time()-cpu
    np.savez_compressed(target,reference=reference,positions=new,
        planning_observations=observations,planning_actions=actions,
        **{f'candidate_prefix_{i}':p for i,p in enumerate(prefixes)})
    result=dict(audit,case=spec['case'],method=spec['method'],
        source_record=spec['record'],source_record_sha256=spec['record_sha256'],
        source_planning_trace=spec['trace'],source_planning_trace_sha256=spec['trace_sha256'],
        source_complete_time_s=record['complete_time_s'],model_sha256=record['model_sha256'],
        trace_sha256=sha(target),registration_sha256=sha(regfile),refinement=evidence,
        runtime_version='v9_offline_boundary_refinement',new_training_updates=0,
        cost=dict(original_planning_cost=record.get('cost'),original_run_wall_s=record.get('wall_s'),
            refinement_cpu_s=refine_cpu,refinement_wall_s=evidence['refinement_wall_s'],
            scope='Original deterministic planning plus measured offline command refinement and acceptance; no real-time claim.'))
    write_json(target.with_suffix('.json'),result)
    print(spec['case'],spec['method'],spec['dt_s'],audit['success'],audit['complete_time_s'],flush=True)


def matrix(group,jobs):
    reg=read(OUT/(group+'_registration.json'))
    def run(i):
        spec=reg['inputs'][i]
        log=OUT/'logs'/group/f"{spec['case']}_{spec['method']}_dt{round(spec['dt_s']*1e6)}us.log"
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('x',encoding='utf-8') as stream:
            result=subprocess.run([sys.executable,'-B',__file__,'worker','--group',group,'--index',str(i)],
                stdout=stream,stderr=subprocess.STDOUT,timeout=1800.)
        print(log.read_text('utf-8')[-1200:],flush=True)
        return result.returncode
    with ThreadPoolExecutor(max_workers=jobs) as pool:assert not any(pool.map(run,range(len(reg['inputs']))))
    rows=[]
    for spec in reg['inputs']:
        path=OUT/'results'/group/f"{spec['case']}_{spec['method']}_dt{round(spec['dt_s']*1e6)}us.json"
        record=read(path)
        rows.append(dict(case=spec['case'],method=spec['method'],dt_s=spec['dt_s'],
            record=str(path),record_sha256=sha(path),trace=str(path.with_suffix('.npz')),
            trace_sha256=record['trace_sha256'],metrics=record))
    write_json(OUT/(group+'_summary.json'),dict(registration_sha256=sha(OUT/(group+'_registration.json')),rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('register','run','worker'))
    p.add_argument('--group',choices=('scaled','original'),required=True);p.add_argument('--index',type=int)
    p.add_argument('--jobs',type=int,default=2);a=p.parse_args()
    if a.mode=='register':register(a.group)
    elif a.mode=='worker':worker(a.group,a.index)
    else:matrix(a.group,a.jobs)
