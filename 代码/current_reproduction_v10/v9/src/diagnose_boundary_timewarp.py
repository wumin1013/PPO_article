"""Bounded startup reparameterization on complete archived trajectories."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json
from pathlib import Path
import sys
import time
import numpy as np
BASE=Path(r'D:\PPO_article_work\JNN_dominance_20261011_v5')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'src'))
from common import sha,write_json,utc
from metrics_exact_index_v5 import metrics
from boundary_timewarp import candidates,geometry_prefix,START_S,END_S,JOIN_S,MAX_SAVING_S

OUT=ROOT/'boundary_timewarp_diagnostic'


if __name__=='__main__':
    OUT.mkdir(exist_ok=False)
    reg=dict(created_utc=utc(),source_sha256={str(p):sha(p) for p in
        (Path(__file__),ROOT/'src/boundary_timewarp.py')},
        purpose='Finite old-command diagnostic, not a new actor rollout. Preserve all candidates and full suffix; no source, tolerance, limits or completion definition change.',
        warp_start_s=START_S,warp_end_s=END_S,join_s=JOIN_S,max_saving_s=MAX_SAVING_S,
        methods=['JNN','FIXED'],cases=['piecewise_curvature_coupon_scale10','asymmetric_neck_contour_scale10'],
        dt_s=[.001,.0005],selection='Largest integer-tick saving with complete vector and prefix interval/gate acceptance; otherwise retain the original prefix.')
    write_json(OUT/'registration.json',reg)
    summary=[]
    for case in reg['cases']:
        for method in reg['methods']:
            for dt in reg['dt_s']:
                name=f'{case}_{method}_dt{round(dt*1e6)}us'
                folder='main' if method=='JNN' else 'fixed'
                source=BASE/'results/representative_scaled_v8/learned'/folder/(name+'.npz')
                record=json.loads(source.with_suffix('.json').read_text('utf-8'))
                assert sha(source)==record['trace_sha256'] and record['success']
                with np.load(source) as bundle:reference=bundle['reference'];positions=bundle['positions']
                joined_count=round(JOIN_S/dt)
                started=time.perf_counter();attempts=[];selected=None
                for prefix,item in candidates(positions,dt):
                    item['geometry_pass']=None
                    if item['vector_pass']:
                        item['geometry_pass']=geometry_prefix(reference,prefix,positions[:joined_count+1],
                            tolerance=record['tolerance_mm'])
                    trace=OUT/name/f"save{item['saved_ticks']:03d}.npz"
                    trace.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(trace,prefix=prefix,dt_s=dt)
                    item.update(trace=str(trace.relative_to(ROOT)),trace_sha256=sha(trace))
                    attempts.append(item)
                    if selected is None and item['vector_pass'] and item['geometry_pass']:
                        selected=(prefix,item)
                if selected:
                    prefix,item=selected
                    full=np.vstack((prefix,positions[joined_count+1:]))
                    audited=metrics(reference,full,dt=dt,tolerance=record['tolerance_mm'])
                    assert audited['success']
                    trace=OUT/name/'selected_full.npz'
                    np.savez_compressed(trace,reference=reference,positions=full,dt_s=dt)
                else:item=dict(saved_s=0.);audited=record;trace=source
                result=dict(case=case,method=method,dt_s=dt,source=str(source),source_sha256=sha(source),
                    original_s=record['complete_time_s'],saved_s=item['saved_s'],metrics=audited,
                    selected_trace=str(trace),selected_trace_sha256=sha(trace),candidates=attempts,
                    wall_s=time.perf_counter()-started,policy_observations='Planning trace only; no physical retimed observations claimed.')
                write_json(OUT/(name+'.json'),result);summary.append(result)
                print(name,'saved',item['saved_s'],'T',audited['complete_time_s'],flush=True)
    write_json(OUT/'summary.json',dict(registration_sha256=sha(OUT/'registration.json'),rows=summary))
