"""Registered common finite startup compression for every traditional cell.

No baseline planner, source trajectory, terminal phase, or acceptance rule is
modified. All failed inputs and every generated candidate remain visible.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime,timezone
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
BASE=Path(r'D:\PPO_article_work\JNN_dominance_20261011_v5')
sys.path.insert(0,str(BASE/'src'))
sys.path.insert(0,str(ROOT/'src'))
from boundary_timewarp import candidates,geometry_prefix,START_S,END_S,JOIN_S,MAX_SAVING_S,LIMITS
from metrics_exact_index_v5 import metrics


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text('utf-8'))


def write(path,value,exclusive=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x' if exclusive else 'w',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2)
        stream.write('\n')


def register():
    inventory=ROOT/'audit/baseline_inventory.json'
    data=read(inventory)
    rows=[row for row in data['rows'] if row['method'] in ('S2','S2R','SCMO','COPP')]
    assert len(rows)==32 and sum(row['metrics']['success'] for row in rows)==28
    for row in rows:
        assert sha(row['record'])==row['record_sha256']
        if row['trace']:assert sha(row['trace'])==row['trace_sha256']
        else:assert not row['metrics']['success'] and row['metrics']['complete_time_s'] is None
    reg=dict(created_utc=datetime.now(timezone.utc).isoformat(),inventory=str(inventory),
        inventory_sha256=sha(inventory),cases=data['cases'],rows=rows,
        grid='Four named complete development programs x S2/S2R/SCMO/COPP x 1/0.5 ms =32 cells; 28 successful and 4 failed inputs retained.',
        source_sha256={str(p):sha(p) for p in (Path(__file__),ROOT/'src/boundary_timewarp.py',BASE/'src/metrics_exact_index_v5.py')},
        warp_start_s=START_S,warp_end_s=END_S,join_s=JOIN_S,max_saving_s=MAX_SAVING_S,
        vector_limits=LIMITS.tolist(),selection='Largest integer-tick compression passing the unchanged boundary_timewarp candidate vector test and prefix full-interval/gate-state test. Independently audit complete assembled commands before adoption. If complete audit fails, retain the original trace without trying another candidate.',
        terminal_policy='No change to baseline terminal commands or rest.',
        failure_policy='Original baseline failures retain null completion time and no invented trajectory; all attempts and exceptions retained.',
        scope='Common offline command startup refinement check on all registered traditional inputs. No algorithm reproduction or independent final test claim.')
    write(HERE/'registration.json',reg,exclusive=True)
    print('REGISTERED',len(rows),'cells',flush=True)


def verify():
    reg=read(HERE/'registration.json')
    assert sha(reg['inventory'])==reg['inventory_sha256']
    for path,digest in reg['source_sha256'].items():assert sha(path)==digest
    return reg


def cell(row,reg):
    run_id=row['run_id'];folder=HERE/'results'/run_id
    folder.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    definition=reg['cases'][row['case']]
    result=dict(run_id=run_id,case=row['case'],method=row['method'],dt_s=row['dt_s'],
        source_record=row['record'],source_record_sha256=row['record_sha256'],
        source_trace=row['trace'],source_trace_sha256=row['trace_sha256'],
        original_metrics=row['metrics'],original_complete_time_s=row['metrics']['complete_time_s'],
        candidates=[],adopted=False,saved_s=0.,selected_trace=row['trace'],
        selected_trace_sha256=row['trace_sha256'],metrics=row['metrics'],
        acceptance_origin='Original SHA-bound complete audit')
    assert sha(row['record'])==row['record_sha256']
    if not row['metrics']['success']:
        result.update(status='original_baseline_failure_retained',saved_s=None,wall_s=time.perf_counter()-started)
        write(folder/'result.json',result,exclusive=True)
        print(run_id,'original failure retained',flush=True)
        return result
    assert sha(row['trace'])==row['trace_sha256']
    with np.load(row['trace'],allow_pickle=False) as archive:
        positions=archive['positions'];reference=archive['reference']
    assert sha(definition['reference'])==definition['reference_sha256']
    expected=np.loadtxt(definition['reference'],delimiter=',',skiprows=1)
    assert np.array_equal(reference,expected)
    dt=row['dt_s'];join=round(JOIN_S/dt);chosen=None
    try:
        for prefix,raw in candidates(positions,dt):
            attempt=dict(raw)
            attempt['geometry_pass']=None
            if attempt['vector_pass']:
                attempt['geometry_pass']=geometry_prefix(reference,prefix,positions[:join+1],
                    tolerance=definition['evaluation_tolerance_mm'])
            file=folder/f"candidate_save{attempt['saved_ticks']:03d}.npz"
            np.savez_compressed(file,prefix=prefix,dt_s=dt)
            attempt.update(trace=str(file),trace_sha256=sha(file))
            result['candidates'].append(attempt)
            if chosen is None and attempt['vector_pass'] and attempt['geometry_pass']:
                chosen=(prefix,attempt)
        if chosen is not None:
            prefix,attempt=chosen
            full=np.vstack((prefix,positions[join+1:]))
            assert np.array_equal(prefix[-3:],positions[join-2:join+1])
            assert np.array_equal(full[len(prefix):],positions[join+1:])
            file=folder/'candidate_full.npz'
            np.savez_compressed(file,reference=reference,positions=full,dt_s=dt)
            audit=metrics(reference,full,dt=dt,tolerance=definition['evaluation_tolerance_mm'])
            result['complete_candidate_audit']=dict(trace=str(file),trace_sha256=sha(file),metrics=audit,
                same_suffix_exact=True,same_join_last_three_exact=True,saved_s=attempt['saved_s'])
            if audit['success']:
                result.update(adopted=True,saved_s=attempt['saved_s'],selected_trace=str(file),
                    selected_trace_sha256=sha(file),metrics=audit,
                    acceptance_origin='New independent complete-vector, full-command-interval, all-gate, final-rest audit',
                    status='startup_compression_accepted')
            else:result['status']='full_candidate_failed_original_retained'
        else:result['status']='no_feasible_compression_original_retained'
    except Exception as exc:
        result.update(status='diagnostic_exception_original_retained',error=type(exc).__name__+': '+str(exc))
    result['wall_s']=time.perf_counter()-started
    write(folder/'result.json',result,exclusive=True)
    print(run_id,result['status'],'saved',result['saved_s'],'T',result['metrics']['complete_time_s'],flush=True)
    return result


def matrix():
    reg=verify()
    for row in reg['rows']:cell(row,reg)
    summarize()


def summarize():
    reg=verify();rows=[]
    learned_file=ROOT/'boundary_refinement/scaled_summary.json'
    learned=read(learned_file)['rows'] if learned_file.exists() else []
    jnn={(r['case'],r['dt_s']):r for r in learned if r['method']=='JNN'}
    for registered in reg['rows']:
        file=HERE/'results'/registered['run_id']/'result.json'
        if not file.exists():
            rows.append(dict(run_id=registered['run_id'],case=registered['case'],method=registered['method'],
                dt_s=registered['dt_s'],status='not_finished',complete_time_s=None))
            continue
        result=read(file)
        competitor=jnn.get((result['case'],result['dt_s']))
        time_jnn=competitor['metrics']['complete_time_s'] if competitor else None
        time_base=result['metrics']['complete_time_s']
        row=dict(run_id=result['run_id'],case=result['case'],method=result['method'],dt_s=result['dt_s'],
            status=result['status'],original_complete_time_s=result['original_complete_time_s'],
            complete_time_s=time_base,saved_s=result['saved_s'],adopted=result['adopted'],
            success=result['metrics']['success'],selected_trace=result['selected_trace'],
            selected_trace_sha256=result['selected_trace_sha256'],record=str(file),record_sha256=sha(file),
            candidate_count=len(result['candidates']),
            vector_pass_count=sum(a['vector_pass'] for a in result['candidates']),
            geometry_pass_count=sum(a['geometry_pass'] is True for a in result['candidates']),
            refined_jnn_time_s=time_jnn,refined_jnn_minus_baseline_s=(time_jnn-time_base) if time_jnn is not None and time_base is not None else None,
            refined_jnn_faster=(time_jnn<time_base) if time_jnn is not None and time_base is not None else None,
            wall_s=result['wall_s'])
        rows.append(row)
    summary=dict(registration_sha256=sha(HERE/'registration.json'),rows=rows,
        grid_count=len(reg['rows']),completed_check_count=sum(r['status']!='not_finished' for r in rows),
        successful_input_count=28,failed_input_count=4,
        compressed_count=sum(r.get('adopted',False) for r in rows),
        jnn_comparison_source=str(learned_file) if learned_file.exists() else None,
        jnn_comparison_source_sha256=sha(learned_file) if learned_file.exists() else None,
        comparison_scope='JNN times are read only from completed boundary-refinement summaries, never inferred from old learned traces. Missing original/dense refined values remain null until those runs finish.')
    write(HERE/'summary.json',summary)
    print('SUMMARY',summary['completed_check_count'],'compressed',summary['compressed_count'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('register','matrix','summary'))
    command=parser.parse_args().command
    {'register':register,'matrix':matrix,'summary':summarize}[command]()
