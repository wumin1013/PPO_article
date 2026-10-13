"""Recompute baseline acceptance from full saved positions without replanning."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from common import sha,write_json,utc
from metrics import metrics
from runtime_configuration import configure_runtime


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--require-complete',action='store_true');args=parser.parse_args()
    frozen=ROOT/'results/baseline_v3/final_freeze_v3b.json'
    for name,digest in json.loads(frozen.read_text('utf-8'))['sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen dependency changed: '+name)
    output=ROOT/'audit/baseline_final_v3b_trajectory_reaudit.json'
    prior=json.loads(output.read_text('utf-8')) if output.exists() else {}
    previous=prior.get('runs',[]) if prior.get('source_sha256')==sha(Path(__file__)) else []
    done={r['record']:r for r in previous};rows=[]
    for p in sorted((ROOT/'results/baseline_v3/final_v3b').glob('*.json')):
        key=str(p.relative_to(ROOT));digest=sha(p)
        if key in done and done[key]['record_sha256']==digest:
            rows.append(done[key]);continue
        record=json.loads(p.read_text('utf-8'));trace=ROOT/record['trace']
        if record['trace_sha256'] is None:
            if record['success']:raise AssertionError('Successful missing trajectory')
            audit=dict(kind='recorded_planning_failure_without_trajectory',success=False,
                complete_time_s=None,error=record.get('error'))
        else:
            if sha(trace)!=record['trace_sha256']:raise ValueError('Trajectory changed')
            configure_runtime('v3',tolerance=record['tolerance_mm'])
            with np.load(trace) as data:
                reference=np.loadtxt(ROOT/'prospective/final_v3'/(record['path_id']+'.csv'),delimiter=',',skiprows=1)
                if not np.array_equal(data['reference'],reference):raise AssertionError('Registered reference mismatch: '+key)
                if not np.allclose(data['positions'][0],reference[0],atol=1e-10,rtol=0):raise AssertionError('Initial point mismatch: '+key)
                audit=metrics(data['reference'],data['positions'],dt=record['dt_s'],tolerance=record['tolerance_mm'])
            for name in ('success','complete_time_s','reason','geometry','rest_pass','kinematic_pass'):
                if audit.get(name)!=record.get(name):raise AssertionError('Acceptance mismatch: '+key+' '+name)
            if not np.allclose(audit['vector_peaks'],record['vector_peaks'],rtol=1e-12,atol=1e-12):
                raise AssertionError('Vector-peak mismatch: '+key)
        row=dict(record=key,record_sha256=digest,trace_sha256=record['trace_sha256'],audit=audit)
        rows.append(row)
        write_json(output,dict(created_utc=utc(),freeze_sha256=sha(frozen),source_sha256=sha(Path(__file__)),
            expected=30,observed=len(rows),complete_matrix=False,runs=rows))
        print('REAUDIT',record['method'],record['path_id'],record['dt_s'],audit['success'],flush=True)
    write_json(output,dict(created_utc=utc(),freeze_sha256=sha(frozen),source_sha256=sha(Path(__file__)),
        expected=30,observed=len(rows),complete_matrix=len(rows)==30,
        scope='Every full saved trajectory recomputed independently of planning; no failure omitted',runs=rows))
    if args.require_complete and len(rows)!=30:raise SystemExit('Final matrix incomplete')


if __name__=='__main__':main()
