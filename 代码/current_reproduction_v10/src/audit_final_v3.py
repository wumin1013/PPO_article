"""Recompute final learned results from complete saved command trajectories."""
import argparse
import json
import numpy as np
from common import ROOT,sha,write_json,utc


def main():
    p=argparse.ArgumentParser();p.add_argument('--recompute',action='store_true');args=p.parse_args()
    from metrics import metrics
    frozen=json.loads((ROOT/'audit/final_v3b_freeze.json').read_text('utf-8'))
    mismatches=[];rows=[];missing=[]
    for name,digest in frozen['sha256'].items():
        if sha(ROOT/name)!=digest:mismatches.append('input:'+name)
    for cell in frozen['grid']:
        stem=f"{cell['method']}_{cell['seed']}_{cell['path']}_{cell['dt_s']}"
        file=ROOT/'results/final_v3b'/(stem+'.json')
        if not file.exists():missing.append(stem);continue
        result=json.loads(file.read_text('utf-8'));trace=file.with_suffix('.npz')
        if not trace.exists() or sha(trace)!=result['trace_sha256']:mismatches.append('trace:'+stem);continue
        for key in ['path','method','seed','dt_s']:
            if result[key]!=cell[key]:mismatches.append('identity:'+stem+':'+key)
        if result['model_sha256']!=cell['sha256']:mismatches.append('model:'+stem)
        if not result['success'] and result['complete_time_s'] is not None:mismatches.append('failure_time:'+stem)
        with np.load(trace,allow_pickle=False) as data:
            ref=np.loadtxt(ROOT/'prospective/final_v3'/(cell['path']+'.csv'),delimiter=',',skiprows=1)
            if not np.array_equal(data['reference'],ref):mismatches.append('reference:'+stem)
            if not np.array_equal(data['positions'][0],ref[0]):mismatches.append('initial_position:'+stem)
            if args.recompute:
                measured=metrics(ref,data['positions'],cell['dt_s'],cell['tolerance_mm'],
                    declared_success=result.get('executor_reason')=='success' and not result.get('error'),reason=result.get('executor_reason'))
                for key,value in measured.items():
                    if result.get(key)!=value:mismatches.append('metric:'+stem+':'+key)
        rows.append(dict(path=cell['path'],method=cell['method'],seed=cell['seed'],dt_s=cell['dt_s'],
            success=result['success'],complete_time_s=result['complete_time_s'],reason=result['reason'],file=str(file.relative_to(ROOT))))
    write_json(ROOT/'audit/final_v3b_trace_audit.json',dict(created_utc=utc(),source_sha256=sha(__file__),
        recomputed_from_positions=args.recompute,expected=len(frozen['grid']),recorded=len(rows),
        missing=missing,mismatches=mismatches,valid=not missing and not mismatches,rows=rows))
    print('AUDIT',len(rows),'missing',len(missing),'mismatches',len(mismatches))
    if missing or mismatches:raise RuntimeError('Final trajectory audit incomplete/inconsistent')


if __name__=='__main__':main()
