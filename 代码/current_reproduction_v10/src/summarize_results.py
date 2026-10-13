"""All paper-facing new values come from retained machine-readable runs."""
import csv
import json
from collections import Counter
import numpy as np
from scipy import stats
from common import ROOT, manifest, write_json, sha
from paired_learning import SEEDS


def read(p):return json.loads(p.read_text('utf-8'))


def csv_write(path, rows):
    if not rows:return
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    out=ROOT/'results'/'summary';out.mkdir(parents=True,exist_ok=True)
    tests=[r['id'] for r in manifest()['paths'] if r['split']=='test']
    expected=[]
    for name in tests:
        for arm in ['adaptive','fixed']:
            for seed in SEEDS:
                expected.append(ROOT/'results'/'unseen'/f'fullrange_{arm}_{seed}_{name}_0.001.json')
            expected.append(ROOT/'results'/'unseen'/f'fullrange_{arm}_101_{name}_0.0005.json')
        for method in ['legacy_adaptive','legacy_fixed']:
            for dt in ['0.001','0.0005']:expected.append(ROOT/'results'/'unseen'/f'{method}_{name}_{dt}.json')
        for method,version in [('scmo','v2'),('s2','v3')]:
            for dt in ['0.001','0.0005']:expected.append(ROOT/'results'/'deterministic'/f'{method}_{version}_t1.25_v100.0_a2000.0_j62500.0_{name}_{dt}.json')
    grid=[dict(result=p.relative_to(ROOT).as_posix(),status='recorded' if p.exists() else 'missing') for p in expected]
    write_json(out/'expected_grid_ledger.json',dict(expected=len(grid),recorded=sum(p.exists() for p in expected),records=grid))
    missing=[str(p) for p in expected if not p.exists()]
    if missing:raise RuntimeError(f'{len(missing)}/{len(expected)} registered results missing. Full grid saved; paper statistics withheld.')
    all_rows=[];per_path=[];pairs=[];resolutions=[];costs=[]
    for name in tests:
        methods={}
        for arm in ['adaptive','fixed']:
            entries=[]
            for seed in SEEDS:
                p=ROOT/'results'/'unseen'/f'fullrange_{arm}_{seed}_{name}_0.001.json'
                if not p.exists():continue
                r=read(p);entries.append((seed,r));all_rows.append(dict(method=arm,seed=seed,path=name,dt_s=.001,success=r['success'],time_s=r['complete_time_s'],reason=r['reason'],error=r.get('error'),max_deviation_mm=r.get('max_polyline_deviation_mm'),trace_sha256=r.get('trace_sha256')))
            methods[arm]=dict(entries)
            values=np.array([r['complete_time_s'] for s,r in entries if r['success']],float);n=len(entries);k=len(values)
            if n:
                low=0. if k==0 else stats.beta.ppf(.025,k,n-k+1)
                high=1. if k==n else stats.beta.ppf(.975,k+1,n-k)
                per_path.append(dict(path=name,method=arm,n=n,successes=k,success_rate=k/n,success_exact95_low=low,success_exact95_high=high,
                    complete_time_mean_s=float(values.mean()) if k else None,complete_time_sd_s=float(values.std(ddof=1)) if k>1 else None,
                    complete_time_min_s=float(values.min()) if k else None,complete_time_max_s=float(values.max()) if k else None,
                    failures=dict(Counter(r['reason'] for s,r in entries if not r['success']))))
        paired=[]
        for s in SEEDS:
            left=methods['fixed'].get(s);right=methods['adaptive'].get(s)
            if left and right and left['success'] and right['success']:paired.append(left['complete_time_s']-right['complete_time_s'])
        n=len(paired)
        if n:
            mean=float(np.mean(paired));sd=float(np.std(paired,ddof=1)) if n>1 else None
            half=float(stats.t.ppf(.975,n-1)*sd/np.sqrt(n)) if n>=3 else None
            pairs.append(dict(path=name,n_complete_pairs=n,fixed_minus_adaptive_mean_s=mean,sd_s=sd,
                              t95_low_s=mean-half if half is not None else None,t95_high_s=mean+half if half is not None else None,
                              inference='Descriptive paired Student interval assuming approximately normal seed differences; n<=5, conditional on both succeeding.'))
        for method in ['adaptive','fixed','legacy_adaptive','legacy_fixed']:
            base=f'fullrange_{method}_101' if method in ['adaptive','fixed'] else method
            a=ROOT/'results'/'unseen'/f'{base}_{name}_0.001.json';b=ROOT/'results'/'unseen'/f'{base}_{name}_0.0005.json'
            if a.exists() and b.exists():
                ra,rb=read(a),read(b)
                resolutions.append(dict(path=name,method=method,seed=101 if method in ['adaptive','fixed'] else None,
                    success_1ms=ra['success'],success_half_ms=rb['success'],time_1ms_s=ra['complete_time_s'],time_half_ms_s=rb['complete_time_s'],
                    delta_half_minus_1ms_s=rb['complete_time_s']-ra['complete_time_s'] if ra['success'] and rb['success'] else None))
        for method in ['legacy_adaptive','legacy_fixed']:
            p=ROOT/'results'/'unseen'/f'{method}_{name}_0.001.json'
            if p.exists():
                r=read(p);all_rows.append(dict(method=method,seed=None,path=name,dt_s=.001,success=r['success'],time_s=r['complete_time_s'],reason=r['reason'],error=r.get('error'),max_deviation_mm=r.get('max_polyline_deviation_mm'),trace_sha256=r.get('trace_sha256')))
        for method in ['SCMO','S2']:
            version='v3' if method=='S2' else 'v2'
            p=ROOT/'results'/'deterministic'/f'{method.lower()}_{version}_t1.25_v100.0_a2000.0_j62500.0_{name}_0.001.json'
            if p.exists():
                r=read(p);all_rows.append(dict(method=method,seed=None,path=name,dt_s=.001,success=r['success'],time_s=r['complete_time_s'],reason=r.get('reason'),error=r.get('error'),max_deviation_mm=r.get('max_polyline_deviation_mm'),trace_sha256=r.get('trace_sha256')))
                half=ROOT/'results'/'deterministic'/f'{method.lower()}_{version}_t1.25_v100.0_a2000.0_j62500.0_{name}_0.0005.json'
                rb=read(half)
                resolutions.append(dict(path=name,method=method,seed=None,
                    success_1ms=r['success'],success_half_ms=rb['success'],time_1ms_s=r['complete_time_s'],time_half_ms_s=rb['complete_time_s'],
                    delta_half_minus_1ms_s=rb['complete_time_s']-r['complete_time_s'] if r['success'] and rb['success'] else None))
    for p in (ROOT/'results'/'paired').glob('*/seed_*/*/receipt.json'):
        r=read(p);reg=read(p.with_name('registration.json'));costs.append(dict(tag=p.parents[2].name,seed=reg['seed'],arm=reg['arm'],complete=r['complete'],**r['cost']))
    teacher_runs=[]
    for folder in (ROOT/'results').glob('teacher_selection*'):
        for p in folder.glob('*.json'):
            r=read(p)
            if 'actor_calls' in r:teacher_runs.append(dict(path=str(p.relative_to(ROOT)),actor_calls=r['actor_calls'],wall_s=r['wall_s'],success=r['success'],time_s=r.get('elapsed_prefix_s')))
    full_grid=[]
    for p in expected:
        r=read(p);stem=p.stem;name=next(n for n in tests if n in stem)
        prefix=stem.split('_'+name)[0]
        seed=None
        if prefix.startswith('fullrange_'):
            method=prefix.split('_')[1];seed=int(prefix.split('_')[2])
        else:method=prefix.split('_v')[0].upper() if prefix.startswith(('scmo','s2')) else prefix
        full_grid.append(dict(result=p.relative_to(ROOT).as_posix(),method=method,seed=seed,path=name,
            dt_s=float(stem.rsplit('_',1)[1]),success=r['success'],time_s=r['complete_time_s'],
            reason=r.get('reason'),error=r.get('error'),trace_sha256=r.get('trace_sha256')))
    summary=dict(test_paths=tests,independent_seeds=SEEDS,records=all_rows,all_grid_records=full_grid,per_path=per_path,paired_differences=pairs,resolution=resolutions,training_costs=costs,
        shared_teacher_cost=dict(runs=len(teacher_runs),actor_calls=sum(r['actor_calls'] for r in teacher_runs),wall_s=sum(r['wall_s'] for r in teacher_runs),
                                 accounting='Count all three pilot/repair rounds, including failed validation candidates and teacher traces. Shared once by both arms, not hidden inside the per-run hyperparameter_trials field. Historical actor_calls count completed decisions and exclude the final failed invocation; reported interaction total is a lower bound.',records=teacher_runs),
        scope='New fresh lower-budget matched comparison; historical fully optimized model separately labelled. No machine or cutting evidence.')
    write_json(out/'all_results.json',summary)
    csv_write(out/'all_test_runs.csv',full_grid);csv_write(out/'primary_test_runs.csv',all_rows);csv_write(out/'per_path_statistics.csv',per_path);csv_write(out/'paired_differences.csv',pairs);csv_write(out/'resolution.csv',resolutions);csv_write(out/'training_costs.csv',costs)
    print('Summary:',len(all_rows),'test records;',len(costs),'training runs')


if __name__=='__main__':main()
