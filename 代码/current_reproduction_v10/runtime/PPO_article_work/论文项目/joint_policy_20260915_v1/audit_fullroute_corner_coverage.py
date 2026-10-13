"""Recompute research guards with all endpoint samples included, without reruns."""
import copy
import numpy as np
from complete_corner_metrics import complete_corner_metrics
from multi_corner_raw_metrics import augment,guards,objective
from study_io import BASE,write_json,sha
from search_complete_return import read


def fresh(r):
    r=copy.deepcopy(r);r.pop('multi_corner_windows',None);r.pop('multi_corner_definition',None)
    return augment(r)


def main():
    stage=BASE/'canonical_gate_band_early_fullroute_joint_learning_20260920'
    data=read(stage/'baseline.json')['baseline']['records']
    old_reference={r['path']:r for r in data if r['path'] in (18,31)}
    reference={r['path']:fresh(r) for r in data if r['path'] in (18,31)}
    baseline_audit=[]
    for p,r in reference.items():
        with np.load(r['trace']) as z:rows=z['rows'].copy()
        windows=r['multi_corner_windows'];covered=np.zeros(len(rows),bool)
        for w in windows:covered[w['start_ms']:w['stop_ms']]=True
        assert covered.all() and min(guards(r,r))>=0.
        baseline_audit.append(dict(path=p,samples=len(rows),covered_samples=int(covered.sum()),
            windows=[dict(vertex=w['vertex'],start_ms=w['start_ms'],stop_ms=w['stop_ms']) for w in windows]))
    # Optional initial-position support must be exactly neutral on every old isolated metric.
    checked=0
    for r in data:
        if 'shape' not in r:continue
        with np.load(r['trace']) as z:ref,rows=z['reference'].copy(),z['rows'].copy()
        assert complete_corner_metrics(ref,rows)==complete_corner_metrics(ref,rows,initial_xy=ref[0])
        checked+=1
    candidates=[];result_file=stage/'state_joint/results.json'
    if result_file.exists():
        for c in read(result_file)['candidates']:
            multi=[fresh(r) for r in c['records'] if r['path'] in reference]
            values=np.concatenate([guards(r,reference[r['path']]) for r in multi])
            old_objective=sum(objective(r,old_reference[r['path']]) for r in c['records'] if r['path'] in reference)
            new_objective=sum(objective(r,reference[r['path']]) for r in multi)
            candidates.append(dict(tag=c['tag'],full_endpoint_guard_minimum=float(values.min()),
                multi_objective=float(new_objective),full_endpoint_total_score=float(c['score']-old_objective+new_objective),
                full_endpoint_pass=bool(values.min()>=-1e-8),
                original_guard_pass=bool(min(c['measured_inequalities'])>=-1e-8)))
    output=dict(baseline=baseline_audit,isolated_default_metrics_bitwise_equal=checked,
        candidates=candidates,physical_reruns=0,
        metric_code_sha256=sha(BASE/'multi_corner_raw_metrics.py'),
        note='The initial local-window metric omitted the first half incoming leg and final half outgoing leg. The running optimizer uses its registered snapshot. All adoption is on hold until COMPLETE saved candidate raw trajectories are re-scored with these endpoint-inclusive guards; this stricter audit does not relax any original constraint.')
    write_json(stage/'full_endpoint_coverage_audit.json',output)
    print('FULL_CORNER_COVERAGE',baseline_audit,'isolated_identity',checked,'candidates',len(candidates),flush=True)


if __name__=='__main__':main()
