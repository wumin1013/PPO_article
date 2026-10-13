"""Independent complete-trace review of the indexed distance statistics."""
import json
import time
from pathlib import Path
import numpy as np
from common import ROOT, sha, utc, write_json
from exact_distance_index_v5 import SegmentDistanceIndex
from metrics_exact_index_v5 import metrics
from runtime_physical_time_v5 import install

OUT=ROOT/'audit/distance_index_independent_v5'
FILES=['exact_distance_index_v5.py','metrics_exact_index_v5.py','metrics.py',
       'review_distance_index_v5.py','runtime_physical_time_v5.py','corridor_v3.py']


def exhaustive(reference,points):
    starts=reference[:-1];vectors=np.diff(reference,axis=0)
    lengths=np.linalg.norm(vectors,axis=1)
    result=[]
    for block in np.array_split(points,max(1,len(points)//2000)):
        delta=block[:,None,:]-starts[None,:,:]
        fraction=np.clip(np.sum(delta*vectors,axis=2)/lengths**2,0,1)
        result.extend(np.linalg.norm(delta-fraction[:,:,None]*vectors,axis=2).min(axis=1).tolist())
    return np.asarray(result)


def boundary_probe():
    """Implementation fixtures only, never research test programs."""
    rng=np.random.default_rng(902711)
    reference=np.array([[0.,0.],[10000.,0.],[10000.,.001],[-5000.,.002],[-5000.,5.],[0.,0.]])
    points=np.vstack([reference,.5*(reference[:-1]+reference[1:]),
        rng.uniform([-15000.,-10000.],[15000.,10000.],size=(1000,2)),
        [[5000.,1e-12],[5000.,-1e-12],[5000.,0.],[10000.,.0005],[1e7,1e7]]])
    exact=exhaustive(reference,points);indexed=SegmentDistanceIndex(reference).distances(points)
    return dict(scope='Numerical implementation fixture only; not a manuscript experiment',queries=len(points),
        all_distances_elementwise_exact=bool(np.array_equal(exact,indexed)),max_absolute_error_mm=float(np.max(abs(exact-indexed))))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    regfile=OUT/'registration.json'
    source_reg=ROOT/'results/historical_strong_v5/registration.json'
    prior=json.loads(source_reg.read_text('utf-8'))
    if not regfile.exists():
        write_json(regfile,dict(created_utc=utc(),scope='Read-only independent implementation audit; no new scientific trajectories',
            source_registration=str(source_reg),source_registration_sha256=sha(source_reg),
            cases=[dict(id=row['id'],trace_sha256=sha(ROOT/'results/historical_strong_v5'/f"{row['id']}_dt1000us.npz"),
                        metric_sha256=sha(ROOT/'results/historical_strong_v5'/f"{row['id']}_dt1000us.json")) for row in prior['cases']],
            source_sha256={name:sha(ROOT/'src'/name) for name in FILES},
            required_checks=['Every position in each of the thirteen canonical historical development traces versus the original exhaustive formula',
                'Every returned indexed metrics field versus the original persisted full audit, including complete geometry/gate/kinematic/rest checks',
                'Long and dense use the complete positions arrays; no sample thinning',
                'Long-segment, close-parallel, midpoint, endpoint, far-query and large-coordinate implementation fixtures'],
            spatial_proof='The nearest vertex distance D upper-bounds the nearest segment distance. For any point q on a segment, distance(p, midpoint) <= distance(p,q) + half_length. Thus any segment with distance(p,segment)<=D has its midpoint in the query ball D+max_half_length. cKDTree uses exact-query eps=0, with an outward roundoff margin. Exact projection and the original norm-squared denominator decide distance; tree distances never decide acceptance.',
            boundary='Only distance statistics are replaced; all corridor capsule tests, GateTracker updates, tolerances, samples and success rules are unchanged.'))
    reg=json.loads(regfile.read_text('utf-8'))
    for name,expected in reg['source_sha256'].items():
        if sha(ROOT/'src'/name)!=expected:raise ValueError('Review source changed: '+name)
    results=[]
    for row in prior['cases']:
        name=row['id'];receipt=OUT/(name+'.json')
        if receipt.exists():
            results.append(json.loads(receipt.read_text('utf-8')));continue
        path=ROOT/'results/historical_strong_v5'/f'{name}_dt1000us.npz'
        saved=json.loads(path.with_suffix('.json').read_text('utf-8'))
        with np.load(path) as data:
            reference=data['reference'];positions=data['positions']
        started=time.perf_counter();indexed=SegmentDistanceIndex(reference).distances(positions)
        indexed_distance_s=time.perf_counter()-started
        started=time.perf_counter();full=exhaustive(reference,positions)
        exhaustive_distance_s=time.perf_counter()-started
        equal=bool(np.array_equal(indexed,full))
        install(dt=.001,tolerance=row['evaluation_tolerance_mm'])
        started=time.perf_counter();new=metrics(reference,positions,dt=.001,tolerance=row['evaluation_tolerance_mm'],
            declared_success=saved['success'],reason=saved.get('executor_reason'))
        indexed_full_audit_s=time.perf_counter()-started
        fields={key:bool(new[key]==saved.get(key)) for key in new}
        result=dict(path_id=name,positions=len(positions),reference_segments=len(reference)-1,
            source_trace_sha256=sha(path),source_metric_sha256=sha(path.with_suffix('.json')),
            every_position_distance_elementwise_exact=equal,
            max_distance_absolute_error_mm=float(np.max(abs(indexed-full))),
            every_metric_field_exact=all(fields.values()),field_exact=fields,
            indexed_metric_result=new,indexed_distance_wall_s=indexed_distance_s,
            exhaustive_distance_wall_s=exhaustive_distance_s,indexed_complete_audit_wall_s=indexed_full_audit_s,
            passed=equal and all(fields.values()),registration_sha256=sha(regfile))
        write_json(receipt,result);results.append(result)
        print(json.dumps(dict(case=name,positions=len(positions),passed=result['passed'],
            pointwise_error=result['max_distance_absolute_error_mm'],indexed_distance_s=indexed_distance_s,
            exhaustive_distance_s=exhaustive_distance_s,full_indexed_audit_s=indexed_full_audit_s)),flush=True)
    fixture=boundary_probe()
    unchanged=all(sha(ROOT/'src'/name)==expected for name,expected in reg['source_sha256'].items())
    report=dict(created_utc=utc(),registration_sha256=sha(regfile),case_count=len(results),
        total_position_queries=sum(r['positions'] for r in results),
        passed=len(results)==13 and all(r['passed'] for r in results) and fixture['all_distances_elementwise_exact'] and unchanged,
        source_files_unchanged=unchanged,fixtures=fixture,cases=results,
        conclusion='Use the separate metrics_exact_index_v5 only when this report passes. Preserve the original metrics.py and all completed old audits unchanged.')
    write_json(ROOT/'audit/distance_index_independent_review_v5.json',report)
    print(json.dumps({key:report[key] for key in ['passed','case_count','total_position_queries','source_files_unchanged','fixtures']},indent=2))


if __name__=='__main__':main()
