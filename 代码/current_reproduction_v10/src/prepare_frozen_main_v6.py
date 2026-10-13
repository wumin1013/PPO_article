"""Import complete original-model evidence into a portable manuscript dataset."""
import json
import shutil
from pathlib import Path

import numpy as np

from common import ROOT, sha, utc, write_json
from main_evidence_v5 import verify, MODEL_SHA

OUT=ROOT/'evidence/frozen_main_v6'
SHARED=ROOT/'shared_manuscript/frozen_main_v6'


def read(path):return json.loads(Path(path).read_text('utf-8'))


def copy_checked(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        assert sha(destination)==sha(source),str(destination)
    else:shutil.copy2(source,destination)


def main():
    manifest,bound=verify()
    if (OUT/'manifest.json').exists():raise FileExistsError('Preserve imported evidence; use the generator to verify it')
    SHARED.mkdir(parents=True,exist_ok=True)
    rows=[];sources={}
    for case in ('original','dense'):
        for name in ('reference.csv','geometry.json'):
            src=ROOT/f'baseline_records/supplementary_reproduction/cases/{case}/{name}'
            copy_checked(src,OUT/'programs'/case/name)
        for method in ('JNN','S2','S2R','SCMO','COPP'):
            for dt in (.001,.0005):
                token=f'{case}_{method}_dt{round(dt*1e6)}us'
                if method=='COPP':
                    record=ROOT/f'results/copp_v5d_complete_development/{case}.json'
                    outer=read(record);iteration=next(x for x in outer['iterations'] if x['iteration']==3)
                    run=next(x for x in iteration['runs'] if x['dt_s']==dt)
                    metrics=run['metrics'];trace=Path(run['trace']);expected=run['trace_sha256']
                    identity=dict(method='Official COPP 0.2.3 TOPP3-SOCP with project geometry and vector adaptation',
                        optimizer_iteration=3,projection_candidate=iteration['selected_projection_candidate'],
                        commit=outer['planner']['upstream_commit'],settings=outer['planner']['settings'])
                else:
                    if dt==.001:
                        record=ROOT/manifest['runs'][token]['record']
                    elif method=='JNN':
                        record=(Path('D:/PPO_article_work/JNN_efficiency_20261011_v4/results/strong_checkpoint_v4/v4e_resolution_control/original_dt500us.json')
                            if case=='original' else ROOT/'results/historical_strong_v5/dense_dt500us.json')
                    else:
                        record=Path(f'D:/PPO_article_work/JNN_revision_20261011_v3/results/baseline_v3/development_unified/{method}_{case}_dt500us_scale1000.json')
                    metrics=read(record);trace=record.with_suffix('.npz');expected=metrics['trace_sha256']
                    identity=dict(method=method,runtime=metrics.get('runtime_version','baseline_development_unified_v3'),
                        model_sha256=metrics.get('model_sha256'),corner_scale=1. if method!='JNN' else None)
                    if method=='JNN':assert metrics['model_sha256']==MODEL_SHA
                assert trace.exists() and sha(trace)==expected,(token,trace)
                assert metrics['success'] and metrics['kinematic_pass'] and metrics['rest_pass']
                assert metrics['geometry']['geometry_complete'] and metrics['geometry']['corridor_inside']
                assert metrics['dt_s']==dt and metrics['limits']==[100.,2000.,62500.]
                with np.load(trace) as data:
                    reference=data['reference'];positions=data['positions']
                assert np.array_equal(reference,np.loadtxt(OUT/'programs'/case/'reference.csv',delimiter=',',skiprows=1))
                assert abs((len(positions)-1)*dt-metrics['complete_time_s'])<1e-9
                destination=OUT/'runs'/f'{token}.npz';copy_checked(trace,destination)
                record_destination=OUT/'source_records'/f'{token}.json';copy_checked(record,record_destination)
                portable=dict(id=token,case=case,method=method,dt_s=dt,identity=identity,
                    trace='runs/'+destination.name,trace_sha256=sha(destination),
                    source_record='source_records/'+record_destination.name,source_record_sha256=sha(record_destination),
                    original_trace=str(trace),original_record=str(record),metrics=metrics)
                rows.append(portable)
    for name in ('main.pt',):copy_checked(ROOT/'baseline_records/supplementary_reproduction/models'/name,OUT/name)
    copy_checked(ROOT/'shared_manuscript/main_model_v5/figure8_main_v5.pdf',SHARED/'figure8.pdf')
    copy_checked(ROOT/'shared_manuscript/main_model_v5/figure8_main_v5.png',SHARED/'figure8.png')
    for name in ('Figure_2.pdf','Figure_3.pdf','elsarticle.cls','elsarticle-num.bst','references.bib','LICENSES.txt'):
        copy_checked(ROOT/'shared_manuscript'/name,SHARED/name)
    for path in OUT.rglob('*'):
        if path.is_file():sources[path.relative_to(OUT).as_posix()]=sha(path)
    write_json(OUT/'manifest.json',dict(created_utc=utc(),scope='Two original complete simulated development programs; one frozen learned controller',
        model_sha256=MODEL_SHA,main_model='main.pt',five_seed_training=False,
        statistical_unit='One complete path execution; deterministic replay of one frozen policy or planner',
        nominal_tolerance_mm=1.25,runs=rows,files=sources,
        figure8_sha256=sha(SHARED/'figure8.pdf'),publication_ready=False,
        pending='Full-source integration, comparator inventory and registered additional-program evaluation'))
    print('IMPORTED_COMPLETE_RUNS',len(rows),'MODEL',MODEL_SHA)


if __name__=='__main__':main()
