"""Assemble immutable final evidence views from explicitly named source cohorts.

No source evidence is changed. Existing canonical outputs are accepted only if
the regenerated content is identical; new evidence needs a new output file.
"""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
BASE=Path(r'D:\PPO_article_work\JNN_dominance_20261011_v5')
CASES=('original','dense','piecewise_curvature_coupon_scale10','asymmetric_neck_contour_scale10')
METHODS=('JNN','FIXED','S2','S2R','SCMO','COPP','NNC')
DTS=(.001,.0005)
GENERALIZATION_CASES=('hilbert_cell_sweep','harmonic_radial_contour','piecewise_curvature_coupon',
                     'asymmetric_neck_contour','annular_sector_raster')
MAIN_SHA='c16d45e38bab022ce8f289148aac4f3dcb3559f98b18bdfa580e23aea4a99764'
FIXED_SHA='40af7ea30bec92d9cd42e060b2d71ee72919c4cc50c3b28281ff036a0790f469'
METRIC_KEYS=('success','reason','complete_time_s','elapsed_prefix_s','samples','dt_s','tolerance_mm',
             'limits','vector_peaks','vector_limit_ratios','kinematic_pass','rest_pass','terminal_rest_s',
             'geometry','max_polyline_deviation_mm','rms_polyline_deviation_mm','executed_length_mm',
             'source','numerical_scope')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text('utf-8'))


def write_immutable(path,value):
    text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if path.exists():
        if path.read_text('utf-8')!=text:
            raise FileExistsError('Refuse to replace frozen canonical evidence: '+str(path))
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)


class Files:
    def __init__(self):self.data={}
    def add(self,path,expected=None,role='support'):
        if path is None:return None
        path=Path(path).resolve();key=str(path)
        if key not in self.data:
            if not path.is_file():raise FileNotFoundError(path)
            self.data[key]=dict(sha256=sha(path),bytes=path.stat().st_size,roles=[])
        row=self.data[key]
        if expected is not None and expected!=row['sha256']:
            raise ValueError('SHA mismatch: '+key)
        if role not in row['roles']:row['roles'].append(role)
        return key
    def output(self):
        return {path:dict(sha256=value['sha256'],bytes=value['bytes'],roles=sorted(value['roles']))
                for path,value in sorted(self.data.items())}


def source_map(files,path,mapping,relative_base=None):
    for name,digest in mapping.items():
        source=Path(name)
        if not source.is_absolute():source=(relative_base or Path(path).parent)/source
        files.add(source,digest,'registered_source')


def registered_support(files,path,*,relative_base=None):
    path=Path(path);files.add(path,role='registration');reg=read(path)
    source_map(files,path,reg.get('source_sha256',{}),relative_base)
    for item in reg.get('inputs',[]) if isinstance(reg.get('inputs',[]),list) else []:
        for field in ('record','trace'):
            if item.get(field):files.add(item[field],item.get(field+'_sha256'),'registered_input_'+field)
    if isinstance(reg.get('inputs'),dict):
        for relative,item in reg['inputs'].items():
            files.add(path.parent/relative,item['sha256'],'registered_input')
    return reg


def add_record_dependencies(files,record):
    for key in ('source_record','source_planning_trace','source_trace'):
        if record.get(key):files.add(record[key],record.get(key+'_sha256'),'predecessor_'+key)
    for candidate in record.get('candidates',[]):
        if candidate.get('trace'):files.add(candidate['trace'],candidate.get('trace_sha256'),'retained_candidate_prefix')
    candidate=record.get('complete_candidate_audit')
    if candidate and candidate.get('trace'):
        files.add(candidate['trace'],candidate.get('trace_sha256'),'complete_candidate_trace')


def normalized_metrics(source,dt,tolerance):
    result={key:source[key] for key in METRIC_KEYS if key in source}
    result.update(dt_s=dt,tolerance_mm=source.get('tolerance_mm',tolerance),
                  limits=source.get('limits',[100.,2000.,62500.]))
    if bool(result['success'])!=(result['complete_time_s'] is not None):
        raise ValueError('Success/null completion mismatch')
    return result


def arithmetic(row,definition):
    metric=row['metrics'];dt=row['dt_s']
    if row['trace'] is None:
        assert not metric['success'] and metric['complete_time_s'] is None
        return dict(trace_available=False,scope='Failed planner retains null time; no invented commands.')
    with np.load(row['trace'],allow_pickle=False) as archive:
        positions=archive['positions'];reference=archive['reference']
    if definition and definition.get('reference'):
        expected=np.loadtxt(definition['reference'],delimiter=',',skiprows=1)
        assert np.array_equal(reference,expected),row['id']
    assert np.isfinite(positions).all() and np.array_equal(positions[0],reference[0]),row['id']
    assert positions.ndim==2 and positions.shape[1]==2 and len(positions)>=2,row['id']
    v=np.diff(positions,axis=0)/dt
    a=np.diff(np.vstack((np.zeros(2),v)),axis=0)/dt
    j=np.diff(np.vstack((np.zeros(2),a)),axis=0)/dt
    vn,an,jn=(np.linalg.norm(x,axis=1) for x in (v,a,j))
    peaks=[float(x.max()) for x in (vn,an,jn)]
    elapsed=(len(positions)-1)*dt
    nonrest=np.flatnonzero((vn>.001)|(an>.01))
    rest_s=(len(vn)-(int(nonrest[-1])+1 if len(nonrest) else 0))*dt
    length=float(vn.sum()*dt)
    assert np.allclose(peaks,metric['vector_peaks'],rtol=1e-10,atol=1e-8),row['id']
    assert abs(elapsed-metric['elapsed_prefix_s'])<1e-8,row['id']
    assert len(positions)==metric['samples'],row['id']
    assert abs(rest_s-metric['terminal_rest_s'])<1e-9,row['id']
    assert abs(length-metric['executed_length_mm'])<1e-7,row['id']
    if metric['success']:
        assert abs(elapsed-metric['complete_time_s'])<1e-8,row['id']
        assert max(np.asarray(peaks)/metric['limits'])<=1+2e-5,row['id']
        assert rest_s>=.02-1e-10 and metric['geometry']['geometry_complete'],row['id']
    return dict(trace_available=True,samples=len(positions),elapsed_s=elapsed,
        vector_peaks=peaks,terminal_rest_s=rest_s,executed_length_mm=length,
        all_recorded_arithmetic_reproduced=True,
        geometry_scope='Complete interval/gate certificate inherited only from the SHA-bound accepted record; arithmetic independently regenerated from every position sample.')


def case_definitions(inventory,files):
    definitions={name:dict(inventory['cases'][name]) for name in CASES}
    for name in ('original','dense'):
        folder=BASE/'baseline_records/supplementary_reproduction/cases'/name
        for key in ('reference','geometry'):definitions[name][key]=str(folder/Path(definitions[name][key]).name)
    for name,definition in definitions.items():
        for key in ('reference','geometry'):
            definition[key]=files.add(definition[key],definition[key+'_sha256'],'program_'+key)
        definition.update(independent_test=False,statistical_unit='One complete deterministic path execution; resolutions are numerical checks, not independent seeds.')
    return definitions


def refined_rows(summary_file,definitions,files):
    summary_file=Path(summary_file);summary=read(summary_file)
    files.add(summary_file,role='refined_summary')
    registration=summary_file.with_name(summary_file.name.replace('_summary.json','_registration.json'))
    files.add(registration,summary['registration_sha256'],'refinement_registration')
    registered_support(files,registration)
    rows=[]
    for source in summary['rows']:
        recordfile=files.add(source['record'],source['record_sha256'],'final_record')
        tracefile=files.add(source['trace'],source['trace_sha256'],'final_trace')
        record=read(recordfile);add_record_dependencies(files,record)
        assert record['trace_sha256']==source['trace_sha256']
        assert record['model_sha256']==(MAIN_SHA if source['method']=='JNN' else FIXED_SHA)
        planning=read(record['source_record'])
        case=source['case'];dt=source['dt_s'];definition=definitions.get(case)
        tolerance=definition['evaluation_tolerance_mm'] if definition else record['tolerance_mm']
        metrics=normalized_metrics(record,dt,tolerance)
        for key in METRIC_KEYS:
            if key in source['metrics'] and key in metrics:assert source['metrics'][key]==metrics[key],(case,key)
        identity=dict(method_code=source['method'],model_sha256=record['model_sha256'],
            planning_runtime=planning.get('runtime_version'),final_runtime=record.get('runtime_version'),
            pipeline=record['refinement']['version'],
            command_stage='Offline boundary refinement after frozen deterministic policy command planning',
            new_training_updates=record.get('new_training_updates',0),
            fixed_preview_mm=planning.get('fixed_preview_mm'),
            planning_observations_are_refined_execution_observations=False,
            initial_planning_success=record['refinement']['original_planning_success'],
            budget_paired_causal_ablation=False)
        row=dict(id=f"{case}_{source['method']}_dt{round(dt*1e6)}us",case=case,
            method=source['method'],dt_s=dt,trace=tracefile,trace_sha256=source['trace_sha256'],
            record=recordfile,record_sha256=source['record_sha256'],metrics=metrics,identity=identity,
            provenance=dict(summary=str(summary_file.resolve()),summary_sha256=sha(summary_file),
                registration=str(registration.resolve()),registration_sha256=sha(registration),
                source_record=record['source_record'],source_record_sha256=record['source_record_sha256'],
                source_planning_trace=record['source_planning_trace'],source_planning_trace_sha256=record['source_planning_trace_sha256'],
                complete_geometry_audit='Independent full final-command audit in bound source record'),
            cost=record.get('cost'),cohort='previously_observed_development_or_direct_variant',independent_test=False)
        row['arithmetic_audit']=arithmetic(row,definition)
        rows.append(row)
    return rows


def traditional_rows(inventory,definitions,files):
    path=ROOT/'boundary_baseline_check/summary.json';summary=read(path)
    files.add(path,role='traditional_refinement_summary')
    registration=path.with_name('registration.json')
    files.add(registration,summary['registration_sha256'],'traditional_refinement_registration')
    registered_support(files,registration)
    prior={row['run_id']:row for row in inventory['rows']}
    rows=[]
    assert summary['completed_check_count']==32
    for source in summary['rows']:
        original=prior[source['run_id']]
        recordfile=files.add(source['record'],source['record_sha256'],'final_record')
        record=read(recordfile);add_record_dependencies(files,record)
        tracefile=files.add(source['selected_trace'],source['selected_trace_sha256'],'final_trace')
        files.add(original['record'],original['record_sha256'],'original_baseline_record')
        source_identity=original.get('identity')
        identity=dict(method_code=source['method'],planning_identity=source_identity,
            pipeline='Unchanged traditional planner followed by common bounded startup refinement',
            common_startup_check=True,startup_compression_adopted=source['adopted'],
            startup_saved_s=source['saved_s'],terminal_commands_changed=False,
            published_method_reproduction_claim='Preserve exact original adapter identity; no new method-reproduction claim')
        row=dict(id=source['run_id'],case=source['case'],method=source['method'],dt_s=source['dt_s'],
            trace=tracefile,trace_sha256=source['selected_trace_sha256'],record=recordfile,
            record_sha256=source['record_sha256'],
            metrics=normalized_metrics(record['metrics'],source['dt_s'],definitions[source['case']]['evaluation_tolerance_mm']),
            identity=identity,provenance=dict(summary=str(path),summary_sha256=sha(path),
                registration=str(registration),registration_sha256=sha(registration),
                original_record=original['record'],original_record_sha256=original['record_sha256'],
                original_trace=original['trace'],original_trace_sha256=original['trace_sha256'],
                complete_geometry_audit=record['acceptance_origin']),
            cost=dict(startup_refinement_and_audit_wall_s=record['wall_s'],
                      original_planning_cost_source=original['record']),
            cohort='previously_observed_development_or_direct_variant',independent_test=False)
        row['arithmetic_audit']=arithmetic(row,definitions[source['case']])
        rows.append(row)
    return rows


def nnc_rows(inventory,definitions,files):
    rows=[]
    for source in inventory['rows']:
        if source['method']!='NNC':continue
        recordfile=files.add(source['record'],source['record_sha256'],'protected_nnc_record')
        tracefile=files.add(source['trace'],source['trace_sha256'],'protected_nnc_failed_prefix')
        record=read(recordfile)
        assert record['original_project_caps_preserved'] is True
        assert not record['original_enable_kcm']
        identity=dict(method_code='NNC',runtime=record['runtime_version'],model_sha256=record['model_sha256'],
            description=record['source_identity'],original_project_caps_preserved=True,
            original_enable_kcm=False,policy_dt_s=record['policy_dt_s'],
            published_method_reproduction=False,
            startup_refinement='No successful complete command to refine; original failed prefix retained')
        row=dict(id=source['run_id'],case=source['case'],method='NNC',dt_s=source['dt_s'],
            trace=tracefile,trace_sha256=source['trace_sha256'],record=recordfile,record_sha256=source['record_sha256'],
            metrics=normalized_metrics(record,source['dt_s'],definitions[source['case']]['evaluation_tolerance_mm']),
            identity=identity,provenance=dict(inventory=str(ROOT/'audit/baseline_inventory.json'),
                inventory_sha256=sha(ROOT/'audit/baseline_inventory.json'),metric_authority=source['metric_authority'],
                complete_geometry_audit=source['geometry_audit']),
            cost={k:record.get(k) for k in ('wall_s','process_cpu_s','actor_calls')},
            cohort='previously_observed_development_or_direct_variant',independent_test=False)
        row['arithmetic_audit']=arithmetic(row,definitions[source['case']])
        rows.append(row)
    assert len(rows)==8
    return rows


def grid_and_order(rows,cases,methods):
    expected=set(itertools.product(cases,methods,DTS))
    keys=[(r['case'],r['method'],r['dt_s']) for r in rows]
    assert len(keys)==len(set(keys)) and set(keys)==expected,'Incomplete or repeated grid'
    assert len({row['id'] for row in rows})==len(rows)
    ordered=sorted(rows,key=lambda r:(cases.index(r['case']),methods.index(r['method']),DTS.index(r['dt_s'])))
    rankings=[]
    for case,dt in itertools.product(cases,DTS):
        subset=[r for r in ordered if r['case']==case and r['dt_s']==dt]
        complete=[r for r in subset if r['metrics']['success']]
        failed=[r for r in subset if not r['metrics']['success']]
        complete.sort(key=lambda r:(round(r['metrics']['complete_time_s']/dt),methods.index(r['method'])))
        ranks=[];previous_ticks=None;rank=0
        for n,row in enumerate(complete,1):
            ticks=round(row['metrics']['complete_time_s']/dt)
            if ticks!=previous_ticks:rank=n
            ranks.append(dict(rank=rank,id=row['id'],method=row['method'],
                              complete_time_s=row['metrics']['complete_time_s'],ticks=ticks))
            previous_ticks=ticks
        rankings.append(dict(case=case,dt_s=dt,successful_order=ranks,
            failed_ids=[r['id'] for r in failed],
            failures_excluded_from_time_ranking=True))
    return ordered,dict(expected_count=len(expected),actual_count=len(rows),unique_ids=True,complete_grid=True,
        success_count=sum(r['metrics']['success'] for r in rows),failure_count=sum(not r['metrics']['success'] for r in rows),
        failures_have_null_completion_time=all(r['metrics']['complete_time_s'] is None for r in rows if not r['metrics']['success'])),rankings


def common_support(files,inventory):
    files.add(Path(__file__),role='canonical_assembler')
    files.add(ROOT/'audit/baseline_inventory.json',role='frozen_input_inventory')
    for path,digest in inventory['source_receipts'].items():files.add(path,digest,'prior_source_receipt')
    frozen=BASE/'evidence/frozen_main_v6/manifest.json'
    # The complete frozen archive index remains evidence, including failures.
    for relative,digest in read(frozen)['files'].items():files.add(frozen.parent/relative,digest,'frozen_support_archive')
    runtime_manifest=BASE/'runtime/source_manifest.json'
    files.add(runtime_manifest,role='frozen_runtime_manifest')
    for item in read(runtime_manifest):files.add(BASE/'runtime'/item['relative'],item['sha256'],'frozen_runtime_source')
    reg=BASE/'results/representative_scaled_v8/learned/registration.json'
    registered_support(files,reg,relative_base=BASE/'src')
    unified=ROOT/'unified_v5_inputs/registration.json'
    if unified.exists():
        registered_support(files,unified,relative_base=BASE)
        files.add(ROOT/'unified_v5_inputs/run_unified.py',read(unified)['runner_sha256'],'unified_planning_generator')
    files.add(BASE/'baseline_records/supplementary_reproduction/models/main.pt',MAIN_SHA,'final_main_model')
    files.add(BASE/'evidence/frozen_main_v6/fixed_two.pt',FIXED_SHA,'fixed_comparator_model')


def build_generalization():
    path=ROOT/'boundary_refinement/generalization_summary.json'
    if not path.exists():return
    files=Files();files.add(Path(__file__),role='canonical_assembler')
    rows=refined_rows(path,{},files)
    rows,grid,rankings=grid_and_order(rows,GENERALIZATION_CASES,('JNN',))
    output=dict(schema_version=1,scope='Five previously observed complete synthetic programs; independent of the four-case comparison grid and not independent final testing.',
        grid=dict(cases=list(GENERALIZATION_CASES),methods=['JNN'],dt_s=list(DTS)),
        rows=rows,files=files.output(),grid_audit=grid,time_ordering=rankings,
        statistical_unit='One frozen-model full program execution; resolutions are numerical checks; zero new training seeds.',
        source_receipts={str(path):sha(path)})
    write_immutable(HERE/'generalization_comparison.json',output)
    print('GENERALIZATION',grid,flush=True)


def build_main():
    inventory=read(ROOT/'audit/baseline_inventory.json');files=Files()
    common_support(files,inventory)
    definitions=case_definitions(inventory,files)
    rows=traditional_rows(inventory,definitions,files)+nnc_rows(inventory,definitions,files)
    for name in ('scaled','original'):
        rows.extend(refined_rows(ROOT/f'boundary_refinement/{name}_summary.json',definitions,files))
    rows,grid,rankings=grid_and_order(rows,CASES,METHODS)
    sources=[ROOT/'audit/baseline_inventory.json',ROOT/'boundary_baseline_check/summary.json',
             ROOT/'boundary_refinement/scaled_summary.json',ROOT/'boundary_refinement/original_summary.json']
    output=dict(schema_version=1,scope='Final complete-command comparison on four previously observed development programs or direct variants.',
        grid=dict(cases=list(CASES),methods=list(METHODS),dt_s=list(DTS),expected_cells=56),
        cases=definitions,methods=list(METHODS),rows=rows,files=files.output(),
        source_receipts={str(p):sha(p) for p in sources},grid_audit=grid,time_ordering=rankings,
        main_model=str(BASE/'baseline_records/supplementary_reproduction/models/main.pt'),model_sha256=MAIN_SHA,
        statistical_unit='One full deterministic path execution. Two resolutions are numerical checks; no new independent training-seed evidence.',
        evidence_contract='Initial rest; complete ordered traversal; every command interval in source-specific one-sided corridor; planar backward-difference velocity/acceleration/jerk limits; qualified final rest. Failures remain null and never receive speedups.',
        file_manifest_scope='All final records and command traces, declared predecessor traces/records, all retained traditional startup candidates, program definitions, registrations, source receipts, immutable runtime archive and model checkpoints. File roles distinguish final evidence from retained supporting history.')
    write_immutable(HERE/'final_comparison.json',output)
    compatible=dict(schema_version=1,scope=output['scope'],main_model=output['main_model'],model_sha256=MAIN_SHA,
        nominal_tolerance_mm=1.25,five_seed_training=False,
        runs=[r for r in rows if r['case'] in ('original','dense')],
        files={path:item['sha256'] for path,item in output['files'].items()},
        canonical_source=str(HERE/'final_comparison.json'),canonical_sha256=sha(HERE/'final_comparison.json'),
        cases={k:v for k,v in definitions.items() if k in ('original','dense')})
    write_immutable(HERE/'plot_manifest.json',compatible)
    print('FINAL',grid,'files',len(output['files']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-seconds',type=float,default=0)
    args=parser.parse_args()
    build_generalization()
    awaited=ROOT/'boundary_refinement/original_summary.json';started=time.monotonic()
    if not awaited.exists():print('WAITING for original_summary.json; no partial final file is published.',flush=True)
    while not awaited.exists() and time.monotonic()-started<args.wait_seconds:time.sleep(5)
    if not awaited.exists():raise SystemExit(2)
    build_main()
