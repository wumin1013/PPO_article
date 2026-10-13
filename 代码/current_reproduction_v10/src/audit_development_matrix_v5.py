"""Index observed development evidence without running or selecting new configurations."""
import csv
import json
from pathlib import Path
import numpy as np
from common import ROOT, sha, utc, write_json

V3 = Path(r'D:\PPO_article_work\JNN_revision_20261011_v3')
V4 = Path(r'D:\PPO_article_work\JNN_efficiency_20261011_v4')
BASELINES = ['S2','S2R','SCMO','COPP_v5b','COPP_v5c','COPP_v5d']
METHODS = BASELINES + ['historical_strong','fresh_physical_fit_seed101',
                      'recovery_fit_v6_adaptive','recovery_fit_v6_fixed',
                      'predictive_v5_1','predictive_v5_2','recoverable_v5_3','resumption_v5_4']
ORDER = ['original','dense','train_box','train_sine','train_bends','train_bezier',
         'validation_trapezoid','validation_zigzag','final_dense_fold','final_chamfer_chain',
         'final_rounded_triangle','final_quintic_double_bend','final_inward_channels']


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def normalized(path, record, method, case, rows, trace=None, declared_hash=None, primary=True, configuration=None):
    row = rows[case]
    trace = trace or path.with_suffix('.npz')
    trace_exists = trace.exists()
    expected_hash = declared_hash or record.get('trace_sha256')
    actual_hash = sha(trace) if trace_exists else None
    reference_match = None
    if trace_exists:
        with np.load(trace) as data:
            if 'reference' in data:
                reference_match = bool(np.array_equal(data['reference'], np.loadtxt(ROOT / row['reference'], delimiter=',', skiprows=1)))
    geometry = record.get('geometry') or {}
    checks = dict(reference_array_matches=reference_match,
        trace_hash_matches=bool(trace_exists and expected_hash and actual_hash == expected_hash),
        tolerance_matches=abs(record.get('tolerance_mm', -1)-row['evaluation_tolerance_mm']) < 1e-10,
        planar_limits_match=record.get('limits') == [100.,2000.,62500.],
        kinematic_pass=bool(record.get('kinematic_pass')), rest_pass=bool(record.get('rest_pass')),
        every_command_interval_inside=bool(geometry.get('corridor_inside')),
        all_gates_in_order=bool(geometry.get('ordered_complete')),
        geometry_complete=bool(geometry.get('geometry_complete')),
        initial_point_matches=geometry.get('start_error_mm') is not None and geometry['start_error_mm'] <= 1e-9)
    accepted = bool(record.get('success')) and all(checks.values())
    status = 'PASS' if accepted else ('FAIL' if record.get('success') is False else 'INCOMPATIBLE_OR_INCOMPLETE_EVIDENCE')
    if record.get('success') and not accepted:
        status = 'INCOMPATIBLE_OR_INCOMPLETE_EVIDENCE'
    return dict(path_id=case, method=method, dt_s=record.get('dt_s'), status=status,
        complete_time_s=record.get('complete_time_s') if accepted else None,
        elapsed_prefix_s=record.get('elapsed_prefix_s'), reason=record.get('reason'), error=record.get('error'),
        primary_configuration=primary, configuration=configuration,
        record=str(path), record_sha256=sha(path), trace=str(trace) if trace_exists else None,
        declared_trace_sha256=expected_hash, actual_trace_sha256=actual_hash,
        reference=str(ROOT / row['reference']), reference_sha256=row['reference_sha256'],
        checks=checks, vector_peaks=record.get('vector_peaks'), tolerance_mm=record.get('tolerance_mm'),
        gate_counts=[geometry.get('gates_passed'),geometry.get('gate_count')],
        cost=record.get('cost',{'total_wall_s':record.get('wall_s')}),
        scope='DEVELOPMENT ONLY; record identity retained; no successful-only pooling')


def collect(rows):
    records=[]
    for folder in ['development_unified','final_v3b','validation_final']:
        for path in sorted((V3/'results/baseline_v3'/folder).glob('*.json')):
            r=read(path);case=r.get('path_id');method=r.get('method')
            if case not in rows or method not in METHODS[:3]:continue
            records.append(normalized(path,r,method,case,rows,
                primary=r.get('corner_scale')==1., configuration={'version':folder,'corner_scale':r.get('corner_scale')}))
    for case in ['validation_trapezoid','validation_zigzag']:
        path=V3/'results/deterministic'/f'scmo_v2_t1.25_v100.0_a2000.0_j62500.0_{case}_0.001.json'
        records.append(normalized(path,read(path),'SCMO',case,rows,
            configuration={'version':'independent SCMO 1.1-finite-difference-rest'}))
    copp_folders = [('COPP_v4e', V4/'results/copp_development_v4e'),
        ('COPP_v5b', ROOT/'results/copp_v5b_complete_development'),
        ('COPP_v5c', ROOT/'results/copp_v5c_complete_development'),
        ('COPP_v5c', ROOT/'results/copp_v5c_observed_additional'),
        ('COPP_v5d', ROOT/'results/copp_v5d_complete_development')]
    for method, folder in copp_folders:
        for path in sorted(folder.glob('*.json')):
            r=read(path)
            if not isinstance(r,dict):continue
            case=r.get('path_id')
            if case not in rows or 'iterations' not in r:continue
            for candidate in r['iterations']:
                projections=candidate.get('projection_candidates',[dict(candidate_index=0,runs=candidate['runs'])])
                for projection in projections:
                    for run in projection['runs']:
                        if 'metrics' not in run:continue
                        selected=projection['candidate_index']==candidate.get('selected_projection_candidate',0)
                        records.append(normalized(path,run['metrics'],method,case,rows,
                            trace=path.parent/run['trace'],declared_hash=run['trace_sha256'],
                            primary=candidate['iteration']==3 and selected,
                            configuration={'version':method,'solver':'COPP 0.2.3 TOPP3-SOCP adapter',
                                'iteration':candidate['iteration'], 'fixed_iteration_for_table':3,
                                'command_projection_candidate':projection['candidate_index'],
                                'selected_projection_candidate':candidate.get('selected_projection_candidate',0),
                                'solver_wall_s':candidate['solver_wall_s'],
                                'parent_profile_sha256':candidate.get('parent_profile_sha256')}))
    for path in sorted((ROOT/'results/historical_strong_v5').glob('*_dt*us.json')):
        r=read(path)
        records.append(normalized(path,r,'historical_strong',r['path'],rows,
            configuration={'version':r['runtime_version'],'checkpoint_sha256':r['model_sha256'],'seed':r['seed']}))
    for path in sorted((ROOT/'results/physical_fit_v5/pilot_seed101/adaptive/evaluation_v5').glob('*_dt*us.json')):
        r=read(path)
        records.append(normalized(path,r,'fresh_physical_fit_seed101',r['path'],rows,
            configuration={'version':r['runtime_version'],'checkpoint_sha256':r['model_sha256'],'seed':r['seed'],
                           'identity':'Fresh seed101 physical-state teacher, 24000 supervised updates; not historical main, PPO, or five independent seeds'}))
    for arm in ('adaptive','fixed'):
        folder=ROOT/'results/recovery_training_v6'/('seed101_'+arm)/'evaluation'
        for path in sorted(folder.glob('*_dt*us.json')):
            r=read(path)
            records.append(normalized(path,r,'recovery_fit_v6_'+arm,r['path'],rows,
                configuration={'version':r['runtime_version'],'checkpoint_sha256':r['model_sha256'],
                    'seed':r['seed'],'arm':arm,'identity':'Fresh student with rebuilt training features and physical recovery perturbations; paired 24000-update pilot conditional on one shared historical teacher'}))
    for method, name in [('predictive_v5_1','predictive_action_candidate_v5'),
                         ('predictive_v5_2','predictive_stop_candidate_v5_2'),
                         ('recoverable_v5_3','recoverable_action_candidate_v5_3'),
                         ('resumption_v5_4','geometric_resumption_candidate_v5_4')]:
        for path in sorted((ROOT/'results'/name).glob('*/actual.json')):
            r=read(path)
            if r.get('path') not in rows:continue
            records.append(normalized(path,r,method,r['path'],rows,
                trace=path.parent/'actual.npz',
                configuration={'version':r.get('method',r.get('version')),'checkpoint_sha256':r['model_sha256'],
                    'identity':'Separate predictive/recoverable-action development candidate',
                    'action_source':r['action_source']}))
    for path in sorted((ROOT/'results/geometric_resolution_check_v5_4/dt500us').glob('*/actual.json')):
        r=read(path)
        if r.get('path') not in rows:continue
        records.append(normalized(path,r,'resumption_v5_4',r['path'],rows,
            trace=path.parent/'actual.npz',configuration={'version':r['version'],
                'checkpoint_sha256':r['model_sha256'],
                'identity':'Same candidate under physical-time-preserving 0.5 ms entry; numerical check, not a training replicate',
                'action_source':r['action_source']}))
    return records


def teacher_inventory():
    registration=V3/'results/unified_v3/registration.json'
    r=read(registration)
    candidates=r.get('teacher_candidates',r.get('candidates',[]))
    if not candidates:
        candidates=[v for v in r.values() if isinstance(v,list) and v and isinstance(v[0],dict) and 'feed' in v[0]]
        candidates=candidates[0] if candidates else []
    records=[]
    for name in ['f60_k1_h12_d3','f80_k1_h8_d1p5']:
        for p in sorted((V3/'results/unified_v3/teachers'/name).glob('*.json')):
            item=read(p)
            if 'success' not in item:continue
            records.append(dict(configuration=name,path_id=p.stem,record=str(p),record_sha256=sha(p),
                success=item['success'],complete_time_s=item.get('complete_time_s'),
                elapsed_prefix_s=item.get('elapsed_prefix_s'),fixed_preview_mm=item.get('fixed_preview_mm')))
    for p in sorted((V4/'results/observation_teacher_observed_development_v4').glob('final_*.json')):
        item=read(p);expected=1.2414357920670511 if p.stem=='final_rounded_triangle' else 1.25
        records.append(dict(configuration='observation_latency070 (not f60/1.5)',path_id=p.stem,
            record=str(p),record_sha256=sha(p),success=item['success'],complete_time_s=item.get('complete_time_s'),
            elapsed_prefix_s=item.get('elapsed_prefix_s'),fixed_preview_mm=item.get('fixed_preview_mm'),
            tolerance_mm=item.get('tolerance_mm'),common_tolerance_matches=abs(item['tolerance_mm']-expected)<1e-10))
    return dict(scope='Read-only inventory of canonical v3 teachers and v4 observed-development teacher; no new teacher searches executed',
        registration=str(registration),registration_sha256=sha(registration),candidates=candidates,
        requested_f60_preview1p5_found=False,
        conclusion='No combined f60 + preview 1.5 mm candidate/run found. f60 has preview 3 mm; preview 1.5 mm candidate has feed 80. Neither has canonical fold/chamfer records.',
        observed_teacher_caveat='observation_latency070 uses preview 3 mm; rounded-triangle record used 1.25 rather than chord-reserved 1.241435792 mm, so it is excluded from common-condition baseline ranking.',
        records=records)


def main():
    registration=read(ROOT/'results/historical_strong_v5/registration.json')
    rows={r['id']:r for r in registration['cases']}
    records=collect(rows)
    matrix=[]
    for case in ORDER:
        cells={}
        for method in METHODS:
            matches=[r for r in records if r['path_id']==case and r['method']==method and r['dt_s']==.001 and r['primary_configuration']]
            if len(matches)>1:raise ValueError('Ambiguous canonical cell: '+case+' '+method)
            not_registered = ((method=='fresh_physical_fit_seed101' and case.startswith('final_')) or
                (method.startswith('recovery_fit_v6') and (case.startswith('final_') or case=='original')) or
                (method.startswith(('predictive','recoverable','resumption')) and case not in ['final_dense_fold','final_chamfer_chain','validation_zigzag']))
            missing='NOT_REGISTERED' if not_registered else 'MISSING'
            cells[method]=matches[0] if matches else dict(status=missing,method=method,path_id=case,complete_time_s=None)
        eligible=[r for m,r in cells.items() if m in BASELINES and r['status']=='PASS']
        best=min(eligible,key=lambda r:r['complete_time_s']) if eligible else None
        strong=cells['historical_strong']
        improvement=(100*(best['complete_time_s']-strong['complete_time_s'])/best['complete_time_s']
                     if best and strong['status']=='PASS' else None)
        matrix.append(dict(path_id=case,evaluation_tolerance_mm=rows[case]['evaluation_tolerance_mm'],
            cells=cells,strongest_observed_common_baseline=best,
            strong_time_reduction_percent_if_both_complete=improvement))
    result=dict(created_utc=utc(),scope='All thirteen are development programs, not an independent final test',
        sources_policy='Canonical S2/S2R corner_scale=1; SCMO v2; separate COPP v5b/v5c/v5d columns use fixed iteration 3; all three optimizer iterations and command-projection attempts remain indexed. Predictor versions remain separate from the historical actor. Strongest is a descriptive development envelope, not a deployable oracle or a frozen global selector.',
        units='One program execution per model/configuration; no inferential intervals from deterministic ticks.',
        no_recomputed_geometry_claim='Metrics are indexed from prior full-command independent audits; NPZ hashes and complete reference-array identity rechecked here.',
        matrix=matrix,all_runs=records,teacher_inventory=teacher_inventory())
    write_json(ROOT/'audit/development_matrix_v5.json',result)
    lines=['# Observed development matrix (1 ms)','',result['scope'],result['sources_policy'],'',
           'PASS requires complete ordered traversal, every command interval inside the common corridor, planar vector limits, final rest, matching reference arrays and verified trace hash. FAIL retains its prefix; MISSING means no canonical execution is available.','',
           '| Program | '+' | '.join(METHODS)+' | Fastest available feasible baseline |',
           '|---|'+'---:|'*len(METHODS)+'---|']
    flat=[]
    for row in matrix:
        cells=row['cells']
        def fmt(cell):
            return f"{cell['complete_time_s']:.3f}" if cell['status']=='PASS' else cell['status']
        best=row['strongest_observed_common_baseline']
        label=f"{best['method']} {best['complete_time_s']:.3f} s" if best else 'MISSING'
        lines.append('| '+row['path_id']+' | '+' | '.join(fmt(cells[m]) for m in METHODS)+' | '+label+' |')
        flat.append(dict(path_id=row['path_id'],**{m:fmt(cells[m]) for m in METHODS},strongest_baseline=label))
    fresh=[r for r in records if r['method']=='fresh_physical_fit_seed101']
    fresh_failures='; '.join(f"{r['path_id']} {r['dt_s']*1000:g} ms: {r['elapsed_prefix_s']:.3f} s prefix" for r in fresh if r['status']=='FAIL')
    lines.extend(['','No f60/1.5 mm teacher run was found in the inspected canonical inventory. f60/d3 and f80/d1.5 are distinct configurations. Neither supplies fold/chamfer coverage.',
        f"The fresh physical-state fit is a separate seed101 checkpoint, with 24000 supervised updates. Its {len(fresh)} registered executions retain these failures: {fresh_failures}. The five observed programs were not registered for this pilot. No failed prefix is a complete time.",
        'The older observation_latency070 teacher completed chamfer (3.683 s) but failed fold. Its rounded-triangle execution did not reserve the arc chord error and is not a common-condition comparator.',
        '', 'Regenerate: `D:\\Anaconda\\envs\\PPO\\python.exe src\\audit_development_matrix_v5.py` from the v5 directory.',''])
    (ROOT/'audit/development_matrix_v5.md').write_text('\n'.join(lines),encoding='utf-8')
    with (ROOT/'audit/development_matrix_v5.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print(json.dumps(dict(matrix_rows=len(matrix),indexed_runs=len(records),
        incompatible=[dict(case=r['path_id'],method=r['method'],checks=r['checks']) for r in records if r['status']=='INCOMPATIBLE_OR_INCOMPLETE_EVIDENCE'],
        strong_complete=sum(r['cells']['historical_strong']['status']=='PASS' for r in matrix)),indent=2))


if __name__=='__main__':
    main()
