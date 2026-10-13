"""Registered physical-time check of the frozen v5_4 controller at two dt values.

Only an in-memory parameter entry is added. All parent files remain byte-frozen.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import ast
import copy
import inspect
import json
import textwrap
import time
from pathlib import Path
import numpy as np
import torch
from common import ROOT,sha,utc,write_json
import run_recoverable_candidate_v5 as runner
import recoverable_action_candidate_v5 as controller
from geometric_resumption_candidate_v5 import GeometricResumptionCandidate,VERSION
from runtime_physical_time_v5 import install as physical_install

OUT=ROOT/'results/geometric_resolution_check_v5_4'
PRIOR=ROOT/'results/geometric_resumption_candidate_v5_4'
CASES=['final_dense_fold','final_chamfer_chain','validation_zigzag']
ENTRY_VERSION='geometric_v5_4_physical_resolution_entry1'
RESOLUTION_DT=.001


def read(path):return json.loads(Path(path).read_text('utf-8'))


def replaced(function,replacements,namespace):
    source=textwrap.dedent(inspect.getsource(function))
    for old,new in replacements:
        assert source.count(old)==1,(function.__name__,old,source.count(old))
        source=source.replace(old,new)
    output={};exec(compile(source,str(Path(__file__)),'exec'),namespace,output)
    return output[function.__name__]


def install_checked(*,dt,tolerance):
    assert dt==RESOLUTION_DT
    settings=physical_install(dt=dt,tolerance=tolerance)
    import joint_runtime as joint
    import turn_aware_projector as turn
    import continuous_gate_guidance as guide
    assert joint.DT==controller.DT==turn.DT==turn.ORIGINAL.DT==dt
    assert joint.MACRO_TICKS*dt==.02 and joint.FILTER_TAU==.012
    assert guide.GATE_TRANSITION_WIDTH_MM==.1
    # The frozen brake adds three numerical hold samples; preserve its original
    # 3 ms duration rather than shortening it to 1.5 ms at the finer resolution.
    terminal=joint.terminal
    if not getattr(terminal.plan_planar_terminal_brake,'_resolution_entry_v5_4',False):
        terminal.plan_planar_terminal_brake=replaced(terminal.plan_planar_terminal_brake,
            [('int(math.ceil(policy.hold_s/dt))+3','int(math.ceil(policy.hold_s/dt))+round(.003/dt)')],terminal.__dict__)
        terminal.plan_planar_terminal_brake._resolution_entry_v5_4=True
    # Preserve the original 21 ms internal terminal-rest threshold in seconds.
    # Build from the frozen source because the v5 physical patch is in memory.
    source_path=Path(joint.__file__);source=source_path.read_text('utf-8-sig')
    klass=next(x for x in ast.parse(source).body if isinstance(x,ast.ClassDef) and x.name=='JointRuntime')
    node=next(x for x in klass.body if isinstance(x,ast.FunctionDef) and x.name=='step')
    step_source=textwrap.dedent(ast.get_source_segment(source,node))
    for old,new in [('range(MACRO_TICKS+2100)','range(MACRO_TICKS+round(2.1/DT))'),
                    ('self.rest_ticks>=21','self.rest_ticks*DT>=.021-1e-10')]:
        assert step_source.count(old)==1,old
        step_source=step_source.replace(old,new)
    output={};exec(compile(step_source,str(source_path),'exec'),joint.__dict__,output)
    joint.JointRuntime.step=output['step'];joint.JointRuntime.step._physical_time_v5=True
    settings.update(resolution_entry=ENTRY_VERSION,controller_dt_s=controller.DT,
        macro_ticks=joint.MACRO_TICKS,internal_terminal_rest_s=.021,
        common_evaluation_terminal_rest_s=.02,stored_stop_rest_s=.02,
        stopping_hold_s=.023,stopping_extra_hold_s=.003,
        tail_allowance_s=2.1,physical_time_budget='5 s + complete reference length / 5 mm/s',
        stopping_family='Unchanged bounded vector cubic with dt passed explicitly',
        geometric_action_unchanged=True)
    return settings


def configure(dt):
    global RESOLUTION_DT
    assert dt in (.001,.0005);RESOLUTION_DT=dt;controller.DT=dt
    folder=OUT/f'dt{round(dt*1e6)}us'
    namespace=dict(runner.__dict__)
    namespace.update(OUT=folder,VERSION=VERSION,RecoverableActionCandidate=GeometricResumptionCandidate,
        install=install_checked,RESOLUTION_DT=dt)
    unit=replaced(runner.unit_audit,[('install(dt=.001,','install(dt=RESOLUTION_DT,'),
        ('env.rest_ticks*.001','env.rest_ticks*RESOLUTION_DT'),
        ("ROOT/'data/train_box.csv'","ROOT/'data/validation_zigzag.csv'")],namespace)
    run=replaced(runner.run,[('install(dt=.001,','install(dt=RESOLUTION_DT,'),
        (')/.001)',')/RESOLUTION_DT)'),('metrics(ref, positions, .001,','metrics(ref, positions, RESOLUTION_DT,')],namespace)
    return folder,unit,run


def register():
    if OUT.exists():raise FileExistsError('Never replace registered resolution evidence')
    prior=read(PRIOR/'registration.json')
    assert all(sha(ROOT/'src'/name)==digest for name,digest in prior['source_sha256'].items())
    assert all(read(PRIOR/case/'actual.json')['success'] for case in CASES)
    reg=copy.deepcopy(prior)
    reg.update(created_utc=utc(),version=ENTRY_VERSION,parent_controller_version=VERSION,
        parent_registration_sha256=sha(PRIOR/'registration.json'),
        scope='Numerical physical-time resolution checks on three previously observed development programs only; not a new seed or final test',
        resolutions_s=[.001,.0005],control_replay=dict(case='validation_zigzag',dt_s=.001,
            requirement='All arrays, actions and command sources equal the frozen 1 ms full run'),
        stop_precheck='At each dt, use the existing zigzag initial state for raw decision, exact stored-stop execution, stale fingerprint rejection and resumption; no additional path',
        full_runs=[dict(case=case,dt_s=.0005,wall_s=300.) for case in CASES],
        physical_constants=dict(control_period_s=.02,heading_filter_tau_s=.012,guide_transition_mm=.1,
            internal_terminal_rest_s=.021,common_final_rest_s=.02,stored_stop_rest_s=.02,terminal_allowance_s=2.1,
            stopping_hold_s=.023,stopping_extra_hold_s=.003),
        exact_parameter_changes=['Recoverable controller DT for finite differences, full-period/terminal checks and stored-stop integration',
            'Runner installation, length-based tick budget and metric dt',
            'Unit-audit installation and rest time; existing zigzag replaces old unit-only box',
            'Internal terminal rest>=21 ticks expressed as >=0.021 s while retaining the v5 2.1 s terminal allowance',
            'Unchanged cubic brake extra3 hold samples expressed as round(0.003/dt), preserving23ms total physical hold'],
        unchanged=['Model weights, observation, action grid and ordering, geometric resumption action, corridor and gates, tolerances, vector limits, stop family, acceptance metrics and 300 s per-program budget'],
        excluded_unused_dt='The old six-decision predictive module DT and unused high-dimensional JointRuntime observation rest/20 are outside this controller call chain; current FarVectorRuntime has14 physical/geometric observations.',
        qualification='Resolution sensitivity is not training-seed variability, servo dynamics, hardware timing or a completeness guarantee',
        no_holdout=True,no_long_original=True,no_model_or_figure_replacement=True,
        runtime_manifest_sha256=sha(ROOT/'runtime/source_manifest.json'),
        parent_actual_sha256={case:sha(PRIOR/case/'actual.npz') for case in CASES})
    reg['source_sha256'][Path(__file__).name]=sha(__file__)
    write_json(OUT/'registration.json',reg)
    for dt in (.001,.0005):
        sub=copy.deepcopy(reg);sub['dt_s']=dt
        write_json(OUT/f'dt{round(dt*1e6)}us'/'registration.json',sub)
    verify();print('RESOLUTION_REGISTERED',sha(OUT/'registration.json'),flush=True)


def verify():
    reg=read(OUT/'registration.json')
    assert sha(PRIOR/'registration.json')==reg['parent_registration_sha256']
    assert sha(runner.MODEL)==reg['model_sha256']
    assert all(sha(ROOT/'src'/name)==digest for name,digest in reg['source_sha256'].items())
    assert all(sha(ROOT/row['reference'])==row['reference_sha256'] for row in reg['cases'])
    assert all(sha(PRIOR/case/'actual.npz')==digest for case,digest in reg['parent_actual_sha256'].items())
    assert sha(ROOT/'runtime/source_manifest.json')==reg['runtime_manifest_sha256']
    return reg


def precheck(dt):
    verify();folder,unit,_=configure(dt);reg=read(folder/'registration.json')
    started,cpu=time.perf_counter(),time.process_time();unit(reg)
    result=read(folder/'unit_audit/report.json')
    result.update(dt_s=dt,entry_version=ENTRY_VERSION,source_case='validation_zigzag',
        settings=install_checked(dt=dt,tolerance=1.25),wall_s=time.perf_counter()-started,cpu_s=time.process_time()-cpu)
    write_json(folder/'unit_audit/report.json',result);verify()


def run_one(case,dt):
    verify();folder,_,run=configure(dt);reg=read(folder/'registration.json')
    if dt==.001:assert case=='validation_zigzag'
    else:
        assert case in CASES
        assert read(OUT/'control_1ms.json')['status']=='passed'
        assert all(read(OUT/f'dt{round(x*1e6)}us/unit_audit/report.json')['status']=='passed' for x in (.001,.0005))
    started,cpu=time.perf_counter(),time.process_time();result=run(case,reg)
    result['cost'].update(full_run_including_archive_and_metrics_process_cpu_s=time.process_time()-cpu,
        entry_full_wall_s=time.perf_counter()-started)
    result.update(resolution_entry=ENTRY_VERSION,numerical_check_not_new_seed=True,
        action_source='Four logged sources: original model, bounded corrected model, geometric resumption action, stored certified cubic stop',
        action_array_definition='Only actually submitted action requests (original, corrected or geometric); action_decision_ids maps each row to its explicit decision source. Certified stops have submitted_action=null and no action row.')
    write_json(folder/case/'actual.json',result);verify()
    if dt==.001:
        with np.load(PRIOR/case/'actual.npz') as old,np.load(folder/case/'actual.npz') as new:
            assert set(old.files)==set(new.files)
            matches={key:np.array_equal(old[key],new[key]) for key in old.files}
            assert all(matches.values()),matches
        assert result['controller']['counts']==read(PRIOR/case/'actual.json')['controller']['counts']
        write_json(OUT/'control_1ms.json',dict(status='passed',created_utc=utc(),all_arrays_equal=matches,
            complete_time_s=result['complete_time_s'],controller_counts_equal=True,
            old_trace_sha256=sha(PRIOR/case/'actual.npz'),new_trace_sha256=sha(folder/case/'actual.npz')))
    return result


def summarize():
    reg=verify();rows=[]
    assert read(OUT/'control_1ms.json')['status']=='passed'
    for case in CASES:
        old=read(PRIOR/case/'actual.json');new=read(OUT/'dt500us'/case/'actual.json')
        rows.append(dict(case=case,one_ms_success=old['success'],one_ms_complete_time_s=old['complete_time_s'],
            half_ms_success=new['success'],half_ms_complete_time_s=new['complete_time_s'],half_ms_prefix_s=new['elapsed_prefix_s'],
            relative_time_change_percent=None if not (old['success'] and new['success']) else 100*(new['complete_time_s']/old['complete_time_s']-1),
            half_ms_reason=new['reason'],half_ms_vector_limit_ratios=new['vector_limit_ratios'],
            half_ms_geometry=new['geometry'],half_ms_deviation_mm=new['max_polyline_deviation_mm'],
            half_ms_rest_s=new['terminal_rest_s'],controller=new['controller'],cost=new['cost'],
            trace_sha256=new['trace_sha256'],actual_json_sha256=sha(OUT/'dt500us'/case/'actual.json')))
    write_json(OUT/'summary.json',dict(created_utc=utc(),version=ENTRY_VERSION,scope=reg['scope'],results=rows,
        completed_half_ms_runs=sum(x['half_ms_success'] for x in rows),all_failures_retained=True,
        training_runs=0,registration_sha256=sha(OUT/'registration.json')))
    print(json.dumps(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['register','precheck','run','summary']);p.add_argument('--dt',type=float,choices=[.001,.0005]);p.add_argument('--case',choices=CASES)
    a=p.parse_args();torch.set_num_threads(1)
    if a.mode=='register':register()
    elif a.mode=='precheck':assert a.dt;precheck(a.dt)
    elif a.mode=='run':assert a.dt and a.case;run_one(a.case,a.dt)
    else:summarize()
