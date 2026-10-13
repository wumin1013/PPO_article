"""Freeze completed continuation searches, then run a bounded complete-set evaluation."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import json
import time
import hashlib
import importlib.util
import sys
import numpy as np
import torch
from deep_joint_credit_policy import load_credit_policy
from learned_action_pulse_policy import environment_factory
from paper_bend_residual_policy import BendResidualPolicy
from paper_complete_evaluation import evaluate
from geometry_spec import audit_commands
from validate_actions import difference_audit
from complete_motion_metrics import complete_motion_metrics
from study_io import BASE,ROOT,register,receipt,sha,write_json


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def s2_case(reference,closed,path,source_receipt):
    source=ROOT/'论文项目/revision_20260913_preview_efficiency_v2/conventional_ts_v1'
    g=module('paper_confirm_s2_geometry',source/'geometry.py')
    s=module('paper_confirm_s2_scheduler',source/'scheduler.py')
    c=module('paper_confirm_s2_command',source/'command.py')
    g.CORNER_ERROR_MM=1.2447;g.SOURCE_REGISTRATION_SHA=source_receipt
    arcs=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(reference,axis=0),axis=1))]
    raw=[g.Line(reference[i],reference[i+1],arcs[i],arcs[i+1]) for i in range(len(reference)-1)]
    item=dict(id=path.stem,closed=closed,reference_file='frozen_confirmatory_set',reference_sha256=source_receipt)
    g._source_geometry=lambda path_id:(item,reference,raw,0.)
    started=time.perf_counter()
    try:
        curve=g.build_geometry(item['id'])
        schedule=s.schedule(curve.curve_lengths_mm,curve.curve_speed_caps_mm_s,curve.straight_lengths_mm,F=100.,A=1998.,J=62437.5)
        assembled=c.assemble(schedule,curve);positions=assembled['pos']
        geometry,_=audit_commands(reference,positions,endpoint_tolerance=1.25,closed=closed)
        rows=np.zeros((len(positions)-1,13));rows[:,:2]=positions[1:]
        rows[:,2:4]=np.diff(positions,axis=0)/.001
        rows[:,4:6]=np.diff(np.vstack([np.zeros(2),rows[:,2:4]]),axis=0)/.001
        rows[:,6:8]=np.diff(np.vstack([np.zeros(2),rows[:,4:6]]),axis=0)/.001
        physical=difference_audit(reference[0],rows)
        success=bool(geometry['geometry_complete'] and physical['kinematic_pass'] and physical['rest_pass'])
        np.savez_compressed(path,reference=reference,rows=rows)
        result=dict(success=success,reason='success' if success else 'audit_failure',error=None,
            complete_time_s=len(rows)*.001 if success else None,elapsed_prefix_or_complete_s=len(rows)*.001,
            geometry=geometry,audit=physical,motion=complete_motion_metrics(reference,rows),
            motion_scope='complete' if success else 'failed_command',trace_sha256=sha(path),
            actor_calls=0,physics_ticks=len(rows),wall_s=time.perf_counter()-started)
    except Exception as exc:
        result=dict(success=False,reason='planner_or_adapter_exception',error=f'{type(exc).__name__}: {exc}',
            complete_time_s=None,actor_calls=0,physics_ticks=0,wall_s=time.perf_counter()-started,
            physical_failure_not_inferred_from_exception=True)
    write_json(path.with_suffix('.json'),result)
    return result


def main():
    torch.set_num_threads(1);review=BASE/'paper_evidence_review';learning=review/'bend_residual_learning'
    lr=read(learning/'receipt.json');assert lr['completed_searches']==12 and lr['stop_reason'] is None
    frozen=read(review/'frozen_evaluation_candidate.json');model=BASE/frozen['model'];assert sha(model)==frozen['sha256']
    runs=read(learning/'results.json')['runs']
    specs=[read(BASE/r['selected_policy']) for r in runs]
    for run in runs:assert sha(BASE/run['selected_policy'])==run['selected_policy_sha256']
    sealed=BASE/'sealed_test_preparation_v1/paths.json'
    assert sha(sealed)=='8ef7d772056373a68e9f3d74ff7d539ebef34a295ad165fc1fb0d92e23ad929a'
    s2source=ROOT/'论文项目/revision_20260913_preview_efficiency_v2/conventional_ts_v1'
    sources=[__file__,BASE/'paper_bend_residual_policy.py',BASE/'paper_state_support_policy.py',
        BASE/'paper_complete_evaluation.py',review/'论文完成执行约定.md',learning/'receipt.json',learning/'results.json',
        review/'frozen_evaluation_candidate.json',review/'evidence_matrix.json',review/'support_repair/results.json',
        BASE/'sealed_test_preparation_v1/receipt.json']+[BASE/r['selected_policy'] for r in runs]
    sources += [s2source/n for n in ('geometry.py','scheduler.py','command.py')]
    stage=register('paper_evidence_review/confirmatory',sources,
        dict(total_wall_s=3600,per_long_case_wall_s=600,development_logical_max=156,
             sealed_logical_max=420,new_long_logical_max=12,primary_seed=921101,
             training_updates=0,optimization_rounds_remaining=0),
        'Freeze all12 independently searched conditional continuation arms and the original single '
        'checkpoint BEFORE reading sealed coordinates. No model or D selection after this point. '
        'Evaluate complete predefined sets; exact identical policies reuse raw outcomes with aliases. '
        'Preserve failures, incomplete coverage and budget stops. Three continuation seeds share '
        'a frozen historical backbone; full-pipeline budgets and from-scratch repetition remain unproved.')
    freeze=dict(backbone=frozen,selected=specs,primary_seed=921101,
        model_selection_complete=True,all_selected_policy_hashes=[r['selected_policy_sha256'] for r in runs],
        full_conditional_learning_method_frozen=True,from_scratch_pipeline_replication_claim=False,
        fixed_d_selection='one global D per independently searched seed, training outcomes only',
        no_post_freeze_training_or_selection=True,source_sha256=read(stage/'registration.json')['source_sha256'],
        sealed_sha256=sha(sealed),sealed_coordinates_opened=False)
    write_json(stage/'method_freeze.json',freeze)
    # Only now are sealed coordinates parsed for the first time in this evidence continuation.
    tests=read(sealed)['cases'];assert len(tests)==30
    write_json(stage/'sealed_open_receipt.json',dict(method_freeze_sha256=sha(stage/'method_freeze.json'),
        sealed_sha256=sha(sealed),case_count=30,post_freeze=True))
    dev=read(BASE/'learning_preparation_v4/paths.json')['validation']
    allcases=[dict(id=f'development_{i:02d}',split='development',reference=r,closed=bool(np.linalg.norm(np.asarray(r)[0]-np.asarray(r)[-1])<1e-9)) for i,r in enumerate(dev)]
    allcases += [dict(x,split='sealed') for x in tests]
    for i,item in enumerate(read(review/'evidence_matrix.json')['new_long3']):
        source=item['source'];assert sha(source)==item['source_sha256']
        points=np.loadtxt(source,delimiter=',',skiprows=1);theta=np.deg2rad(item['rotation_deg'])
        R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        points=points[0]+item['scale']*(points-points[0])@R.T
        allcases.append(dict(id=f'new_long_{i}',split='new_long',reference=points.tolist(),
            closed=bool(np.linalg.norm(points[0]-points[-1])<1e-9),transformation=item))
    write_json(stage/'evaluation_cases.json',allcases)
    logical=[dict(arm='original',seed=None,coefficients=None,fixed_d=None)]+specs
    unique={};aliases=[]
    for spec in logical:
        content={k:spec.get(k) for k in ('arm','coefficients','fixed_d','upper')}
        identity=hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()
        aliases.append(dict(arm=spec['arm'],seed=spec.get('seed'),identity=identity))
        unique.setdefault(identity,spec)
    write_json(stage/'policy_aliases.json',aliases)
    parent=torch.load(model,map_location='cpu',weights_only=False)
    started=time.perf_counter();deadline=started+3600;results=[];stop=None
    # Short, complete sets first; long checks cannot consume their evaluation budget.
    for case in allcases:
        reference=np.asarray(case['reference'],float)
        for identity,spec in unique.items():
            if case['split']=='new_long' and spec['arm'] not in ('original','joint'):continue
            if time.perf_counter()>=deadline:stop='registered_confirmation_wall_limit';break
            folder=stage/identity[:12];folder.mkdir(exist_ok=True);path=folder/(case['id']+'.npz')
            # Original development outcomes are already complete and must not be rerun.
            if case['split']=='development' and spec['arm']=='original':
                old=read(review/'development/results.json')['records'][int(case['id'].split('_')[-1])]
                result=dict(old,reused_existing_trace=True,actor_calls=0,physics_ticks=0,wall_s=0.)
            elif case['split']=='development' and spec['arm']=='joint' and not np.any(spec['coefficients']):
                old=read(review/'support_repair/results.json')['development'][int(case['id'].split('_')[-1])]
                if old.get('reused_by_exact_input_identity'):
                    old=read(review/'development/results.json')['records'][int(case['id'].split('_')[-1])]
                result=dict(old,reused_existing_trace=True,actor_calls=0,physics_ticks=0,wall_s=0.)
            else:
                base=load_credit_policy(parent)
                policy=base if spec['arm']=='original' else BendResidualPolicy(base,spec['upper'],spec['coefficients'],spec['arm'])
                env=environment_factory(reference);env.fixed_d=spec.get('fixed_d')
                per_case_deadline=min(deadline,time.perf_counter()+600) if case['split']=='new_long' else deadline
                result=evaluate(policy,env,path,per_case_deadline)
                result['trace']=str(path.relative_to(BASE))
            result.update(case=case['id'],split=case['split'],family=case.get('family'),identity=identity,arm=spec['arm'])
            results.append(result)
            print('CONFIRM',case['id'],spec['arm'],result['success'],result.get('complete_time_s'),flush=True)
            write_json(stage/'results.json',dict(results=results,stop_reason=stop))
        if stop:break
        if case['split']=='sealed':
            folder=stage/'S2';folder.mkdir(exist_ok=True)
            result=s2_case(reference,case['closed'],folder/(case['id']+'.npz'),sha(stage/'method_freeze.json'))
            result.update(case=case['id'],split='sealed',family=case['family'],identity='S2',arm='S2',trace=str((folder/(case['id']+'.npz')).relative_to(BASE)))
            results.append(result)
            print('CONFIRM',case['id'],'S2',result['success'],result.get('complete_time_s'),flush=True)
            write_json(stage/'results.json',dict(results=results,stop_reason=stop))
    write_json(stage/'results.json',dict(results=results,stop_reason=stop,policy_aliases=aliases,
        complete_planned_loop=stop is None,wall_s=time.perf_counter()-started))
    receipt(stage,dict(status='completed_frozen_confirmatory_evaluation' if not stop else 'stopped_bounded_confirmatory_evaluation',
        results_sha256=sha(stage/'results.json'),method_freeze_sha256=sha(stage/'method_freeze.json'),
        unique_evaluation_records=len(results),unique_learned_policy_identities=len(unique),
        aliases=len(aliases),actor_calls=sum(r['actor_calls'] for r in results),
        physics_ticks=sum(r['physics_ticks'] for r in results),wall_s=time.perf_counter()-started,
        sealed_test_opened=True,post_freeze_updates=0,post_freeze_selection=0,stop_reason=stop,
        original_research_positive_claims_not_presumed=True,formal_retained_model_unchanged=True))
    print('CONFIRMATION_FINISHED',len(results),stop,flush=True)


if __name__=='__main__':main()
