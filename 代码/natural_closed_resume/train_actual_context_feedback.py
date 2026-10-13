"""Conditional three-action learning from exact full-route causal corner states."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time,pickle,shutil,argparse
import numpy as np
import torch
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from deep_joint_credit_policy import load_credit_policy
from learned_action_pulse_policy import environment_factory
from paper_complete_evaluation import evaluate
from natural_case_audit import analytic_command_audit
from natural_case_metrics import metrics
from study_io import write_json,sha
from arc_cruise_policy import build_policy
from cache_corner_contexts import restore_memory


def local_distance(points,context):
    c=context['corner'];r=context['reference'];k=c['reference_vertex_index'];v=r[k]
    incoming=v-r[k-1];incoming/=np.linalg.norm(incoming);outgoing=r[k+1]-v;outgoing/=np.linalg.norm(outgoing)
    starts=np.array([v-60*incoming,v]);ends=np.array([v,v+60*outgoing]);edge=ends-starts
    delta=points[:,None,:]-starts;fraction=np.clip((delta*edge).sum(2)/(edge*edge).sum(1),0.,1.)
    return np.linalg.norm(delta-fraction[:,:,None]*edge,axis=2).min(1)


def context_rollout(metadata,spec,context,file,deadline):
    policy=build_policy(load_credit_policy(metadata),spec);env=environment_factory(context['reference'])
    if context.get('fixed_d') is not None:env.fixed_d=spec['fixed_d']
    policy.bind(env)
    env.restore(context['state']);restore_memory(policy,context['memory']);initial=env.state()
    c=context['corner'];k=c['reference_vertex_index'];vertex=context['reference'][k]
    outgoing=context['reference'][k+1]-vertex;outgoing/=np.linalg.norm(outgoing)
    chunks=[];actions=[];observations=[];latents=[];hit=False;error=None
    try:
        while not env.done and env.ticks-initial['ticks']<2500:
            if time.perf_counter()>=deadline:raise TimeoutError('Context learning budget')
            x=env.observation();latent=policy.act(x,True)[0];assert not np.any(env.pending_pulse)
            _,_,_,info=env.step(latent);rows=info['rows']
            crossing=np.flatnonzero((rows[:,:2]-vertex)@outgoing>=50.)
            if len(crossing):rows=rows[:crossing[0]+1];hit=True
            chunks.append(rows);actions.append(info['held_action']);observations.append(x);latents.append(latent)
            if hit:break
    except Exception as exc:error=f'{type(exc).__name__}: {exc}'
    rows=np.vstack(chunks) if chunks else np.empty((0,13));peaks=None;maximum=None
    if len(rows):
        v=np.diff(np.vstack([initial['position'],rows[:,:2]]),axis=0)/.001
        a=np.diff(np.vstack([initial['velocity'],v]),axis=0)/.001
        j=np.diff(np.vstack([initial['acceleration'],a]),axis=0)/.001
        peaks=[float(np.linalg.norm(x,axis=1).max()) for x in (v,a,j)]
        maximum=float(local_distance(rows[:,:2],context).max())
    passed=bool(error is None and hit and env.tracker.inside and peaks and np.all(np.array(peaks)<=np.array([100,2000,62500])*1.00001))
    np.savez_compressed(file,rows=rows,actions=np.array(actions),latents=np.array(latents),observations=np.array(observations),initial_position=initial['position'])
    result=dict(context=context['label'],context_pass=passed,local_elapsed_s=len(rows)*.001,
        maximum_local_distance_mm=maximum,raw_xy_vector_peaks=peaks,reason=env.reason,error=error,
        full_lap_success_not_tested=True,trace_sha256=sha(file),start_decision=initial['decisions'])
    write_json(file.with_suffix('.json'),result);return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['joint','fixed'],default='joint');args=ap.parse_args()
    torch.set_num_threads(1);stage=OUT/(args.arm+'_actual_context_feedback_training');stage.mkdir(exist_ok=False)
    cache=OUT/('actual_fixed_corner_contexts/contexts.pkl' if args.arm=='fixed' else 'actual_corner_contexts/contexts.pkl')
    with cache.open('rb') as f:contexts=pickle.load(f)
    inherited='fixed_curve_context_cruise_training' if args.arm=='fixed' else 'joint_separate_recovery_training'
    selection=json.loads((OUT/inherited/'selected_policy.json').read_text(encoding='utf-8'))
    model=Path(selection['model']);assert sha(model)==selection['model_sha256']
    metadata=torch.load(model,map_location='cpu',weights_only=False);template=selection['spec'].copy()
    seeds=[921221,921222,921223]
    registration=dict(arm=args.arm,kind='conditional_closed_loop_parameter_optimization',ppo_updates=0,independent_full_pipeline_ppo=False,
        wall_budget_s=1800,training_limit_s=570,all34_limit_s=870,full_deadline_s=1720,seeds=seeds,
        contexts=[c['label'] for c in contexts],context_cache_sha256=sha(cache),parameters=['steering_error_feedback','feed_absolute_error_feedback','global_fixed_preview_mm' if args.arm=='fixed' else 'preview_absolute_error_feedback'],
        objective='Complete context traversal first; target local distance<=1.20mm provides robustness margin within unchanged1.25mm task; then summed local physical duration. These are local consequences, not complete-task times.',
        selection='Retain each conditional seed; check all34 complete snippets; among eligible candidates choose context success/margin then local duration.',
        adoption='Full original lap must pass all audits and beat both strongest baselines at two resolutions before adoption; local checks cannot certify full success.',
        stop='Stop proposal after failed context, retain all failures; bounded training/selection/full-run partitions.',
        source_sha256={str(p):sha(p) for p in [Path(__file__),OUT/'turn_feedback_policy.py',OUT/'arc_cruise_policy.py',OUT/'cruise_residual_policy.py',OUT/'cache_corner_contexts.py',model]})
    write_json(stage/'registration.json',registration);(stage/'source').mkdir()
    for p in registration['source_sha256']:
        p=Path(p)
        if p.suffix=='.py':shutil.copy2(p,stage/'source'/p.name)
    start=time.perf_counter();trials=[];seed_best=[]
    # Put the actually failing context first; no credit from a successful prefix.
    contexts.sort(key=lambda c:0 if c['label']=='failed_recovery_corner_02' else 1)
    def run(coeff,label):
        spec=dict(template,turn_feedback=([float(coeff[0]),float(coeff[1]),0.] if args.arm=='fixed' else np.asarray(coeff).tolist()))
        if args.arm=='fixed':spec['fixed_d']=float(coeff[2])
        records=[]
        for c in contexts:
            if time.perf_counter()>start+570:break
            r=context_rollout(metadata,spec,c,stage/(label+'_'+c['label']+'.npz'),start+570);records.append(r)
            if not r['context_pass']:break
        passed=len(records)==len(contexts) and all(r['context_pass'] for r in records)
        margin=sum(max(0.,r['maximum_local_distance_mm']-1.20) for r in records if r['maximum_local_distance_mm'] is not None)
        key=[0 if passed else 1,margin if passed else len(contexts)-len(records),sum(r['local_elapsed_s'] for r in records)]
        trial=dict(label=label,spec=spec,success=passed,key=key,score=key[-1],records=records);trials.append(trial)
        write_json(stage/'training.json',dict(trials=trials,seed_best=seed_best,wall_s=time.perf_counter()-start))
        print('ACTUAL_CONTEXT',label,coeff,passed,key,flush=True);return trial
    baseline=run([0.,0.,template['fixed_d'] if args.arm=='fixed' else 0.],'unrepaired_baseline')
    if args.arm=='fixed':
        grid=[]
        for distance in (.8,1.5,3.,4.6,6.,8.,12.):
            r=run([0.,0.,distance],'distance_'+str(distance));grid.append(r)
            write_json(stage/'distance_grid.json',grid)
        baseline=min([baseline]+grid,key=lambda r:r['key'])
    for seed in seeds:
        rng=np.random.default_rng(seed);best=baseline if args.arm=='fixed' and baseline['success'] else None
        for step in range(7):
            if time.perf_counter()>start+535:break
            if args.arm=='fixed':
                center=np.array([best['spec']['turn_feedback'][0],best['spec']['turn_feedback'][1],best['spec']['fixed_d']])
                coeff=np.clip(center+rng.normal(0,[.025,.045,.25]),[-.15,-.3,.8],[.05,.2,12.])
            elif step==0:coeff=np.array([-.035,0.,-.065])+rng.normal(0,[.012,.02,.025])
            else:coeff=np.clip(np.array(best['spec']['turn_feedback'])+rng.normal(0,[.015,.035,.035]),[-.15,-.3,-.3],[.05,.2,.15])
            r=run(coeff,f'seed{seed}_step{step}')
            if best is None or r['key']<best['key']:best=r
        if best is not None:
            if args.arm=='fixed':best=dict(best,label=f'seed{seed}_retained',selected_trial_label=best['label'])
            seed_best.append(best)
    case=BASE/'paper_evidence_review/natural_closed_case';reference=np.loadtxt(case/'reference.csv',delimiter=',',skiprows=1)
    geometry=json.loads((case/'geometry.json').read_text(encoding='utf-8'));checked=[]
    for candidate in sorted(seed_best,key=lambda r:r['key']):
        if not candidate['success'] or time.perf_counter()>start+835:continue
        rr=[]
        for c in geometry['sharp_corners']:
            if time.perf_counter()>start+870:break
            k=c['reference_vertex_index'];v=reference[k];u=v-reference[k-1];u/=np.linalg.norm(u);w=reference[k+1]-v;w/=np.linalg.norm(w)
            local=np.array([v-50*u,v-25*u,v,v+25*w,v+50*w]);env=environment_factory(local)
            if args.arm=='fixed':env.fixed_d=candidate['spec']['fixed_d']
            r=evaluate(build_policy(load_credit_policy(metadata),candidate['spec']),env,stage/(candidate['label']+'_all34_'+c['id']+'.npz'),start+870,max_calls=2000)
            rr.append(r)
            if not r['success']:break
        passed=len(rr)==34 and all(r['success'] for r in rr)
        checked.append(dict(label=candidate['label']+'_all34',spec=candidate['spec'],success=passed,records=rr,
            score=sum(r['complete_time_s'] for r in rr) if passed else None,context_key=candidate['key']))
        write_json(stage/'all34_selection.json',checked);print('ALL34_ACTUAL_CONTEXT',candidate['label'],passed,len(rr),flush=True)
    eligible=[r for r in checked if r['success']]
    if not eligible:
        write_json(stage/'receipt.json',dict(status='no_eligible_actual_context_candidate',goal_complete=False,wall_s=time.perf_counter()-start));return
    chosen=min(eligible,key=lambda r:r['context_key']);spec=chosen['spec']
    write_json(stage/'selected_policy.json',dict(spec=spec,selected_from=chosen['label'],model=str(model),model_sha256=sha(model),
        seed_best=[dict(label=r['label'],spec=r['spec'],score=r['score']) for r in seed_best],goal_complete=False))
    env=environment_factory(reference)
    if args.arm=='fixed':env.fixed_d=spec['fixed_d']
    r=evaluate(build_policy(load_credit_policy(metadata),spec),env,stage/'full_trace.npz',start+1720)
    if r['success']:
        rows=np.load(stage/'full_trace.npz')['rows'];r['analytic']=analytic_command_audit(reference[0],rows,geometry)
        m=metrics(reference,rows,geometry);write_json(stage/'full_metrics.json',m)
        r['success']=bool(r['analytic']['analytic_corridor_pass'] and m['all34_windows_complete'] and m['projection_raw_backward_steps']==0)
        if not r['success']:r['complete_time_s']=None
    r['spec']=spec;write_json(stage/'full_result.json',r)
    write_json(stage/'receipt.json',dict(status='completed_actual_context_feedback_round',success=r['success'],complete_time_s=r['complete_time_s'],
        goal_complete=False,wall_s=time.perf_counter()-start,ppo_updates=0,conditional_search_seeds=len(seed_best),result_sha256=sha(stage/'full_result.json')))
    print('ACTUAL_CONTEXT_COMPLETE',r['success'],r['complete_time_s'],r['reason'],time.perf_counter()-start,flush=True)


if __name__=='__main__':main()
