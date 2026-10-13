"""Bounded three-seed conditional parameter search and one complete-lap check."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,argparse,shutil
from pathlib import Path
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
from cruise_residual_policy import CruiseResidualPolicy
from arc_cruise_policy import build_policy

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['joint','fixed'],default='joint')
    ap.add_argument('--with-arcs',action='store_true');ap.add_argument('--arc-context',action='store_true')
    ap.add_argument('--arc-specific',action='store_true');args=ap.parse_args()
    if args.arc_specific:args.arc_context=True
    if args.arc_context:args.with_arcs=True
    torch.set_num_threads(1);case=BASE/'paper_evidence_review/natural_closed_case'
    stage=OUT/(args.arm+('_arc_specific' if args.arc_specific else '_curve_context' if args.arc_context else '_curve' if args.with_arcs else '')+'_cruise_training');stage.mkdir(exist_ok=False)
    geometry=json.loads((case/'geometry.json').read_text(encoding='utf-8'))
    reference=np.loadtxt(case/'reference.csv',delimiter=',',skiprows=1)
    model=BASE/'canonical_gate_band_worst_guard_joint_learning_20260920/shadow_training_policy.pt'
    metadata=torch.load(model,map_location='cpu',weights_only=False)
    local=[]
    for corner in geometry['sharp_corners']:
        k=corner['reference_vertex_index'];v=reference[k];u=v-reference[k-1];u/=np.linalg.norm(u)
        w=reference[k+1]-v;w/=np.linalg.norm(w)
        local.append(dict(id=corner['id'],reference=np.array([v-50*u,v-25*u,v,v+25*w,v+50*w])))
    train=[r for r in local if r['id'] in geometry['mechanism_columns']]
    train.append(dict(id='straight_1000mm',reference=np.array([[0.,0.],[500.,0.],[1000.,0.]])))
    if args.with_arcs:
        for primitive in geometry['primitives']:
            if primitive['kind']=='arc':
                first,last=primitive['start_reference_index'],primitive['end_reference_index']
                arc=reference[first:last+1].copy()
                if args.arc_context:
                    u=reference[first]-reference[first-1];u/=np.linalg.norm(u)
                    w=reference[last+1]-reference[last];w/=np.linalg.norm(w)
                    arc=np.vstack([arc[0]-100*u,arc[0]-50*u,arc,arc[-1]+50*w,arc[-1]+100*w])
                train.append(dict(id=primitive['label'],reference=arc))
    training_end=660 if args.with_arcs else 520
    corners_end=880 if args.with_arcs else 740
    steps_per_seed=3 if args.with_arcs else 4
    seeds=[921201,921202,921203]
    registration=dict(arm=args.arm,kind='conditional_closed_loop_parameter_optimization',independent_full_pipeline_ppo=False,
        ppo_updates=0,offline_fit_updates=0,seeds=seeds,training_ids=[r['id'] for r in train],
        objective='lexicographic full-success then summed complete physical time on prespecified development snippets',
        final_selection='All34 corner times plus complete straight and arc training times; representative corners are not double-counted.' if args.with_arcs else 'All34 corner complete time',
        parameters=['distributed_curvature_feed_amplitude','visible_curvature_scale_mm_inv'] if args.arc_specific else ['shared_feed_latent_amplitude','continuous_local_basis_width_rad'],
        unchanged='Three-action learned backbone, observation support cap, dynamics, coverage, reference, terminal/rest, thresholds.',
        wall_budget_s=1800,training_limit_s=training_end,all34_limit_s=corners_end,full_run_deadline_s=1740,
        adoption='Research candidate only: full success, all 34 corners, independent vector/analytic audits, time <=445.085 s (>=0.100 s ahead of historical S2). Final acceptance still requires strong fixed repair opportunity, numerical sensitivity and mechanism checks.',
        stop='On failed snippet retain result and skip remaining snippets for that candidate; at wall partition stop proposing; no full-lap feedback tuning in this round.',
        sources={str(p):sha(p) for p in (Path(__file__),OUT/'cruise_residual_policy.py',OUT/'arc_cruise_policy.py',model,case/'reference.csv',case/'geometry.json',BASE/'paper_complete_evaluation.py')})
    retained=json.loads((OUT/'joint_cruise_training/selected_policy.json').read_text(encoding='utf-8'))['spec'] if args.arc_specific else None
    if args.arc_specific:registration['retained_narrow_basis']=retained
    write_json(stage/'registration.json',registration)
    (stage/'source').mkdir()
    for source in registration['sources']:
        path=Path(source)
        if path.suffix=='.py':shutil.copy2(path,stage/'source'/path.name)
    start=time.perf_counter();records=[];trials=[];seed_best=[]
    def run(spec,items,label,deadline):
        local_records=[]
        for item in items:
            if time.perf_counter()>deadline:break
            policy=build_policy(load_credit_policy(metadata),spec)
            env=environment_factory(item['reference'])
            if args.arm=='fixed':env.fixed_d=spec['fixed_d']
            r=evaluate(policy,env,stage/(label+'_'+item['id']+'.npz'),deadline,max_calls=2000)
            r['item']=item['id'];local_records.append(r);records.append(r)
            if not r['success']:break
        success=len(local_records)==len(items) and all(r['success'] for r in local_records)
        score=sum(r['complete_time_s'] for r in local_records) if success else 1e6+len(items)-len(local_records)
        return dict(spec=spec,success=success,score=score,records=local_records,label=label)
    fixed_d=4.6
    def make_spec(amplitude,width,distance):
        if args.arc_specific:
            return dict(amplitude=retained['amplitude'],width=retained['width'],arc_amplitude=amplitude,arc_scale=width,fixed_d=distance)
        return dict(amplitude=amplitude,width=width,fixed_d=distance)
    if args.arm=='fixed':
        grid=[]
        for distance in [.8,1.5,3.,4.6,6.,8.,12.]:
            if time.perf_counter()>start+170:break
            r=run(make_spec(3.,.0002 if args.arc_specific else .05 if args.with_arcs else .02,distance),train[:3],'distance_'+str(distance),start+170)
            grid.append(r);trials.append(r);write_json(stage/'distance_grid.json',grid)
            print('DISTANCE',distance,r['success'],r['score'],flush=True)
        fixed_d=min(grid,key=lambda x:x['score'])['spec']['fixed_d']
    baseline=run(make_spec(0.,.0002 if args.arc_specific else .02,fixed_d),train,'baseline',start+training_end)
    trials.append(baseline);print('TRAIN_BASELINE',baseline['success'],baseline['score'],flush=True)
    write_json(stage/'training.json',dict(trials=trials,seed_best=seed_best,wall_s=time.perf_counter()-start))
    if not baseline['success']:
        write_json(stage/'receipt.json',dict(status='stopped_baseline_training_path_failure',goal_complete=False,
            wall_s=time.perf_counter()-start,ppo_updates=0,records=baseline['records']))
        return
    for seed in seeds:
        rng=np.random.default_rng(seed);best=None
        for step in range(steps_per_seed):
            if time.perf_counter()>start+training_end-30:break
            if step==0:
                amplitude=float(rng.uniform(2.,4.));width=float(np.exp(rng.uniform(np.log(.0001 if args.arc_specific else .03 if args.with_arcs else .008),np.log(.0005 if args.arc_specific else .12 if args.with_arcs else .06))))
            elif best is not None:
                amplitude=float(np.clip(best['spec']['arc_amplitude' if args.arc_specific else 'amplitude']+rng.normal(0,.8),.5,5.))
                width=float(np.clip(best['spec']['arc_scale' if args.arc_specific else 'width']*np.exp(rng.normal(0,.5)),.00005 if args.arc_specific else .003,.0008 if args.arc_specific else .16 if args.with_arcs else .12))
            spec=make_spec(amplitude,width,fixed_d)
            r=run(spec,train,f'seed{seed}_step{step}',start+training_end);trials.append(r)
            if best is None or r['score']<best['score']:best=r
            write_json(stage/'training.json',dict(trials=trials,seed_best=seed_best,wall_s=time.perf_counter()-start))
            print('PARAMETER_TRAIN',seed,step,spec,r['success'],r['score'],flush=True)
        if best is not None:seed_best.append(best)
    checked=[]
    for r in sorted(seed_best,key=lambda x:x['score']):
        if time.perf_counter()>start+corners_end-30:break
        q=run(r['spec'],local,r['label']+'_all34',start+corners_end)
        q['training_score']=r['score']
        q['selection_score']=q['score']+sum(t['complete_time_s'] or 0. for t in r['records'] if t['item'] not in geometry['mechanism_columns'])
        checked.append(q)
        write_json(stage/'all34_selection.json',checked)
        print('ALL34',q['label'],q['success'],q['score'],flush=True)
    eligible=[r for r in checked if r['success']]
    if not eligible:
        write_json(stage/'receipt.json',dict(status='no_candidate_passed_all34',goal_complete=False,wall_s=time.perf_counter()-start,ppo_updates=0));return
    chosen=min(eligible,key=lambda x:x['selection_score'] if args.with_arcs else x['score']);spec=chosen['spec']
    write_json(stage/'selected_policy.json',dict(spec=spec,selected_from=chosen['label'],seed_best=[dict(label=r['label'],spec=r['spec'],score=r['score']) for r in seed_best],model=str(model),model_sha256=sha(model),goal_complete=False))
    policy=build_policy(load_credit_policy(metadata),spec)
    env=environment_factory(reference)
    if args.arm=='fixed':env.fixed_d=spec['fixed_d']
    result=evaluate(policy,env,stage/'full_trace.npz',start+1740)
    if result['success']:
        z=np.load(stage/'full_trace.npz')
        result['analytic']=analytic_command_audit(reference[0],z['rows'],geometry)
        m=metrics(reference,z['rows'],geometry);write_json(stage/'full_metrics.json',m)
        result['success']=bool(result['analytic']['analytic_corridor_pass'] and m['all34_windows_complete'] and m['projection_raw_backward_steps']==0)
        if not result['success']:result['complete_time_s']=None
    result['spec']=spec;write_json(stage/'full_result.json',result)
    records.append(result)
    write_json(stage/'receipt.json',dict(status='completed_bounded_conditional_parameter_training',goal_complete=False,
        wall_s=time.perf_counter()-start,complete_success=result['success'],complete_time_s=result['complete_time_s'],
        conditional_search_seeds=len(seed_best),independent_ppo_seeds=0,ppo_updates=0,
        actor_calls=sum(r['actor_calls'] for r in records),physics_ticks=sum(r['physics_ticks'] for r in records),
        selected_policy_sha256=sha(stage/'selected_policy.json'),full_result_sha256=sha(stage/'full_result.json')))
    print('COMPLETE_ROUND',args.arm,result['success'],result['complete_time_s'],'wall',time.perf_counter()-start,flush=True)

if __name__=='__main__':main()
