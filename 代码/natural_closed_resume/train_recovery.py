"""Three conditional search seeds for separate visibility/recovery feed scales."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys,json,time,shutil,argparse
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


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['joint','fixed'],default='joint');args=ap.parse_args()
    torch.set_num_threads(1)
    case=BASE/'paper_evidence_review/natural_closed_case'
    stage=OUT/(args.arm+'_separate_recovery_training');stage.mkdir(exist_ok=False)
    inherited=OUT/'joint_arc_specific_cruise_training'
    selected=json.loads((inherited/'selected_policy.json').read_text(encoding='utf-8'))
    model=Path(selected['model']);assert sha(model)==selected['model_sha256']
    metadata=torch.load(model,map_location='cpu',weights_only=False)
    reference=np.loadtxt(case/'reference.csv',delimiter=',',skiprows=1)
    geometry=json.loads((case/'geometry.json').read_text(encoding='utf-8'))
    local=[];training=[]
    for corner in geometry['sharp_corners']:
        k=corner['reference_vertex_index'];v=reference[k]
        u=v-reference[k-1];u/=np.linalg.norm(u)
        w=reference[k+1]-v;w/=np.linalg.norm(w)
        item=dict(id=corner['id'],reference=np.array([v-50*u,v-25*u,v,v+25*w,v+50*w]))
        local.append(item)
        if item['id'] in geometry['mechanism_columns']:
            for offset in (0.,.5,1.,1.5):
                shifted=item['reference'].copy();shifted[0]-=offset*u
                training.append(dict(id=item['id']+'_lead_'+str(offset),reference=shifted))
    training.append(dict(id='straight_1000mm',reference=np.array([[0.,0.],[500.,0.],[1000.,0.]])))
    seeds=[921211,921212,921213]
    registration=dict(arm=args.arm,kind='conditional_closed_loop_parameter_optimization',ppo_updates=0,
        independent_full_pipeline_ppo=False,seeds=seeds,wall_budget_s=1800,training_limit_s=640,
        all34_limit_s=880,full_run_deadline_s=1720,training_ids=[r['id'] for r in training],
        parameters=['feed_latent_amplitude','heading_recovery_width_rad'],
        retained_geometry_width=selected['spec']['width'],retained_arc_spec=selected['spec'],
        hypothesis='Separating narrow visible-bend suppression from broader heading recovery restores post-corner feed without broadening approach activation.',
        objective='Complete-success first, then summed complete physical time on three angles at four approach-length offsets and straight; select using all34 complete corner times.',
        adoption='Research only. Full original lap must pass all audits, beat S2 and strongest fixed by registered numerical margins, and retain actual mechanism. No freeze here.',
        stop='Stop candidate after first failed snippet; bounded stage deadlines; failures and prefixes preserved.',
        source_hashes={str(p):sha(p) for p in [Path(__file__),OUT/'cruise_residual_policy.py',OUT/'arc_cruise_policy.py',BASE/'paper_complete_evaluation.py',model,case/'reference.csv']})
    write_json(stage/'registration.json',registration);(stage/'source').mkdir()
    for p in registration['source_hashes']:
        path=Path(p)
        if path.suffix=='.py':shutil.copy2(path,stage/'source'/path.name)
    start=time.perf_counter();trials=[];seed_best=[];records=[]
    def run(spec,items,label,limit):
        rr=[]
        for item in items:
            if time.perf_counter()>=start+limit:break
            policy=build_policy(load_credit_policy(metadata),spec);env=environment_factory(item['reference'])
            if args.arm=='fixed':env.fixed_d=spec['fixed_d']
            r=evaluate(policy,env,stage/(label+'_'+item['id']+'.npz'),start+limit,max_calls=2000)
            r['item']=item['id'];rr.append(r);records.append(r)
            if not r['success']:break
        success=len(rr)==len(items) and all(r['success'] for r in rr)
        score=sum(r['complete_time_s'] for r in rr) if success else 1e6+len(items)-len(rr)
        return dict(label=label,spec=spec,success=success,score=score,records=rr)
    template=selected['spec'].copy()
    if args.arm=='fixed':
        fixed=[]
        for distance in (.8,1.5,3.,4.6,6.,8.,12.):
            spec=dict(template,fixed_d=distance,recovery_width=.05)
            r=run(spec,[training[i] for i in (0,4,8)],'distance_'+str(distance),150);fixed.append(r)
            write_json(stage/'distance_grid.json',fixed)
            print('DISTANCE',distance,r['success'],r['score'],flush=True)
        template['fixed_d']=min(fixed,key=lambda r:r['score'])['spec']['fixed_d']
    baseline=run(template,training,'inherited_baseline',640)
    trials.append(baseline)
    write_json(stage/'training.json',dict(trials=trials,seed_best=seed_best,wall_s=time.perf_counter()-start))
    print('PHASE_BASELINE',baseline['success'],baseline['score'],flush=True)
    for seed in seeds:
        rng=np.random.default_rng(seed);best=None
        for step in range(3):
            if time.perf_counter()>start+610:break
            spec=template.copy()
            if step==0:
                spec.update(amplitude=float(rng.uniform(2.5,4.5)),recovery_width=float(rng.uniform(.025,.10)))
            else:
                spec.update(amplitude=float(np.clip(best['spec']['amplitude']+rng.normal(0,.5),1.,5.)),
                    recovery_width=float(np.clip(best['spec']['recovery_width']*np.exp(rng.normal(0,.3)),.015,.15)))
            trial=run(spec,training,f'seed{seed}_step{step}',640);trials.append(trial)
            if best is None or trial['score']<best['score']:best=trial
            write_json(stage/'training.json',dict(trials=trials,seed_best=seed_best,wall_s=time.perf_counter()-start))
            print('RECOVERY_TRAIN',seed,step,spec,trial['success'],trial['score'],flush=True)
        if best is not None:seed_best.append(best)
    checked=[]
    for candidate in sorted(seed_best,key=lambda r:r['score']):
        if not candidate['success'] or time.perf_counter()>start+845:continue
        check=run(candidate['spec'],local,candidate['label']+'_all34',880);checked.append(check)
        write_json(stage/'all34_selection.json',checked)
        print('ALL34',check['label'],check['success'],check['score'],flush=True)
    eligible=[r for r in checked if r['success']]
    if not eligible:
        write_json(stage/'receipt.json',dict(status='no_candidate_passed_all34',wall_s=time.perf_counter()-start,goal_complete=False));return
    best=min(eligible,key=lambda r:r['score']);spec=best['spec']
    write_json(stage/'selected_policy.json',dict(spec=spec,selected_from=best['label'],model=str(model),model_sha256=sha(model),
        seed_best=[dict(label=r['label'],spec=r['spec'],score=r['score']) for r in seed_best],goal_complete=False))
    env=environment_factory(reference)
    if args.arm=='fixed':env.fixed_d=spec['fixed_d']
    result=evaluate(build_policy(load_credit_policy(metadata),spec),env,stage/'full_trace.npz',start+1720)
    if result['success']:
        rows=np.load(stage/'full_trace.npz')['rows'];result['analytic']=analytic_command_audit(reference[0],rows,geometry)
        m=metrics(reference,rows,geometry);write_json(stage/'full_metrics.json',m)
        result['success']=bool(result['analytic']['analytic_corridor_pass'] and m['all34_windows_complete'] and m['projection_raw_backward_steps']==0)
        if not result['success']:result['complete_time_s']=None
    result['spec']=spec;write_json(stage/'full_result.json',result)
    write_json(stage/'receipt.json',dict(status='completed_bounded_conditional_parameter_search',wall_s=time.perf_counter()-start,
        success=result['success'],complete_time_s=result['complete_time_s'],goal_complete=False,
        ppo_updates=0,conditional_search_seeds=len(seed_best),independent_ppo_seeds=0,
        result_sha256=sha(stage/'full_result.json')))
    print('RECOVERY_COMPLETE',args.arm,result['success'],result['complete_time_s'],time.perf_counter()-start,flush=True)


if __name__=='__main__':main()
