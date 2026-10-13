"""Three conditional seeds fit geometric anticipation; not PPO or full-lap evidence."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,time,json,pickle,shutil
import numpy as np
import torch
from scipy.optimize import differential_evolution
OUT=Path(__file__).resolve().parent
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
sys.path[:0]=[str(BASE),str(OUT)]
from study_io import write_json,sha
from probe_all34_feed_boundary import rollout

class AnticipationProbe:
    def __init__(self,base,parameters):self.base=base;self.p=np.asarray(parameters,dtype=float)
    def bind(self,env):return self.base.bind(env)
    def act(self,observation,deterministic=True):
        latent=self.base.act(observation,deterministic)[0].astype(float)
        d=np.asarray(observation)[42:62].reshape(10,2);a=np.unwrap(np.arctan2(d[:,1],d[:,0]))
        # Differences of observed directions form reflection-odd anticipation bands.
        basis=np.array([a[5]-a[3],a[7]-a[5],a[9]-a[7]])
        steering=self.p[1]*(np.pi/2)*np.tanh(latent[0])+basis@self.p[2:5]
        latent[0]=np.arctanh(np.clip(steering/(np.pi/2),-.999999999,.999999999))
        latent[1]=20.  # fixed intervention, not a learned feed controller
        latent[2]=np.arctanh(np.clip((self.p[0]-6.4)/5.6,-.999999999,.999999999))
        return latent,0.,0.

def main():
    torch.set_num_threads(1);root=OUT/'all34_dominance_research';stage=root/'conditional_anticipation_search';stage.mkdir(exist_ok=False)
    cache=root/'selected_causal_contexts/contexts.pkl'
    with cache.open('rb') as f:all_contexts=pickle.load(f)
    training=[all_contexts[i] for i in [0,1]]
    validation=[all_contexts[i] for i in [3,5,7,8,9,11,12,13,15,17,19,22,24,25,27,32,33]]
    seeds=[922101,922102,922103];bounds=[(3.,7.7),(-.6,1.2),(-.45,.45),(-.12,.12),(-.035,.035)]
    source=[Path(__file__),OUT/'probe_all34_feed_boundary.py',OUT/'arc_cruise_policy.py',OUT/'cache_corner_contexts.py']
    (stage/'source').mkdir()
    for p in source:shutil.copy2(p,stage/'source'/p.name)
    write_json(stage/'registration.json',dict(kind='conditional_local_parameter_optimization',ppo_updates=0,
        full_pipeline_independent_training=False,seeds=seeds,wall_budget_s=1800,optimization_deadline_s=1400,
        parameters=['constant_preview_mm','parent_steering_scale','7minus3mm_tangent_weight','15minus7mm_tangent_weight','30minus15mm_tangent_weight'],
        bounds=bounds,feed='100mm/s diagnostic intervention throughout local episode',
        objective='First full local feasibility. Then mean whole-local integrated speed deficit plus 0.25 times worst registered +/-10mm deficit. All local rows count, preventing window-only reward.',
        training_contexts=[c['label'] for c in training],check_contexts=[c['label'] for c in validation],
        adopt='Local diagnosis only. A candidate needs all34 original full-lap and two-resolution validation before any method adoption.',
        stop='80 objective calls per seed, equal 440-second optimization budget per seed; failure terminates that context; preserve negatives.',
        source_sha256={str(p):sha(p) for p in source},source_cache_sha256=sha(cache)))
    metadata=torch.load(all_contexts[0]['model'],map_location='cpu',weights_only=False)
    start=time.perf_counter();trials=[];retained=[]
    for si,seed in enumerate(seeds):
        seed_start=time.perf_counter();deadline=min(start+1400,seed_start+440);best=None;count=0
        def objective(parameters):
            nonlocal best,count
            if time.perf_counter()>deadline-12:raise TimeoutError('Equal per-seed budget exhausted')
            label=f'seed{seed}_trial{count:03d}';count+=1;rr=[]
            for c in training:
                r=rollout(metadata,c,parameters.tolist(),stage/(label+'_'+c['corner']['id']+'.npz'),deadline,AnticipationProbe);rr.append(r)
                if not r['context_pass']:break
            valid=len(rr)==len(training) and all(r['context_pass'] for r in rr)
            if valid:
                cost=float(np.mean([r['total_local_speed_deficit_mm'] for r in rr])+.25*max(r['speed_deficit_mm'] for r in rr))
                cost+=10.*sum(max(r['maximum_local_distance_mm']-1.23,0.) for r in rr)
            else:
                cost=100.+10.*(len(training)-len(rr))+sum(max((r['maximum_local_distance_mm'] or 10)-1.25,0.) for r in rr)
            record=dict(label=label,parameters=parameters.tolist(),valid=valid,cost=cost,records=rr)
            trials.append(record)
            if best is None or cost<best['cost']:
                best=record
                print('ANTICIPATION_BEST',seed,count,'valid',valid,'cost',cost,'window',[r['speed_deficit_mm'] for r in rr],'p',parameters.tolist(),flush=True)
            if count%10==0:print('ANTICIPATION_PROGRESS',seed,count,'wall',time.perf_counter()-start,flush=True)
            write_json(stage/'progress.json',dict(trials=trials,retained=retained,wall_s=time.perf_counter()-start,goal_complete=False))
            return cost
        try:
            differential_evolution(objective,bounds,seed=seed,popsize=4,maxiter=3,polish=False,workers=1,
                                   x0=np.array([4.6,0.,0.,0.,0.]),updating='immediate',tol=0.)
        except TimeoutError as exc:print('SEED_STOP',seed,str(exc),flush=True)
        if best is not None:retained.append(best)
        write_json(stage/'retained.json',retained)
    candidates=[r for r in retained if r['valid']]
    if not candidates:
        write_json(stage/'receipt.json',dict(goal_complete=False,status='no_feasible_local_candidate',wall_s=time.perf_counter()-start));return
    chosen=min(candidates,key=lambda r:r['cost']);checked=[]
    for c in validation:
        if time.perf_counter()>start+1760:break
        r=rollout(metadata,c,chosen['parameters'],stage/('checked_'+c['corner']['id']+'.npz'),start+1780,AnticipationProbe)
        checked.append(r);print('ANTICIPATION_CHECK',r['corner'],r['context_pass'],r['speed_deficit_mm'],r['maximum_local_distance_mm'],flush=True)
        write_json(stage/'checks.json',checked)
    write_json(stage/'selected_diagnostic_parameters.json',dict(parameters=chosen['parameters'],selected_from=chosen['label'],
        model=all_contexts[0]['model'],source_spec=all_contexts[0]['source_spec'],goal_complete=False,deployable=False))
    write_json(stage/'receipt.json',dict(goal_complete=False,status='conditional_diagnostic_search_complete',ppo_updates=0,
        conditional_seeds=len(retained),objective_calls=len(trials),selected=chosen,checks=checked,wall_s=time.perf_counter()-start))
    print('ANTICIPATION_FINISHED',len(trials),time.perf_counter()-start,flush=True)

if __name__=='__main__':main()
