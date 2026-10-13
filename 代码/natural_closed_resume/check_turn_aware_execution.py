"""Registered safety and local-physics checks for an explicit executor candidate."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,time,json,pickle,shutil,math,argparse
from types import SimpleNamespace
import numpy as np
import torch
OUT=Path(__file__).resolve().parent
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
sys.path[:0]=[str(BASE),str(OUT)]
from study_io import write_json,sha
import probe_all34_feed_boundary as local
from turn_aware_projector import turn_certificate,turn_backup,TurnAwareProjector,environment_factory
from joint_runtime import tracking

def safety_checks(deadline):
    backups=[];maximum=np.zeros(3)
    for radius in (1.,10.,50.,80.,100.-2e-8):
        for fraction in (0.,.01,.25,.5,.9,.999):
            magnitude=min(1997.,math.sqrt(62375.*radius)*fraction)
            delta=2*math.asin(magnitude*.001/(2*radius))
            for orientation in (0.,.4,2.,-1.7):
                for sign in (-1,1):
                    v=radius*np.array([math.cos(orientation),math.sin(orientation)])
                    previous=radius*np.array([math.cos(orientation-sign*delta),math.sin(orientation-sign*delta)])
                    a=(v-previous)/.001;assert turn_certificate(v,a)
                    count=0
                    while np.linalg.norm(a)>1e-8:
                        if time.perf_counter()>deadline:raise TimeoutError('Safety-check deadline')
                        new_v,new_a,j=turn_backup(v,a)
                        raw_a=(new_v-v)/.001;raw_j=(raw_a-a)/.001
                        maximum=np.maximum(maximum,[np.linalg.norm(new_v),np.linalg.norm(raw_a),np.linalg.norm(raw_j)])
                        assert np.all(maximum<=[100.000001,2000.00001,62500.01])
                        assert np.linalg.norm(new_a)<=np.linalg.norm(a)+1e-7
                        v,a=new_v,new_a;count+=1;assert count<10000
                    backups.append(dict(radius=radius,fraction=fraction,orientation=orientation,sign=sign,steps=count))
    # Exercise certificate switching and braking requests, not just backup formulas.
    env=SimpleNamespace(held=np.array([0.,100.,4.6]));projector=TurnAwareProjector(env)
    v=np.zeros(2);a=np.zeros(2);positions=[np.zeros(2)];rows=[];previous_heading=0.
    for tick in range(2600):
        if time.perf_counter()>deadline:raise TimeoutError('Transition-check deadline')
        t=tick*.001
        if t<.5:heading=0.;feed=100.
        elif t<1.2:heading=.4*(1.-math.cos((t-.5)*math.pi/.7));feed=100.
        elif t<1.7:heading=.8;feed=70.
        elif t<2.1:heading=.8-.2*(1.-math.cos((t-1.7)*math.pi/.4));feed=100.
        else:heading=.4;feed=0.
        env.held[1]=feed;desired=feed*np.array([math.cos(heading),math.sin(heading)])
        _,_,d,_=tracking.compute_tracking_demand(v,a,desired,(heading-previous_heading)/.001)
        vv,aa,j,info=projector.enforce_kinematic_constraints(v,a,d)
        positions.append(positions[-1]+.001*vv);rows.append(np.r_[vv,aa,j]);v,a=vv,aa;previous_heading=heading
    xy=np.asarray(positions);vv=np.diff(xy,axis=0)/.001;aa=np.diff(np.vstack([np.zeros(2),vv]),axis=0)/.001
    jj=np.diff(np.vstack([np.zeros(2),aa]),axis=0)/.001
    peaks=[float(np.linalg.norm(q,axis=1).max()) for q in (vv,aa,jj)]
    assert np.all(np.asarray(peaks)<=[100.000001,2000.00001,62500.01]),peaks
    return dict(backup_cases=len(backups),maximum_backup_steps=max(r['steps'] for r in backups),
        backup_peaks=maximum.tolist(),raw_xy_transition_peaks=peaks,transition_ticks=len(rows),projector_counts=projector.counts,
        backups=backups,success=True),np.asarray(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',default='turn_aware_execution_probe');args=ap.parse_args()
    torch.set_num_threads(1);root=OUT/'all34_dominance_research';stage=root/args.stage;stage.mkdir(exist_ok=False)
    source=[Path(__file__),OUT/'turn_aware_projector.py',OUT/'turn_release_certificate.md',OUT/'probe_all34_feed_boundary.py']
    (stage/'source').mkdir()
    for p in source:shutil.copy2(p,stage/'source'/p.name)
    write_json(stage/'registration.json',dict(kind='explicit_common_executor_research',wall_budget_s=900,ppo_updates=0,
        physical_limits=[100,2000,62500],tolerance_mm=1.25,action_period_ms=20,action_ranges_unchanged=True,
        safety_first=True,adoption='Local diagnosis only. Both Joint and Fixed use the same experimental executor. No method adoption until full success and two-resolution validation.',
        stop='Abort all local experiments upon a safety-check failure or physical-constraint violation. Retain geometric failures.',
        source_sha256={str(p):sha(p) for p in source}))
    start=time.perf_counter()
    try:safety,rows=safety_checks(start+180)
    except Exception as exc:
        write_json(stage/'receipt.json',dict(goal_complete=False,status='safety_check_failed',error=f'{type(exc).__name__}: {exc}',wall_s=time.perf_counter()-start))
        raise
    write_json(stage/'safety_checks.json',safety);np.savez_compressed(stage/'transition_check.npz',rows=rows)
    print('TURN_SAFETY_PASS',safety['backup_cases'],safety['maximum_backup_steps'],safety['raw_xy_transition_peaks'],flush=True)
    with (root/'selected_causal_contexts/contexts.pkl').open('rb') as f:joint=pickle.load(f)
    with (OUT/'actual_fixed_corner_contexts/contexts.pkl').open('rb') as f:fixed=pickle.load(f)
    records=[]
    for method,contexts,modes in [('Joint',[joint[i] for i in (0,1,2,6,11,22,33)],['original','max_feed','shallow_max_feed']),
                                  ('Fixed',fixed,['original','max_feed'])]:
        for c in contexts:
            metadata=torch.load(c['model'],map_location='cpu',weights_only=False)
            def factory(reference):
                env=environment_factory(reference)
                if c.get('fixed_d') is not None:env.fixed_d=c['fixed_d']
                return env
            local.environment_factory=factory
            for mode in modes:
                if time.perf_counter()>start+860:break
                label=method+'_'+c['corner']['id']+'_'+mode
                r=local.rollout(metadata,c,mode,stage/(label+'.npz'),start+880);r['method']=method
                records.append(r);write_json(stage/'results.json',dict(records=records,goal_complete=False,wall_s=time.perf_counter()-start))
                print('TURN_EXECUTOR',method,r['corner'],mode,r['context_pass'],'deficit',r['speed_deficit_mm'],'distance',r['maximum_local_distance_mm'],'error',r['error'],flush=True)
                if r['raw_xy_vector_peaks'] and np.any(np.asarray(r['raw_xy_vector_peaks'])>np.array([100.,2000.,62500.])*1.00001):
                    raise ValueError('Physical-bound violation: stop entire executor candidate')
    write_json(stage/'receipt.json',dict(goal_complete=False,status='local_executor_diagnosis_complete',ppo_updates=0,
        records=len(records),wall_s=time.perf_counter()-start,complete_lap_not_tested=True))

if __name__=='__main__':main()
