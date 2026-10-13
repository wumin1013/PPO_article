"""Bounded local physics probes from verified causal full-route states."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time,pickle,argparse,shutil
import numpy as np
import torch
OUT=Path(__file__).resolve().parent
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
sys.path[:0]=[str(BASE),str(OUT)]
from deep_joint_credit_policy import load_credit_policy
from learned_action_pulse_policy import environment_factory
from arc_cruise_policy import build_policy
from cache_corner_contexts import restore_memory
from train_actual_context_feedback import local_distance
from natural_case_metrics import ordered_projection
from study_io import write_json,sha

class BoundaryProbe:
    def __init__(self,base,mode):self.base=base;self.mode=mode
    def bind(self,env):return self.base.bind(env)
    def act(self,observation,deterministic=True):
        latent=self.base.act(observation,deterministic)[0].astype(float)
        if self.mode=='max_feed':latent[1]=20.
        elif self.mode.startswith('max_feed_preview_'):
            tokens=self.mode.split('_');preview=float(tokens[3]);scale=float(tokens[5])
            latent[1]=20.;latent[2]=np.arctanh(np.clip((preview-6.4)/5.6,-.999999999,.999999999))
            latent[0]=np.arctanh(np.clip(np.tanh(latent[0])*scale,-.999999999,.999999999))
        elif self.mode=='shallow_max_feed':
            x=np.asarray(observation);d=x[42:62].reshape(10,2)
            a=np.unwrap(np.arctan2(d[:,1],d[:,0]));bend=np.max(abs(a-a[0])*x[62:72])
            heading=abs(np.arctan2(x[7],x[6]))
            q=np.clip((1.15-max(bend,heading))/.25,0.,1.);support=q*q*(3.-2.*q)
            latent[1]+=20.*support
        elif self.mode!='original':raise ValueError(self.mode)
        return latent,0.,0.

def rollout(metadata,context,mode,file,deadline,policy_factory=BoundaryProbe):
    policy=policy_factory(build_policy(load_credit_policy(metadata),context['source_spec']),mode)
    env=environment_factory(context['reference']);policy.bind(env);env.restore(context['state']);restore_memory(policy,context['memory'])
    initial=env.state();c=context['corner'];k=c['reference_vertex_index'];vertex=context['reference'][k]
    outgoing=context['reference'][k+1]-vertex;outgoing/=np.linalg.norm(outgoing)
    chunks=[];actions=[];obs=[];hit=False;error=None
    try:
        while not env.done and env.ticks-initial['ticks']<2500:
            if time.perf_counter()>=deadline:raise TimeoutError('Feed-boundary probe budget')
            x=env.observation();latent=policy.act(x,True)[0];assert not np.any(env.pending_pulse)
            _,_,_,info=env.step(latent);rr=info['rows'];crossing=np.flatnonzero((rr[:,:2]-vertex)@outgoing>=50.)
            if len(crossing):rr=rr[:crossing[0]+1];hit=True
            chunks.append(rr);actions.append(info['held_action']);obs.append(x)
            if hit:break
    except Exception as exc:error=f'{type(exc).__name__}: {exc}'
    rows=np.vstack(chunks) if chunks else np.empty((0,13));peaks=None;maximum=None;deficit=None;window_complete=False;minimum=None;wide_deficit=None;total_deficit=None
    if len(rows):
        v=np.diff(np.vstack([initial['position'],rows[:,:2]]),axis=0)/.001
        a=np.diff(np.vstack([initial['velocity'],v]),axis=0)/.001
        j=np.diff(np.vstack([initial['acceleration'],a]),axis=0)/.001
        peaks=[float(np.linalg.norm(q,axis=1).max()) for q in (v,a,j)]
        maximum=float(local_distance(rows[:,:2],context).max())
        p=ordered_projection(context['reference'],rows[:,:2]);lo=c['reference_s_mm']-10.;hi=c['reference_s_mm']+10.
        mask=(p>=lo)&(p<=hi);speed=np.linalg.norm(rows[:,2:4],axis=1)
        window_complete=bool(p.min()<lo and p.max()>hi)
        wide=abs(p-c['reference_s_mm'])<=30.
        wide_deficit=float(np.maximum(100.-speed[wide],0.).sum()*.001)
        total_deficit=float(np.maximum(100.-speed,0.).sum()*.001)
        if np.any(mask):
            deficit=float(np.maximum(100.-speed[mask],0.).sum()*.001);minimum=float(speed[mask].min())
    passed=bool(error is None and hit and env.tracker.inside and peaks and np.all(np.array(peaks)<=np.array([100,2000,62500])*1.00001))
    np.savez_compressed(file,rows=rows,actions=np.array(actions),observations=np.array(obs),initial_position=initial['position'])
    result=dict(corner=c['id'],mode=mode,context_pass=passed,full_lap_success_not_tested=True,
        complete_window=window_complete,speed_deficit_mm=deficit,minimum_speed_mm_s=minimum,
        wide30_speed_deficit_mm=wide_deficit,total_local_speed_deficit_mm=total_deficit,
        local_elapsed_s=len(rows)*.001,maximum_local_distance_mm=maximum,raw_xy_vector_peaks=peaks,
        reason=env.reason,error=error,start_decision=initial['decisions'],trace_sha256=sha(file))
    write_json(file.with_suffix('.json'),result);return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');ap.add_argument('--stage',default='feed_boundary_probe')
    ap.add_argument('--corners',default='0,1,2,6');ap.add_argument('--modes',default='original,max_feed');ap.add_argument('--budget',type=int,default=600);args=ap.parse_args()
    torch.set_num_threads(1);root=OUT/'all34_dominance_research';stage=root/args.stage;stage.mkdir(exist_ok=False)
    shutil.copy2(Path(__file__),stage/'source.py')
    cache=root/'selected_causal_contexts'/('contexts.partial.pkl' if args.partial else 'contexts.pkl')
    with cache.open('rb') as f:contexts=pickle.load(f)
    ids=['corner_'+str(int(x)).zfill(2) for x in args.corners.split(',')];contexts=[next(c for c in contexts if c['corner']['id']==cid) for cid in ids]
    write_json(stage/'registration.json',dict(kind='diagnostic_local_feed_intervention',wall_budget_s=args.budget,ppo_updates=0,
        corners=ids,modes=args.modes.split(','),source_cache_sha256=sha(cache),script_sha256=sha(Path(__file__)),
        adopt='Diagnose physical headroom only. No full-lap/model adoption allowed from local probes.',
        stop='Per-context failure stops that rollout. Round stops before wall budget. Keep failures and original controls.',
        unchanged=['1.25mm one-sided Euclidean corridor','100/2000/62500 planar vector limits','20ms ZOH','registered +/-10mm windows']))
    start=time.perf_counter();metadata=torch.load(contexts[0]['model'],map_location='cpu',weights_only=False);records=[]
    for c in contexts:
        for mode in args.modes.split(','):
            if time.perf_counter()>start+args.budget-20:break
            r=rollout(metadata,c,mode,stage/(c['corner']['id']+'_'+mode+'.npz'),start+args.budget-10);records.append(r)
            write_json(stage/'results.json',dict(records=records,goal_complete=False,wall_s=time.perf_counter()-start,ppo_updates=0))
            print('FEED_PROBE',r['corner'],mode,'pass',r['context_pass'],'deficit',r['speed_deficit_mm'],'distance',r['maximum_local_distance_mm'],'wall',time.perf_counter()-start,flush=True)
    write_json(stage/'receipt.json',dict(goal_complete=False,status='diagnostic_only',records=len(records),wall_s=time.perf_counter()-start,ppo_updates=0))

if __name__=='__main__':main()
