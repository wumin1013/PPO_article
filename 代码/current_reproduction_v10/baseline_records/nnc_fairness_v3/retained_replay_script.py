"""Common feasibility check of the preserved submission NNC on current cases.

Keeps the original actor and action execution. A failed prefix is never assigned
a complete time. The fixed post-failure observation interval tests whether a
violation is confined to initialization; it does not restore eligibility.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parent
WORK=Path('D:/PPO_article_work')
OLD=WORK/'论文项目/revision_20260913_restart_30m_v1'
ARCHIVE=WORK/'论文项目/revision_20260908/historical/source/PPO_project'
MODEL=WORK/'论文项目/revision_20260908/historical/baseline_policy/best_model.pth'
CONFIG=MODEL.with_name('baseline_policy.yaml')
BINDING=WORK/'论文项目/joint_policy_20260915_v1/paper_evidence_review/verified_results/lowdim_original_revision/selected923202_revision/paper_result_binding.json'
OUT=ROOT/'experimental_supplement_20260923/original_nnc_current_cases'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def worker(case):
    import numpy as np
    import torch
    import yaml
    sys.path[:0]=[str(ARCHIVE),str(WORK/'论文项目/revision_20260913_experiments_v1'),str(OLD)]
    from tools import phase32_export_best_trajectories as original
    from nnc_planar_environment import OriginalNNCPlanarEnv, verify_archived_environment
    from nnc_geometry_index import install
    from nnc_scan_adapter import memoized_scan
    verify_archived_environment()
    assert Path(original.__file__).resolve()==ARCHIVE/'tools/phase32_export_best_trajectories.py'
    torch.set_num_threads(1)
    torch.manual_seed(42);np.random.seed(42)
    protocol=read(OUT/'registration.json')
    assert all(sha(p)==h for p,h in protocol['source_sha256'].items())
    cfg=yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    cfg['environment'].update(epsilon=2.5,interpolation_period=.001,max_steps=1200000)
    points=np.loadtxt(read(BINDING)['cases'][case]['reference'],delimiter=',',skiprows=1)
    folder=OUT/case;folder.mkdir(exist_ok=False)
    write(folder/'effective_config.json',cfg)
    # Proven exact geometry acceleration; no change to scalar formulas or ties.
    OriginalNNCPlanarEnv._scan_for_next_turn=memoized_scan(OriginalNNCPlanarEnv._scan_for_next_turn)
    install(OriginalNNCPlanarEnv)
    e=cfg['environment']
    env=OriginalNNCPlanarEnv(device=torch.device('cpu'),epsilon=2.5,interpolation_period=.001,
        **cfg['kinematic_constraints'],Pm=[p.copy() for p in points],max_steps=1200000,
        lookahead_points=e.get('lookahead_points',5),lookahead_obs_enabled=e.get('lookahead_obs_enabled',True),
        lookahead_obs_scales=e.get('lookahead_obs_scales',[1.]),reward_weights=cfg.get('reward_weights',{}),
        curvature_observation=e.get('curvature_observation'),return_normalized_obs=True,
        path_name=case,disable_kcm=True)
    env.debug_collector.dump_dir=folder/'diagnostic_dumps'
    agent=original._make_agent(cfg,env,torch.device('cpu'))
    payload=torch.load(MODEL,map_location='cpu',weights_only=False)
    assert tuple(payload['actor']['shared_fc1.weight'].shape)==(256,14)
    agent.actor.load_state_dict(payload['actor'],strict=True)
    agent.critic.load_state_dict(payload['critic'],strict=True)
    agent.actor.eval();agent.critic.eval()
    obs=env.reset(random_start=False)
    assert obs.shape==(14,) and env.action_space_dim==2 and not cfg['experiment']['enable_kcm']
    positions=[np.array(env.current_position,dtype=float).copy()]
    observations=[];actions=[];speeds=[];accels=[];jerks=[];distances=[]
    vprev=np.zeros(2);aprev=np.zeros(2);first=None;done=False;info={};start=time.perf_counter()
    dt=.001
    # Exact full-polyline Euclidean distance at each recorded endpoint.
    p0=points[:-1];dv=np.diff(points,axis=0);den=np.sum(dv*dv,axis=1)
    def distance(x):
        t=np.clip(np.sum((x-p0)*dv,axis=1)/np.maximum(den,1e-30),0,1)
        return float(np.linalg.norm(x-p0-t[:,None]*dv,axis=1).min())
    with torch.no_grad():
        for tick in range(1,1200001):
            if time.perf_counter()-start>protocol['worker_deadline_s']:break
            observed=np.asarray(obs,dtype=float).copy()
            mu,_=agent.actor(torch.as_tensor(obs,dtype=torch.float,device=agent.device).reshape(1,14))
            action=np.clip(mu.squeeze(0).cpu().numpy().astype(float),env.action_space.low,env.action_space.high)
            obs,reward,done,info=env.step(action)
            x=np.array(env.current_position,dtype=float).copy()
            v=(x-positions[-1])/dt;a=(v-vprev)/dt;j=(a-aprev)/dt
            norms=[float(np.linalg.norm(z)) for z in [v,a,j]];dist=distance(x)
            positions.append(x);observations.append(observed);actions.append(action)
            speeds.append(norms[0]);accels.append(norms[1]);jerks.append(norms[2]);distances.append(dist)
            reasons=[name for name,value,limit in zip(['speed','acceleration','jerk','corridor'],norms+[dist],[100.,2000.,62500.,1.25]) if value>limit+max(1e-7,1e-8*limit)]
            if reasons and first is None:first=dict(tick=tick,time_s=tick*dt,reasons=reasons,values=dict(zip(['v','a','j','distance'],norms+[dist])))
            vprev=v;aprev=a
            if done or (first is not None and tick>=1000):break
    n=len(actions)
    np.savez_compressed(folder/'recorded_prefix.npz',positions=np.array(positions),timestamps=np.arange(n+1)*dt,
        observations=np.array(observations),actions=np.array(actions),speed=np.array(speeds),acceleration=np.array(accels),jerk=np.array(jerks),distance=np.array(distances),reference=points)
    peak=lambda z:float(max(z)) if z else None
    result=dict(case=case,method='Preserved submission NNC (planar adaptation)',checkpoint_sha256=sha(MODEL),
        policy_inputs=14,policy_outputs=2,policy_period_s=dt,physical_interval_s=dt,
        common_complete_success=False,complete_time_s=None,first_common_violation=first,
        observed_prefix_time_s=n*dt,recorded_ticks=n,environment_done=bool(done),environment_reason=str(info.get('done_reason','')),
        prefix_peaks=dict(speed_mm_s=peak(speeds),acceleration_mm_s2=peak(accels),jerk_mm_s3=peak(jerks),euclidean_distance_mm=peak(distances)),
        post_start_20ms_peaks=dict(speed_mm_s=peak(speeds[20:]),acceleration_mm_s2=peak(accels[20:]),jerk_mm_s3=peak(jerks[20:])),
        prefix_peak_relative_jerk_exceedance=max(0.,peak(jerks)/62500.-1.) if jerks else None,
        full_task_peaks=False,complete_time_comparison_eligible=False,new_training_updates=0,
        stopping_rule='After first infeasibility, retain up to1s total prefix or original environment termination; no completion time or full-route peak inferred.',
        wall_s=time.perf_counter()-start,trace_sha256=sha(folder/'recorded_prefix.npz'))
    write(folder/'result.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=['original','dense']);args=p.parse_args()
    if args.worker:return worker(args.worker)
    OUT.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),BINDING,MODEL,CONFIG,ARCHIVE/'src/environment/cnc_env.py',ARCHIVE/'tools/phase32_export_best_trajectories.py',
        WORK/'论文项目/revision_20260913_experiments_v1/nnc_planar_environment.py',OLD/'nnc_geometry_index.py',OLD/'nnc_scan_adapter.py']
    sources += [Path(v['reference']) for v in read(BINDING)['cases'].values()]
    hashes={str(p):sha(p) for p in sources}
    assert sha(MODEL)=='24d263a0e07fe471e7995ebae2d0a0ee7a2ee03e2feec80ea50aa5b94b0cfe6b'
    write(OUT/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=hashes,
        evidence_gap='Original submission NNC has no common-feasibility result on either current reference.',
        plan='Preserve actor,14D observation,2D returned action and original1ms execution; retain existing planar adapter and exact geometry acceleration. Check velocity/acceleration/jerk fromXY starting atrest. Evaluate both current cases unconditionally.',
        worker_deadline_s=780,wall_budget_s=1800,new_training_updates=0,
        adoption='A violation rejects common complete feasibility; no failed-prefix time enters efficiency. No claim of exact reproduction of the external published NNC.',
        stop='Both cases; fatal error; or1800wall seconds. After detected failure retain up to1s diagnostic prefix unless original environment terminates earlier. If no violation and no completion by deadline, classify unresolved, not failed.'))
    start=time.perf_counter();records=[]
    for case in ['original','dense']:
        with (OUT/f'{case}.log').open('w',encoding='utf-8') as log:
            try:
                r=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--worker',case],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=850)
                records.append(dict(case=case,exit_code=r.returncode))
            except subprocess.TimeoutExpired:records.append(dict(case=case,status='timeout'))
    write(OUT/'receipt.json',dict(records=records,wall_s=time.perf_counter()-start,sources_unchanged=all(sha(p)==h for p,h in hashes.items()),new_training_updates=0))
    print('NNC_CHECK',records,flush=True)

if __name__=='__main__':main()
