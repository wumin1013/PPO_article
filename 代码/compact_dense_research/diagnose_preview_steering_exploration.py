"""Geometry-only saved-state check of possible correlated exploration; no simulation."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import json
import numpy as np
from evaluate_historical_dense import ROOT,write,sha
from compact_previewcue_runtime import PreviewCueRuntime


def main():
    stage=ROOT/'direct14_previewcue_fit_seed922302_round11';report={}
    for case in ['original','dense']:
        trace=stage/f'eval_12000_{case}.npz';gates=stage/f'eval_12000_{case}_gates.json'
        with np.load(trace) as d:ref=d['reference'];rows=d['rows'];obs=d['observations'];actions=d['actions']
        events=json.loads(gates.read_text(encoding='utf8'));crossed=np.array([x['command_segment'] for x in events])
        ids=np.flatnonzero((abs(obs[:,-1])>.005)|(actions[:,1]<99)|(abs(actions[:,0])>.001))
        if len(ids)>2000:ids=ids[np.unique(np.linspace(0,len(ids)-1,2000).astype(int))]
        env=PreviewCueRuntime(ref).env;actual=[];approx=[]
        for i in ids:
            tick=20*i
            env.position=rows[tick-1,:2].copy() if tick else ref[0].copy()
            env.tracker.frontier=int(np.searchsorted(crossed,tick,side='left'))
            held=actions[i].copy();d=held[2];beta=[]
            for dd in [d-.001,d+.001]:
                env.held=held.copy();env.held[2]=dd
                delta=env.guidance_target()-env.position;beta.append(np.arctan2(delta[1],delta[0]))
            derivative=float(np.arctan2(np.sin(beta[1]-beta[0]),np.cos(beta[1]-beta[0]))/.002)
            p2,p6,p15=obs[i,7:9]*15,obs[i,9:11]*15,obs[i,11:13]*15
            slope=(p6-p2)/4 if d<=6 else (p15-p6)/9
            point=p2+(d-2)*slope if d<=6 else p6+(d-6)*slope
            estimate=float((point[0]*slope[1]-point[1]*slope[0])/max(point@point,1e-8))
            actual.append(derivative);approx.append(estimate)
        actual=np.array(actual);approx=np.array(approx)
        independent=(np.pi/2*.0017)**2+(5.6*.02*actual)**2
        result={}
        for strength in [.5,1.]:
            coefficient=np.clip(-strength*5.6/(np.pi/2)*approx,-3,3)
            variance=(np.pi/2*.0017)**2+(.02*(np.pi/2*coefficient+5.6*actual))**2
            result[str(strength)]=dict(mean_linearized_heading_variance_ratio=float(variance.mean()/independent.mean()),
                fraction_states_with_larger_variance=float((variance>independent+1e-14).mean()),
                p95_target_heading_std_deg=float(np.rad2deg(np.sqrt(variance)).__array__().take(np.argsort(variance)[int(.95*(len(variance)-1))])))
        report[case]=dict(states=len(ids),independent_p95_heading_std_deg=float(np.percentile(np.rad2deg(np.sqrt(independent)),95)),
            correlation=float(np.corrcoef(actual,approx)[0,1]),estimated_compensation=result,
            trace_sha256=sha(trace),gate_sha256=sha(gates))
    write(ROOT/'preview_steering_exploration_saved_state_audit.json',dict(cases=report,new_physics_ticks=0,new_training_updates=0,
        scope='Local linearized guide-heading variation only; this does not establish safe physical rollouts or faster policies'))
    print(json.dumps(report))


if __name__=='__main__':main()
