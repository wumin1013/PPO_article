"""Geometry-only reconstruction from existing witnessed states; no simulation."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import json
import math
from pathlib import Path
import numpy as np
import torch
from evaluate_historical_dense import ROOT,BASE,sha,write
from compact_farvector_runtime import FarVectorRuntime
from compact_previewcue_runtime import PreviewCueRuntime


def set_physical(env,row):
    env.position=row[:2].copy()
    env.velocity=row[2:4].copy()
    env.acceleration=row[4:6].copy()
    if np.linalg.norm(row[8:10])>1e-12:
        env.heading=math.atan2(row[9],row[8])


def main():
    output=ROOT/'farvector_saved_data'
    output.mkdir(exist_ok=False)
    stage=ROOT/'direct14_previewcue_ppo_seed922302_round13'
    reports=[]
    for case in ['original','dense']:
        trace=stage/f'eval_0024_{case}.npz'
        gates=stage/f'eval_0024_{case}_gates.json'
        with np.load(trace) as d:
            ref=d['reference'];rows=d['rows'];expected=d['observations'];actions=d['actions']
        crossings=np.array([v['command_segment'] for v in json.loads(gates.read_text(encoding='utf-8'))])
        runtime=FarVectorRuntime(ref)
        before=[];after=[]
        for i in range(len(expected)):
            tick=i*20
            if tick:
                set_physical(runtime.env,rows[tick-1])
            runtime.env.tracker.frontier=int(np.searchsorted(crossings,tick,side='left'))
            before.append(PreviewCueRuntime.observation(runtime));after.append(runtime.observation())
        before=np.asarray(before);after=np.asarray(after)
        error=float(np.max(abs(before-expected)))
        assert error<2e-6,(case,error)
        retained=[i for i in range(14) if i!=5]
        assert np.array_equal(before[:,retained],after[:,retained])
        np.savez_compressed(output/f'{case}_retained.npz',observations=after,original_observations=expected,actions=actions)
        reports.append(dict(name=case,samples=len(expected),maximum_reconstruction_error=error,
            trace_sha256=sha(trace),gates_sha256=sha(gates),scope='Exposed development trajectory'))
        print(case,len(expected),error,flush=True)
    targets_path=ROOT/'direct14_joint_response_probe_round14/offline_target_feasibility.json'
    targets=json.loads(targets_path.read_text(encoding='utf-8'))['selected']
    contexts=torch.load(stage/'all34_contexts.pt',map_location='cpu',weights_only=False)
    ref=np.loadtxt(BASE/'paper_evidence_review/natural_closed_case/reference.csv',delimiter=',',skiprows=1)
    with np.load(ROOT/'direct14_local_target_fit_round15/supervision.npz') as data:
        expected=data['observation'][:34*48]
    before=[];after=[];actions=[];source_hashes={}
    for context,selection in zip(contexts,targets):
        assert context['id']==selection['id']
        trace=ROOT/'direct14_joint_response_probe_round14'/f"response_{selection['response']:02d}"/(context['id']+'.npz')
        source_hashes[str(trace)]=sha(trace)
        with np.load(trace) as data:
            rows=data['rows'];held=data['actions']
        assert len(rows)==960 and len(held)==48
        runtime=FarVectorRuntime(ref);runtime.restore(context['state'])
        for i in range(48):
            if i:
                for row in rows[(i-1)*20:i*20]:
                    runtime.env.tracker.update(row[:2])
                set_physical(runtime.env,rows[i*20-1])
            before.append(PreviewCueRuntime.observation(runtime));after.append(runtime.observation());actions.append(held[i])
    before=np.asarray(before);after=np.asarray(after)
    error=float(np.max(abs(before-expected)))
    assert error<2e-6,error
    assert np.array_equal(before[:,[i for i in range(14) if i!=5]],after[:,[i for i in range(14) if i!=5]])
    np.savez_compressed(output/'local_optimized.npz',observations=after,original_observations=expected,actions=np.asarray(actions))
    reports.append(dict(name='local_optimized',samples=len(after),maximum_reconstruction_error=error,
        source_sha256=source_hashes,scope='Individually selected valid local probes, not a full policy'))
    write(output/'audit.json',dict(sources=reports,physics_ticks_new=0,training_updates_new=0,
        observation_dimension=14,policy_memory_scalars=0,changed_feature_index=5,
        source_sha256={str(p):sha(p) for p in [Path(__file__),ROOT/'compact_farvector_runtime.py',targets_path]},
        scope='Candidate encoder data only; no new learned model or complete success claim'))
    print('local',len(after),error,flush=True)


if __name__=='__main__':
    main()
