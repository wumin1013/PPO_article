"""Bounded offline compact-support preview fit, with strict physical screening."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse
import json
import multiprocessing as mp
import shutil
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
from evaluate_historical_dense import ROOT,write,sha,verify_bound_inputs
from probe_paired_local_repair import INITIAL
from train_mixed_resolution_ppo import policy_from
from localized_preview_head import LocalizedPreviewPolicy,physics_worker


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);args=p.parse_args()
    assert Path(args.name).name==args.name;verify_bound_inputs()
    import numpy as np
    import torch
    torch.set_num_threads(1)
    stage=ROOT/args.name;stage.mkdir(exist_ok=False);(stage/'source').mkdir()
    started=time.perf_counter();deadline=started+1680.
    sources=[Path(__file__),ROOT/'localized_preview_head.py',ROOT/'train_mixed_resolution_ppo.py']
    hashes={str(s):sha(s) for s in sources}
    for s in sources:shutil.copy2(s,stage/'source'/s.name)
    radii=[.3,.6,1.2,2.4]
    write(stage/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),kind='Offline localized preview-head fit;zeroPPO',
        initial=str(INITIAL),initial_sha256=sha(INITIAL),source_sha256=hashes,wall_budget_s=1800,deadline_s=1680,
        support_radii=radii,basis_count=96,ridge_penalty=1e-4,inputs=14,outputs=3,policy_memory_scalars=0,
        evidence_gap='Global nonlinear preview corrections disturb dense states;local support may preserve unmodified regions exactly.',
        procedure='Farthest-point centers selected deterministically from positive teacher-current-state inputs. Fit96linear weights in a smooth compact-support actor using physical local targets and negative dense/other-corner anchors.',
        model_scope='Centers are static fitted model parameters,not evolving policy history. No route coordinates,case IDs,corner IDs or clocks enter inference.',
        adoption='Both dense complete,all34local valid+budget,then both fulloriginal and joint protocol. Local fit never establishes full success.',
        stop='Four support scales or30minute deadline. PPO only if physical results support this architecture.',new_ppo_updates=0,goal_complete=False,method_frozen=False))
    saved=torch.load(INITIAL,map_location='cpu',weights_only=False);base=policy_from(saved,922302)
    pieces=[];targets=[];weights=[];bindings={}
    def add(path,x,y,weight):
        x=np.asarray(x,np.float32)
        with torch.no_grad():b=base.actor(torch.tensor(x)).numpy()
        pieces.append(x);targets.append(np.asarray(y,np.float32)-b[:,2]);weights.extend([weight]*len(x));bindings[str(path)]=sha(path)
    teacher=ROOT/'direct14_preview_only_head_round41_checked'
    for key in ['1ms','half_ms']:
        path=teacher/f'teacher_dataset_{key}.npz'
        with np.load(path) as d:add(path,d['observations'],d['targets'][:,2],1.)
    selection=json.loads((ROOT/'preview_only_local_targets_round41/result.json').read_text())['selected']
    for item in selection:
        for filename in item['selected']['traces']:
            path=Path(filename)
            with np.load(path) as d:add(path,d['observations'],d['latents'][:,2],2.)
    for path in [INITIAL.parent/'dense.npz',ROOT/'direct14_head_bias_search_round29_dense_half_08/trace.npz']:
        with np.load(path) as d:x=d['observations']
        with torch.no_grad():b=base.actor(torch.tensor(x)).numpy()
        add(path,x,b[:,2],4.)
    x=np.vstack(pieces);target=np.concatenate(targets);weight=np.array(weights);scale=x.std(0).clip(.03)
    active=x[target>.001];normalized=active/scale
    chosen=[int(np.argmax(np.linalg.norm(normalized,axis=1)))];distance=((normalized-normalized[chosen[0]])**2).sum(1)
    for _ in range(95):
        index=int(np.argmax(distance));chosen.append(index);distance=np.minimum(distance,((normalized-normalized[index])**2).sum(1))
    centers=active[chosen];assert len(np.unique(centers,axis=0))==96
    np.savez_compressed(stage/'fitting_data.npz',observations=x,target_delta=target,weights=weight,centers=centers,scale=scale)
    write(stage/'data_binding.json',dict(source_sha256=bindings,dataset_sha256=sha(stage/'fitting_data.npz')))
    transformed=np.arctanh(np.clip(target/.4,-.999,.999));tensor_x=torch.tensor(x);fit_records=[]
    for i,radius in enumerate(radii):
        policy=LocalizedPreviewPolicy();policy.initialize(base)
        with torch.no_grad():
            policy.actor.centers.copy_(torch.tensor(centers));policy.actor.feature_scale.copy_(torch.tensor(scale));policy.actor.radius.fill_(radius)
            assert torch.equal(policy.actor(tensor_x),base.actor(tensor_x))
            features=policy.actor.features(tensor_x).numpy().astype(float)
        lhs=features.T@(features*weight[:,None])+1e-4*np.eye(96);rhs=features.T@(transformed*weight)
        coefficients=np.linalg.solve(lhs,rhs)
        with torch.no_grad():
            policy.actor.correction.weight.copy_(torch.tensor(coefficients[None],dtype=torch.float32))
            prediction=policy.actor(tensor_x);assert torch.equal(prediction[:,:2],base.actor(tensor_x)[:,:2])
            delta=(prediction[:,2]-base.actor(tensor_x)[:,2]).numpy()
        fit_records.append(dict(index=i,radius=radius,weighted_rmse_mm=float(np.sqrt(np.average((delta-target)**2,weights=weight))*5.6),
            dense_max_preview_perturbation_mm=float(abs(delta[np.array(weights)==4.]).max()*5.6),nonzero_centers=96))
        payload=dict(saved);payload.update(policy=policy.state_dict(),updates=i,interface=[14,3,922302],policy_class='LocalizedPreviewPolicy',
            runtime_class='FarVectorRuntime',local_basis_count=96,new_ppo_updates=0,training_kind='Offline linear fit of compact-support features',
            checkpoint_index_kind='support_scale_selection_index',dataset_sha256=sha(stage/'fitting_data.npz'))
        torch.save(payload,stage/f'checkpoint_{i:04d}.pt')
    write(stage/'fitting_audit.json',fit_records);print('LOCALIZED_FIT',fit_records,flush=True)
    ctx=mp.get_context('spawn');worker_args=SimpleNamespace(seed=922302,initial=stage/'checkpoint_0000.pt')
    connections=[];processes=[];records=[];error=None
    def receive(c):
        while time.perf_counter()<deadline:
            if c.poll(1.):
                r=c.recv()
                if 'success' not in r and isinstance(r.get('error'),str):raise RuntimeError(r['error'])
                return r
        raise TimeoutError('Registered30minute limit')
    try:
        for dt in [.001,.0005]:
            parent,child=ctx.Pipe();proc=ctx.Process(target=physics_worker,args=(child,dt,worker_args,stage,deadline))
            proc.start();child.close();processes.append(proc);connections.append(parent)
        for c in connections:receive(c)
        for i in range(len(radii)):
            if time.perf_counter()>deadline-90:break
            checkpoint=stage/f'checkpoint_{i:04d}.pt'
            for c in connections:c.send(dict(kind='evaluate',case='dense',checkpoint=str(checkpoint)))
            pair=[receive(c) for c in connections];record=dict(update=i,dense=pair)
            print('LOCALIZED_DENSE',i,[(r['success'],r['complete_time_s'],r['elapsed_prefix_s']) for r in pair],flush=True)
            if all(r['success'] for r in pair):
                for c in connections:c.send(dict(kind='local_check',checkpoint=str(checkpoint)))
                record['local']=[receive(c) for c in connections]
                print('LOCALIZED_ALL34',i,[(r['valid_count'],r['budget_count']) for r in record['local']],flush=True)
                if all(r['valid_count']==34 and r['budget_count']==34 for r in record['local']) and time.perf_counter()<deadline-400:
                    for c in connections:c.send(dict(kind='evaluate',case='original',checkpoint=str(checkpoint)))
                    record['original']=[receive(c) for c in connections]
                    print('LOCALIZED_ORIGINAL',i,[(r['success'],r['complete_time_s']) for r in record['original']],flush=True)
            records.append(record);write(stage/'evaluations.json',records)
    except BaseException:error=traceback.format_exc();print(error,flush=True)
    finally:
        for c in connections:
            try:c.send(dict(kind='stop'))
            except (BrokenPipeError,EOFError,OSError):pass
        for proc in processes:
            proc.join(timeout=5.)
            if proc.is_alive():proc.terminate();proc.join(timeout=5.)
        for c in connections:c.close()
        write(stage/'result.json',dict(evaluations=records,error=error,new_ppo_updates=0,wall_s=time.perf_counter()-started,goal_complete=False,method_frozen=False))
        write(stage/'receipt.json',dict(status='finished' if error is None else 'error',wall_s=time.perf_counter()-started,sources_unchanged=all(sha(p)==h for p,h in hashes.items())))


if __name__=='__main__':main()
