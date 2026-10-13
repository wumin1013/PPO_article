"""Offline fit to physically verified actual-arrival teachers;zeroPPO."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from evaluate_historical_dense import ROOT,BASE,write,sha,verify_bound_inputs
from localized_preview_head import LocalizedPreviewPolicy
from localized_joint_correction import LocalizedJointPolicy
from compact_farvector_runtime import FarVectorRuntime
from observable_feedback_gate import modify,strength
from research_policy_loader import from_saved


def observed_progress(path,gates,dt):
    with np.load(path) as d:
        x=d['observations'];ref=d['reference'];rows=d['rows'];times=d['decision_times_s'] if 'decision_times_s' in d else np.arange(len(x))*.02
    tick=np.rint(times/dt).astype(int);env=FarVectorRuntime(ref);c=env.corridor
    events=json.loads(gates.read_text(encoding='utf-8'));crossing=np.array([v['command_segment'] for v in events])
    frontier=np.searchsorted(crossing,tick,side='left');segment=np.minimum(frontier,len(c.lengths)-1)
    pos=np.vstack([ref[0],rows[tick[1:]-1,:2]])
    along=np.clip(((pos-c.starts[segment])*c.tangents[segment]).sum(1),0,c.lengths[segment])
    return x,c.arcs[segment]+along


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True)
    p.add_argument('--replacement-teachers',type=Path);p.add_argument('--anchor-policy',type=Path)
    p.add_argument('--extra-feedback-teachers',type=Path,action='append',default=[])
    p.add_argument('--anchor-executions',type=Path);p.add_argument('--anchor-index',type=int,default=1)
    p.add_argument('--basis-count',type=int,default=128);args=p.parse_args()
    assert args.basis_count in [128,192,256]
    assert Path(args.name).name==args.name;verify_bound_inputs();torch.set_num_threads(1)
    started=time.perf_counter();stage=ROOT/(args.name+'_initialization');stage.mkdir(exist_ok=False);(stage/'source').mkdir()
    parent=ROOT/'direct14_localized_preview_fit_round49/checkpoint_0002.pt';teachers=ROOT/'direct14_actual_arrival_repair_round51'
    sources=[Path(__file__),ROOT/'localized_joint_correction.py',ROOT/'localized_preview_head.py',ROOT/'screen_saved_mixed_checkpoints.py',ROOT/'research_policy_loader.py']
    hashes={str(s):sha(s) for s in sources}
    for s in sources:shutil.copy2(s,stage/'source'/s.name)
    selected=json.loads((teachers/'result.json').read_text(encoding='utf-8'))['selected'];assert all(v['selected'] for v in selected)
    feedback_directories=[teachers]+[(ROOT/v).resolve() for v in args.extra_feedback_teachers]
    feedback_corner_ids=[v['id'] for directory in feedback_directories for v in json.loads((directory/'result.json').read_text(encoding='utf-8'))['selected']]
    if args.anchor_executions:
        assert args.anchor_policy and args.anchor_executions.is_dir()
        assert sha(args.anchor_policy)==sha(args.anchor_executions/f'checkpoint_{args.anchor_index:04d}.pt')
    radii=[.6,1.2,2.4]
    write(stage/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),kind='Offline local three-action fit from verified actual arrivals',
        parent=str(parent),parent_sha256=sha(parent),teacher_result_sha256=sha(teachers/'result.json'),source_sha256=hashes,
        support_radii=radii,basis_count=args.basis_count,ridge=1e-4,inputs=14,outputs=3,policy_memory_scalars=0,new_ppo_updates=0,
        replacement_teacher_directory=str(args.replacement_teachers) if args.replacement_teachers else None,
        retained_anchor_policy=str(args.anchor_policy) if args.anchor_policy else None,
        extra_feedback_directories=[str(v) for v in feedback_directories[1:]],
        anchor_execution_directory=str(args.anchor_executions) if args.anchor_executions else None,
        anchor_execution_index=args.anchor_index if args.anchor_executions else None,
        wall_budget_s=1800,screening_timeout_s=1740,
        evidence_gap=('Retained55candidate1 completes both cases at both resolutions but nominal corner30 exceeds the newly validated two-action budget; verified actual-arrival30teacher must be encoded without regressing other gates.' if args.extra_feedback_teachers else
                     ('Actual10/14feedback and18replacement pass both dt and five phases; fullpath18failure remains.'
                      if args.replacement_teachers else 'Actual10and14feedback targets passbothdtand5phases;must encode in one policy without disturbing existing successes.')),
        procedure=(f'Frozen49actor plus smooth{args.basis_count}basis3output correction. Positive actual teacher transitions '
                   'plus explicitly queried teacher commands on observed baseline states; negative anchors outside '
                   + '/'.join(feedback_corner_ids+(['corner_18'] if args.replacement_teachers else [])) + ' windows and dense. Each counterfactual teacher query uses its own registered source actor. No route ID is an input.'),
        adoption='Same fullpaired joint protocol,not local teachers or offline fit. No corner ID supplied at inference.',
        stop='Three fixed support sizes and physical screening,or30minute combinedwalllimit.',goal_complete=False,method_frozen=False))
    saved=torch.load(parent,map_location='cpu',weights_only=False);base=LocalizedPreviewPolicy();base.load_state_dict(saved['policy'])
    anchor=from_saved(torch.load(args.anchor_policy,map_location='cpu',weights_only=False)) if args.anchor_policy else base
    geometry=json.loads((BASE/'paper_evidence_review/natural_closed_case/geometry.json').read_text(encoding='utf-8'))
    critical_ids=list(feedback_corner_ids)
    pieces=[];targets=[];weights=[];bindings={};labels=[]
    def add(path,x,y,weight,label):
        x=np.asarray(x,np.float32)
        with torch.no_grad():b=base.actor(torch.tensor(x)).numpy()
        pieces.append(x);targets.append(np.asarray(y,np.float32)-b);weights.extend([weight]*len(x));labels.extend([label]*len(x));bindings[str(path)]=sha(path)
    for directory in feedback_directories:
        registration=json.loads((directory/'registration.json').read_text(encoding='utf-8'))
        source_checkpoint=(ROOT/registration['checkpoint']).resolve()
        assert sha(source_checkpoint)==registration['checkpoint_sha256']
        source_policy=from_saved(torch.load(source_checkpoint,map_location='cpu',weights_only=False))
        for path in [source_checkpoint,directory/'registration.json',directory/'result.json']:
            bindings[str(path)]=sha(path)
        selected=json.loads((directory/'result.json').read_text(encoding='utf-8'))['selected']
        for item in selected:
            chosen=item['selected'];assert chosen and len(chosen['traces'])==10
            steer,feed,gain=chosen['response']
            for name in chosen['traces']:
                path=Path(name)
                with np.load(path) as d:add(path,d['observations'],d['latents'],2.,'executed_teacher')
            for key in ['1ms','half_ms']:
                path=directory/key/f'{item["id"]}_r04_p00.npz'
                with np.load(path) as d:x=d['observations']
                with torch.no_grad():b=source_policy.actor(torch.tensor(x)).numpy()
                y=[]
                for row,z in zip(x,b):
                    value=modify(z,row,feed,gain);value[0]+=strength(row)*steer;y.append(value)
                add(path,x,y,2.,'counterfactual_teacher_query_at_observed_baseline_state')
    if args.replacement_teachers:
        extra=args.replacement_teachers.resolve();payload=json.loads((extra/'result.json').read_text(encoding='utf-8'))
        assert payload['all_phases_both_dt_pass'] and len(payload['records'])==10
        extra_registration=json.loads((extra/'registration.json').read_text(encoding='utf-8'))
        replacement_path=Path(extra_registration['teacher']);assert sha(replacement_path)==extra_registration['teacher_sha256']
        replacement=from_saved(torch.load(replacement_path,map_location='cpu',weights_only=False))
        critical_ids.append('corner_18');corner_s=next(c['reference_s_mm'] for c in geometry['sharp_corners'] if c['id']=='corner_18')
        for record in payload['records']:
            path=Path(record['trace']);assert sha(path)==record['trace_sha256']
            with np.load(path) as d:add(path,d['observations'],d['latents'],2.,'executed_teacher')
        failure_source=ROOT/'direct14_actual_arrival_fit_round52'
        for key,dt in [('1ms',.001),('half_ms',.0005)]:
            path=failure_source/f'eval_0000_original_{key}.npz';gates=failure_source/f'eval_0000_original_{key}_gates.json'
            xx,ss=observed_progress(path,gates,dt);xx=xx[abs(ss-corner_s)<=40.]
            with torch.no_grad():yy=replacement.actor(torch.tensor(xx)).numpy()
            add(path,xx,yy,2.,'counterfactual_teacher_query_at_observed_baseline_state');bindings[str(gates)]=sha(gates)
        bindings[str(extra/'result.json')]=sha(extra/'result.json');bindings[str(replacement_path)]=sha(replacement_path)
    if args.anchor_policy:bindings[str(args.anchor_policy)]=sha(args.anchor_policy)
    critical_s=np.array([c['reference_s_mm'] for c in geometry['sharp_corners'] if c['id'] in critical_ids])
    specs=[(parent.parent/'eval_0002_original_1ms.npz',parent.parent/'eval_0002_original_1ms_gates.json',.001),
           (parent.parent/'eval_0002_original_half_ms.npz',parent.parent/'eval_0002_original_half_ms_gates.json',.0005),
           (ROOT/'direct14_farvector_local_feedback_fit_round21_original_half_ms/trace.npz',ROOT/'direct14_farvector_local_feedback_fit_round21_original_half_ms/gate_events.json',.0005)]
    if args.anchor_executions:
        specs=[(args.anchor_executions/f'eval_{args.anchor_index:04d}_original_{key}.npz',
                args.anchor_executions/f'eval_{args.anchor_index:04d}_original_{key}_gates.json',dt)
               for key,dt in [('1ms',.001),('half_ms',.0005)]]
    for path,gates,dt in specs:
        x,s=observed_progress(path,gates,dt)
        keep=(abs(s[:,None]-critical_s)>40).all(1)&(np.arange(len(x))%4==0)
        x=x[keep]
        with torch.no_grad():y=anchor.actor(torch.tensor(x)).numpy()
        add(path,x,y,1.,'unchanged_full_execution_anchor');bindings[str(gates)]=sha(gates)
    for key in ['1ms','half_ms']:
        path=parent.parent/f'eval_0002_dense_{key}.npz'
        if args.anchor_executions:path=args.anchor_executions/f'eval_{args.anchor_index:04d}_dense_{key}.npz'
        with np.load(path) as d:x=d['observations']
        with torch.no_grad():y=anchor.actor(torch.tensor(x)).numpy()
        add(path,x,y,4.,'unchanged_dense_anchor')
    x=np.vstack(pieces);target=np.vstack(targets);weight=np.array(weights);scale=x.std(0).clip(.03)
    action_scale=np.array([.01,.08,.4]);positive=np.isin(np.array(labels),['executed_teacher','counterfactual_teacher_query_at_observed_baseline_state'])
    active=x[positive&(np.linalg.norm(target/action_scale,axis=1)>.005)];z=active/scale
    selected_centers=[int(np.argmax(np.linalg.norm(z,axis=1)))];dist=((z-z[selected_centers[0]])**2).sum(1)
    for _ in range(args.basis_count-1):
        idx=int(np.argmax(dist));selected_centers.append(idx);dist=np.minimum(dist,((z-z[idx])**2).sum(1))
    centers=active[selected_centers];assert len(np.unique(centers,axis=0))==args.basis_count
    np.savez_compressed(stage/'fit_data.npz',observations=x,target_delta=target,weights=weight,source_kind=np.array(labels),centers=centers,scale=scale)
    write(stage/'data_binding.json',dict(source_sha256=bindings,fit_data_sha256=sha(stage/'fit_data.npz'),counterfactual_labels_explicit=True))
    tensor_x=torch.tensor(x);transformed=np.arctanh(np.clip(target/action_scale,-.999,.999));records=[]
    with torch.no_grad():baseline=base.actor(tensor_x)
    for i,radius in enumerate(radii):
        policy=LocalizedJointPolicy(count=args.basis_count);policy.initialize(base)
        with torch.no_grad():
            policy.actor.centers.copy_(torch.tensor(centers));policy.actor.feature_scale.copy_(torch.tensor(scale));policy.actor.radius.fill_(radius)
            assert torch.equal(policy.actor(tensor_x),baseline)
            f=policy.actor.features(tensor_x).numpy().astype(float)
        coefficients=np.linalg.solve(f.T@(f*weight[:,None])+1e-4*np.eye(args.basis_count),f.T@(transformed*weight[:,None]))
        with torch.no_grad():
            policy.actor.correction.weight.copy_(torch.tensor(coefficients.T,dtype=torch.float32))
            delta=(policy.actor(tensor_x)-baseline).numpy()
        records.append(dict(index=i,radius=radius,weighted_normalized_rmse=np.sqrt(np.average(((delta-target)/action_scale)**2,weights=weight,axis=0)).tolist(),
            dense_max_delta=abs(delta[np.array(labels)=='unchanged_dense_anchor']).max(0).tolist(),initial_all_outputs_bitwise_equal=True))
        payload=dict(saved);payload.update(policy=policy.state_dict(),updates=i,policy_class='LocalizedJointPolicy',joint_basis_count=args.basis_count,base_local_basis_count=96,
            parent_checkpoint=str(parent),parent_checkpoint_sha256=sha(parent),new_ppo_updates=0,training_kind='Offline actual-arrival three-action local fit',checkpoint_index_kind='support_scale_index')
        torch.save(payload,stage/f'checkpoint_{i:04d}.pt')
    write(stage/'fit_audit.json',records);print('JOINT_LOCAL_FIT',records,flush=True)
    remaining=min(1740.,1800.-(time.perf_counter()-started)-5.)
    try:
        result=subprocess.run([sys.executable,'-B',str(ROOT/'screen_saved_mixed_checkpoints.py'),'--name',args.name,'--source',str(stage),
            '--counts','0','1','2','--localized-joint-head'],timeout=remaining)
        status='finished';code=result.returncode
    except subprocess.TimeoutExpired:status='timeout';code=None
    write(stage/'receipt.json',dict(status=status,exit_code=code,combined_wall_s=time.perf_counter()-started,
        sources_unchanged=all(sha(p)==h for p,h in hashes.items()),new_ppo_updates=0,goal_complete=False))


if __name__=='__main__':main()
