"""Reuse inactive static features, preserving all other incumbent parameters."""
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
from research_policy_loader import from_saved
from observable_feedback_gate import modify,strength
from fit_actual_arrival_corrections import observed_progress


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


@torch.no_grad()
def array_call(function,x):
    return np.concatenate([function(torch.tensor(chunk)).numpy() for chunk in np.array_split(x,max(1,len(x)//512))])


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True)
    p.add_argument('--reuse-count',type=int,choices=[16,32,64],default=16)
    p.add_argument('--anchor-diagnostic',type=Path,default=ROOT/'retained55_reusable_centers_diagnostic.json')
    p.add_argument('--release-repair-anchors',action='store_true');a=p.parse_args()
    assert Path(a.name).name==a.name;verify_bound_inputs();torch.set_num_threads(1)
    started=time.perf_counter();stage=ROOT/(a.name+'_initialization');stage.mkdir(exist_ok=False);(stage/'source').mkdir()
    parent=ROOT/'direct14_actual_arrival_fit_round55/checkpoint_0001.pt'
    teachers=ROOT/'direct14_two_action_corner30_repair_round61'
    diagnostic=a.anchor_diagnostic;d=read(diagnostic)
    registration=read(teachers/'registration.json')
    assert d['checkpoint_sha256']==sha(parent)==registration['checkpoint_sha256']
    saved=torch.load(parent,map_location='cpu',weights_only=False);incumbent=from_saved(saved)
    reusable=np.array(d['unused_centers']);assert len(reusable)>=a.reuse_count
    # Deterministic choice limits changes in unevaluated observations as well.
    old_weights=incumbent.actor.correction.weight.detach().numpy()
    reuse=reusable[np.argsort(np.linalg.norm(old_weights[:,reusable],axis=0))[:a.reuse_count]]
    sources=[Path(__file__),ROOT/'research_policy_loader.py',ROOT/'localized_joint_correction.py',ROOT/'localized_preview_head.py',ROOT/'screen_saved_mixed_checkpoints.py',ROOT/'fit_actual_arrival_corrections.py']
    hashes={str(s):sha(s) for s in sources}
    for s in sources:shutil.copy2(s,stage/'source'/s.name)
    write(stage/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),kind=f'Offline constrained reuse of{a.reuse_count}inactive static features',
        parent=str(parent),parent_sha256=sha(parent),source_sha256=hashes,basis_count=192,updated_centers=reuse.tolist(),
        anchor_weights=[1.,4.,16.],radius='Retain incumbent1.2',inputs=14,outputs=3,policy_memory_scalars=0,new_ppo_updates=0,
        release_repair_window_anchors=a.release_repair_anchors,training_anchor_exclusion_half_width_mm=40. if a.release_repair_anchors else None,
        anchor_diagnostic=str(diagnostic),anchor_diagnostic_scope=d.get('scope','Full executions plus earlier teacher phase observations'),
        wall_budget_s=1800,screening_timeout_s=1740,
        evidence_gap=('Sixteen reused features miss98of204strongrepair observations;64centerscoverallstrongtargets without increasing192totalfeatures. Retain actual paired full behavior;earlier teacher-phase inputs are outside the retention guarantee.' if a.reuse_count==64 else ('Round63has51exactlycontradictory positive/no-change labels; release only actual30+-40mm no-change anchors while retaining all other behavior and physical gates.' if a.release_repair_anchors else 'Full head refitting can fix30while changing10and18; preserve existing active features and use only16inactive centers.')),
        procedure=f'Keep all other centers,scale,radius,weights and backbone unchanged. Fit only{a.reuse_count}three-output coefficient columns using actual61teachers and the explicitly bound incumbent anchor sources.',
        adoption='Full paired joint protocol including two-action all34addendum; no local or fitted-label result substitutes for full success.',
        stop='Three anchor weights or30minute combinedwalllimit.',goal_complete=False,method_frozen=False))
    bindings={str(parent):sha(parent),str(diagnostic):sha(diagnostic)};anchor_pieces=[];anchor_keep=[];exclusions=[]
    geometry_path=BASE/'paper_evidence_review/natural_closed_case/geometry.json';geometry=read(geometry_path)
    repair_ids={v['id'] for v in read(teachers/'result.json')['selected']}
    repair_s=np.array([c['reference_s_mm'] for c in geometry['sharp_corners'] if c['id'] in repair_ids])
    bindings[str(geometry_path)]=sha(geometry_path)
    for item in d['sources']:
        path=Path(item['path']);assert sha(path)==item['sha256'];bindings[str(path)]=sha(path)
        with np.load(path) as data:x=data['observations'].copy()
        keep=np.ones(len(x),bool)
        if a.release_repair_anchors and '_original_' in path.stem:
            gates=path.with_name(path.stem+'_gates.json');dt=.0005 if 'half_ms' in path.stem else .001
            xx,progress=observed_progress(path,gates,dt);assert np.array_equal(x,xx)
            keep=(abs(progress[:,None]-repair_s[None,:])>40.).all(1)
            bindings[str(gates)]=sha(gates)
        anchor_pieces.append(x);anchor_keep.append(keep);exclusions.append(dict(path=str(path),excluded=int((~keep).sum())))
    all_anchors=np.vstack(anchor_pieces);keep=np.concatenate(anchor_keep)
    all_anchor_y=array_call(incumbent.actor,all_anchors)
    assert (array_call(incumbent.actor.features,all_anchors)[:,reuse]==0).all()
    anchors=all_anchors[keep];anchor_y=all_anchor_y[keep]
    positive_x=[];positive_y=[]
    for item in read(teachers/'result.json')['selected']:
        chosen=item['selected'];assert chosen and len(chosen['traces'])==10
        steer,feed,gain=chosen['response']
        for path in [Path(v) for v in chosen['traces']]:
            bindings[str(path)]=sha(path)
            with np.load(path) as data:positive_x.append(data['observations'].copy());positive_y.append(data['latents'].copy())
        for key in ['1ms','half_ms']:
            path=teachers/key/f'{item["id"]}_r04_p00.npz';bindings[str(path)]=sha(path)
            with np.load(path) as data:x=data['observations'].copy()
            # Pointwise calls match the registered policy's rollout evaluation.
            y=[]
            for row in x:
                z=modify(incumbent.act(row,True)[0],row,feed,gain);z[0]+=strength(row)*steer;y.append(z)
            positive_x.append(x);positive_y.append(np.array(y))
    px=np.vstack(positive_x).astype(np.float32);py=np.vstack(positive_y).astype(np.float32)
    actual_change=py-array_call(incumbent.actor,px)
    active=px[np.linalg.norm(actual_change/np.array([.01,.08,.4]),axis=1)>.005]
    scaled=active/incumbent.actor.feature_scale.numpy();chosen=[int(np.argmax(np.linalg.norm(scaled,axis=1)))]
    distance=((scaled-scaled[chosen[0]])**2).sum(1)
    for _ in range(a.reuse_count-1):
        index=int(np.argmax(distance));chosen.append(index);distance=np.minimum(distance,((scaled-scaled[index])**2).sum(1))
    centers=active[chosen];assert len(np.unique(centers,axis=0))==a.reuse_count
    modified=from_saved(saved)
    with torch.no_grad():
        modified.actor.centers[reuse]=torch.tensor(centers)
        modified.actor.correction.weight[:,reuse]=0.
    zero_y=array_call(modified.actor,all_anchors)
    assert np.array_equal(all_anchor_y,zero_y),'Zero coefficient reuse changed a retained source observation'
    lookup={row.tobytes():target for row,target in zip(anchors,anchor_y)}
    conflicts=[float(abs(target-lookup[row.tobytes()]).max()) for row,target in zip(px,py) if row.tobytes() in lookup]
    conflict_count=sum(value>1e-5 for value in conflicts)
    if a.release_repair_anchors:assert conflict_count==0,('Remaining exact label conflict',conflict_count)
    write(stage/'anchor_selection_audit.json',dict(exclusions=exclusions,retained_anchor_samples=len(anchors),
        zero_initialization_all_source_observations_bitwise_equal=True,exact_label_conflicts_above1e_5=conflict_count,
        selection_scope='Training samples only;no change to evaluation windows,reference,physical constraints,or inputs.'))
    x=np.vstack([px,anchors]);y=np.vstack([py,anchor_y]);scale=modified.actor.output_scale.numpy()
    backbone=array_call(modified.actor.backbone,x)
    features=array_call(modified.actor.features,x).astype(float)
    held=features@modified.actor.correction.weight.detach().numpy().T
    target=np.arctanh(np.clip((y-backbone)/scale,-.999999,.999999))-held
    f=features[:,reuse];records=[]
    np.savez_compressed(stage/'fit_data.npz',observations=x,targets=y,positive_count=len(px),reused_indices=reuse,centers=centers)
    bindings[str(teachers/'result.json')]=sha(teachers/'result.json');bindings[str(teachers/'registration.json')]=sha(teachers/'registration.json')
    write(stage/'data_binding.json',dict(source_sha256=bindings,fit_data_sha256=sha(stage/'fit_data.npz'),counterfactual_labels_explicit=True))
    for index,anchor_weight in enumerate([1.,4.,16.]):
        weight=np.r_[np.full(len(px),2.),np.full(len(anchors),anchor_weight)]
        coefficients=np.linalg.solve(f.T@(f*weight[:,None])+1e-4*np.eye(a.reuse_count),f.T@(target*weight[:,None]))
        policy=from_saved(saved)
        with torch.no_grad():
            policy.actor.centers[reuse]=torch.tensor(centers)
            policy.actor.correction.weight[:,reuse]=torch.tensor(coefficients.T,dtype=torch.float32)
        kept=np.setdiff1d(np.arange(192),reuse)
        for key,value in policy.state_dict().items():
            original=saved['policy'][key]
            if key=='actor.centers':assert torch.equal(value[kept],original[kept])
            elif key=='actor.correction.weight':assert torch.equal(value[:,kept],original[:,kept])
            else:assert torch.equal(value,original),(key,'Unexpected incumbent parameter change')
        delta=array_call(policy.actor,anchors)-anchor_y
        records.append(dict(index=index,anchor_weight=anchor_weight,zero_initialization_all_anchors_bitwise_equal=True,
            anchor_max_delta=abs(delta).max(0).tolist(),anchor_rms_delta=np.sqrt((delta**2).mean(0)).tolist()))
        payload=dict(saved);payload.update(policy=policy.state_dict(),updates=index,new_ppo_updates=0,
            parent_checkpoint=str(parent),parent_checkpoint_sha256=sha(parent),training_kind=f'Offline{a.reuse_count}inactive-basis constrained reuse',checkpoint_index_kind='anchor_weight_index')
        torch.save(payload,stage/f'checkpoint_{index:04d}.pt')
    write(stage/'fit_audit.json',records);print('RETAINED_BASIS_FIT',records,flush=True)
    remaining=min(1740.,1800.-(time.perf_counter()-started)-5.)
    try:
        result=subprocess.run([sys.executable,'-B',str(ROOT/'screen_saved_mixed_checkpoints.py'),'--name',a.name,'--source',str(stage),
            '--counts','0','1','2','--localized-joint-head'],timeout=remaining)
        status='finished';code=result.returncode
    except subprocess.TimeoutExpired:status='timeout';code=None
    write(stage/'receipt.json',dict(status=status,exit_code=code,combined_wall_s=time.perf_counter()-started,
        sources_unchanged=all(sha(path)==digest for path,digest in hashes.items()),new_ppo_updates=0,goal_complete=False))


if __name__=='__main__':main()
