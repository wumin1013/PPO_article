"""Expand a constrained learner with data-derived normals of its binding constraints."""
import time
from datetime import datetime,timezone
import numpy as np
import torch
from kernel_local_feedback_policy import KernelLocalFeedbackPolicy
from diagnose_local_basis_fit import feature_batches
from probe_one_decision_credit import AMPLITUDE
from learn_single_decision_credit import review
from train_pilot import evaluate,save_torch
from joint_runtime_v5 import environment_factory
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    torch.set_num_threads(1)
    parent=BASE/'recentered_reverse_shape_credit_learning_20260917'
    previous=BASE/'headwise_function_credit_probe_20260917'
    limitation=BASE/'headwise_smoothness_preserving_credit_learning_20260917'
    dense=BASE/'dense_corner_complete_credit_collection_20260917'
    sparse=BASE/'single_decision_complete_credit_probe_20260917'
    selection=read(parent/'training_selection.json');model=parent/selection['model']
    old=read(previous/'results.json');screen=[r['original_path'] for r in old['candidates'][0]['results']]
    duals=read(limitation/'optimizer.json')['largest_shape_optimum_duals'][:4]
    targets=[]
    for row in duals:
        name,path=row['constraint'].rsplit('_',1)
        targets.append((int(path),'in_edge_outward_mm' if name=='entry' else name))
    assert targets==[(12,'in_edge_outward_mm'),(5,'full_normalized_energy'),
        (2,'full_spatial_peak_curvature_mm_inv'),(9,'full_spatial_peak_curvature_mm_inv')]
    stage=register('constraint_normal_function_credit_probe_20260917',
        [__file__,model,parent/'training_selection.json',previous/'basis.npz',previous/'receipt.json',
         previous/'results.json',limitation/'optimizer.json',limitation/'receipt.json',
         dense/'results.json',dense/'queries.npz',sparse/'results.json',
         BASE/'kernel_local_feedback_policy.py',BASE/'diagnose_local_basis_fit.py'],
        dict(new_parameter_candidates=8,reused_parameter_candidates=26,
             evaluation_actor_calls=20000,evaluation_physics_ticks=480000,wall_s=420,
             coefficient_amplitude=.5,binding_constraints=[dict(path=i,metric=m) for i,m in targets]),
        'The matched13-head-direction local model gives only2.95% common shape improvement '
        'when per-corner energy and peak curvature are preserved, below the predeclared '
        'rollout threshold. Its four largest inequality duals identify missing constraint '
        'correction directions. Build four per-trajectory metric normals from ALREADY '
        'SAVED all-time single-action consequences, using the same256-kernel feature metric, '
        'ridge and path16/31 nullspace. Old-parent action effects are proposals only; '
        'measure each new parameter normal at+/-0.5 on the identical29 routes before '
        'using it as current function credit. Reuse all26 previous exact-parent effects. '
        'A training trajectory chooses a gradient contribution, never a runtime path/angle '
        'lookup, teacher template or intervention schedule. All outputs remain learned '
        'feedback functions. No dimensionality change, altered executor, development '
        'or sealed tests. Stop this expansion if no useful actual shape-and-smoothness '
        'direction exists; do not relax constraints to force a positive result.')
    started=time.perf_counter();costs=dict(evaluation_actor_calls=0,evaluation_physics_ticks=0)
    saved=torch.load(model,map_location='cpu',weights_only=False)
    p=KernelLocalFeedbackPolicy(256);p.load_state_dict(saved['policy'])
    assert read(previous/'basis_mapping.json')['parent_sha256']==sha(model)
    with np.load(previous/'basis.npz') as z:prepared={k:z[k].copy() for k in z.files}
    with np.load(dense/'queries.npz') as z:qx=z['observation'].copy()
    x,w,cx=prepared['observation'],prepared['weight'],prepared['protected_observations']
    f,cf,qf=[feature_batches(p,obs) for obs in (x,cx,qx)]
    raw=read(dense/'results.json');old_action=read(sparse/'results.json')
    lookup={(r['query'],r['axis'],r['direction']):r for r in raw['records']}
    baselines={r['path']:r for r in old_action['baselines']}
    gradients=[]
    for path,metric in targets:
        sensitivities=np.zeros((len(qx),3))
        for q in raw['queries']:
            if q['path']!=path:continue
            for h in range(3):
                lo,hi=[lookup[q['query'],h,s] for s in (-1,1)]
                if lo['success'] and hi['success']:g=(hi['shape'][metric]-lo['shape'][metric])/(2*AMPLITUDE[h])
                elif lo['success']:g=(lo['shape'][metric]-baselines[path]['shape'][metric])/(-AMPLITUDE[h])
                elif hi['success']:g=(hi['shape'][metric]-baselines[path]['shape'][metric])/AMPLITUDE[h]
                else:g=0.
                sensitivities[q['query'],h]=g
        gradients.append(-np.einsum('nhf,nh,h->hf',qf,sensitivities,AMPLITUDE))
    added=np.zeros((4,3,256))
    for h in range(3):
        _,sv,vt=np.linalg.svd(cf[:,h],full_matrices=True);z=vt[(sv>1e-10).sum():].T
        metric=f[:,h].T@(w[:,None]*f[:,h])+.001*np.eye(256)
        reduced=z.T@metric@z
        for i,g in enumerate(gradients):added[i,h]=z@np.linalg.solve(reduced,z.T@g[h])
    audits=[];support=np.concatenate([f,qf])
    for i in range(4):
        scale=float(abs(np.einsum('nhf,hf->nh',support,added[i])).max());assert scale>1e-10
        added[i]/=scale
        error=float(abs(np.einsum('nhf,hf->nh',cf,added[i])*AMPLITUDE).max());assert error<1e-8
        audits.append(dict(path=targets[i][0],metric=targets[i][1],raw_direction_scale=scale,
            protected_latent_error=error,maximum_coefficient=float(abs(added[i]).max())))
    offset=len(prepared['basis']);names=prepared['names'].tolist()+[f'constraint_{i}_{m}' for i,m in targets]
    prepared['basis']=np.concatenate([prepared['basis'],added]);prepared['names']=np.asarray(names)
    np.savez_compressed(stage/'basis.npz',**prepared)
    write_json(stage/'proposal_audit.json',dict(proposals=audits,old_parent_data_is_proposal_only=True))
    candidates=[dict(r,reused_from=previous.name) for r in old['candidates']]
    baseline=old['baseline'];paths=read(BASE/'learning_preparation_v4/paths.json')['training']
    status_path=BASE/'当前执行状态.json';status=read(status_path)
    status['current'].update(state='constraint_normal_function_probe_running',training_running=False,physical_diagnostic_running=True)
    status['running']=stage.name;status['active_experiment']=stage.name;write_json(status_path,status)
    for i,direction in enumerate(added):
        for sign in (-1,1):
            p.load_state_dict(saved['policy'])
            with torch.no_grad():p.residual_coefficients.add_(torch.tensor(sign*.5*direction,dtype=torch.float32))
            tag=f'b{offset+i}_{sign:+d}'
            save_torch(stage/f'{tag}.pt',dict(policy=p.state_dict(),architecture='kernel_local_feedback',centers=256,
                parent_sha256=sha(model),basis=offset+i,coefficient=sign*.5,learning_method='constraint_normal_function_probe'))
            rs=evaluate(p,[np.asarray(paths[j]) for j in screen],stage,tag,None,costs,started+420,environment_factory)
            assert costs['evaluation_actor_calls']<=20000 and costs['evaluation_physics_ticks']<=480000
            rs=[dict(r,original_path=j) for r,j in zip(rs,screen)];traces=[stage/tag/f'{j}.npz' for j in range(len(screen))]
            r=dict(review(rs,traces),basis=offset+i,name=names[offset+i],sign=sign,coefficient=sign*.5,
                tag=tag,results=rs,traces=[str(t.relative_to(BASE)) for t in traces])
            candidates.append(r);write_json(stage/'results.json',dict(baseline=baseline,candidates=candidates,costs=costs,goal_complete=False))
            print('CONSTRAINT_NORMAL',names[offset+i],sign,{k:r[k] for k in ('successes','entry_sum_mm','exit_sum_mm','full_reverse_sum_deg')},flush=True)
    receipt(stage,dict(status='completed_constraint_normal_function_probe',new_candidates=8,reused_candidates=26,
        costs=costs,wall_s=time.perf_counter()-started))
    status=read(status_path);status['current'].update(state='constraint_normal_function_probe_complete',physical_diagnostic_running=False)
    status['completed'].append(dict(stage=stage.name,receipt_sha256=sha(stage/'receipt.json')))
    status['running']=None;status['active_experiment']=None;status['updated_utc']=datetime.now(timezone.utc).isoformat();write_json(status_path,status)


if __name__=='__main__':main()
