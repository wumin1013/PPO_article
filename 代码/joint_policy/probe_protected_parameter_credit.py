"""Preserve a successful learned function on an interfering training trajectory."""
import time
from datetime import datetime,timezone
import numpy as np
import torch
from learn_exit_local_credit import last_layer_features
from learn_single_decision_credit import SCREEN,review,adoption_checks
from local_geometry_feedback_policy import LocalGeometryFeedbackPolicy
from probe_coordinated_suffix import reconstruct
from probe_one_decision_credit import AMPLITUDE
from joint_runtime_v5 import environment_factory
from train_pilot import evaluate,save_torch
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    torch.set_num_threads(1)
    parent=BASE/'matched_dense_corner_credit_learning_20260917'
    selection=read(parent/'training_selection.json');model=parent/selection['model']
    control=BASE/'closed_loop_parameter_credit_probe_20260917'
    support_file=BASE/'matched_entry_interaction_feedback_20260916/window_training_data.npz'
    stage=register('protected_closed_loop_parameter_credit_probe_20260917',
        [__file__,model,parent/'training_selection.json',control/'receipt.json',control/'basis.npz',
         control/'results.json',support_file,BASE/'probe_coordinated_suffix.py',
         BASE/'learn_exit_local_credit.py',BASE/'learn_single_decision_credit.py',
         BASE/selection['traces'][31]],
        dict(protected_training_path=31,protected_states=72,basis_dimensions=5,
             parameter_perturbations=10,coefficient_amplitude=.5,evaluation_actor_calls=18000,
             evaluation_physics_ticks=420000,wall_s=360,saved_prefix_geometry_ticks=1500),
        'Test a specific interference correction: four shape/feed-positive learned-function '
        'changes failed the same already-successful training path31 in the previous probe. '
        'Project the identical five output-layer proposal directions into the nullspace '
        'of all72 path31 nominal action-feature rows, using the identical4177-state metric. '
        'Renormalize each projected direction to the SAME maximum saved-support action '
        'amplitude as its unprojected control. Only the optimizer projection changes; '
        'the deployed actor architecture/executor remain identical. No route identifier, '
        'path-specific rule, intervention schedule or online projector enters the policy. '
        'Exact real-arithmetic preservation is checked algebraically; float32 output and '
        'actual full path31 reproduction are measured separately, not assumed. '
        'Evaluate the same+/-0.5 coefficients on the same21 routes, reusing unprojected '
        'controls. Failed prefixes never count as completion or aggregate shape gains. '
        'Stop if nullspace leaves no useful safe learning directions; no development or '
        'sealed tests. Preserving one training skill is not final generalization evidence.')
    started=time.perf_counter();costs=dict(saved_prefix_geometry_ticks=0,evaluation_actor_calls=0,evaluation_physics_ticks=0)
    state=torch.load(model,map_location='cpu',weights_only=False)
    p=LocalGeometryFeedbackPolicy(mask_endpoint=True);p.load_state_dict(state['policy'])
    with np.load(BASE/selection['traces'][31]) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
    protected=[];env=reconstruct(ref,rows,actions,(len(actions)-1)*20,costs,observations=protected)
    protected.append(env.observation());protected=np.asarray(protected);assert len(protected)==72
    with np.load(support_file) as z:obs=z['observation'][:4177].copy();weights=z['weight'][:4177].copy()*3
    with np.load(control/'basis.npz') as z:original=z['basis'].copy();names=z['names'].tolist()
    with np.load(BASE/'dense_corner_complete_credit_collection_20260917/queries.npz') as z:qobs=z['observation'].copy()
    with np.load(BASE/'single_decision_complete_credit_probe_20260917/queries.npz') as z:sobs=z['observation'].copy()
    f=last_layer_features(p,obs);cf=last_layer_features(p,protected)
    support=np.concatenate([f,last_layer_features(p,qobs),last_layer_features(p,sobs)])
    projected=np.zeros_like(original);ranks=[]
    for h in range(3):
        c=cf[:,h];_,singular,vt=np.linalg.svd(c,full_matrices=True)
        rank=int((singular>1e-12).sum());z=vt[rank:].T;ranks.append(rank)
        fisher=f[:,h].T@(weights[:,None]*f[:,h])+.001*np.eye(f.shape[-1])
        reduced=z.T@fisher@z
        for i in range(5):projected[i,h]=z@np.linalg.solve(reduced,z.T@fisher@original[i,h])
    audits=[]
    for i in range(5):
        oldmax=float(abs(np.einsum('nhf,hf->nh',support,original[i])/AMPLITUDE).max())
        newmax=float(abs(np.einsum('nhf,hf->nh',support,projected[i])/AMPLITUDE).max())
        factor=oldmax/max(newmax,1e-20);projected[i]*=factor
        error=float(abs(np.einsum('nhf,hf->nh',cf,projected[i])).max());assert error<1e-9
        audits.append(dict(name=names[i],renormalization=factor,protected_algebraic_max_error=error,
            max_parameter_delta=float(abs(projected[i]).max()),saved_normalized_action_amplitude=oldmax))
    np.savez_compressed(stage/'basis.npz',basis=projected,names=np.asarray(names),protected_observations=protected,
        protected_features=cf,source_basis=original)
    write_json(stage/'projection_audit.json',dict(rank_by_head=ranks,nullity_by_head=[97-r for r in ranks],directions=audits))
    baseline=read(control/'results.json')['baseline'];paths=read(BASE/'learning_preparation_v4/paths.json')['training']
    with torch.no_grad():protected_mean=p.distribution(torch.tensor(protected)).mean.numpy()
    status_path=BASE/'当前执行状态.json';status=read(status_path)
    status['current'].update(state='protected_parameter_credit_running',training_running=False,
        physical_diagnostic_running=True,diagnostic_stage=stage.name)
    status['running']=stage.name;status['active_experiment']=stage.name;status['updated_utc']=datetime.now(timezone.utc).isoformat()
    write_json(status_path,status)
    candidates=[]
    for i,name in enumerate(names):
        for sign in (-1,1):
            delta=sign*.5*projected[i];p.load_state_dict(state['policy'])
            with torch.no_grad():
                p.actor[-1].weight.add_(torch.tensor(delta[:,:-1],dtype=torch.float32))
                p.actor[-1].bias.add_(torch.tensor(delta[:,-1],dtype=torch.float32))
                function_error=float(abs(p.distribution(torch.tensor(protected)).mean.numpy()-protected_mean).max())
            tag=f'b{i}_{sign:+d}';save_torch(stage/f'{tag}.pt',dict(policy=p.state_dict(),architecture='local_geometry_feedback',
                mask_endpoint=True,parent_sha256=sha(model),basis=i,coefficient=sign*.5,training_nullspace_protection=True))
            rs=evaluate(p,[np.asarray(paths[k]) for k in SCREEN],stage,tag,None,costs,started+360,environment_factory)
            assert costs['evaluation_actor_calls']<=18000 and costs['evaluation_physics_ticks']<=420000
            rs=[dict(r,original_path=k) for r,k in zip(rs,SCREEN)];traces=[stage/tag/f'{j}.npz' for j in range(len(SCREEN))]
            with np.load(traces[SCREEN.index(31)]) as z:
                physical_error=float(abs(z['rows'][:,:2]-rows[:,:2]).max()) if z['rows'].shape==rows.shape else None
                physical_equal=np.array_equal(z['rows'],rows)
            r=dict(review(rs,traces),basis=i,name=name,sign=sign,coefficient=sign*.5,tag=tag,
                results=rs,traces=[str(t.relative_to(BASE)) for t in traces],protected_float32_latent_error=function_error,
                protected_xy_max_error_mm=physical_error,protected_rows_bitwise=physical_equal)
            r['checks']=adoption_checks(r,baseline);r['eligible']=all(r['checks'].values());candidates.append(r)
            write_json(stage/'results.json',dict(baseline=baseline,candidates=candidates,costs=costs,goal_complete=False))
            print('PROTECTED_CREDIT',name,sign,{k:r[k] for k in ('successes','full_time_s','entry_sum_mm','exit_sum_mm',
                'full_reverse_sum_deg','protected_float32_latent_error','protected_xy_max_error_mm','eligible')},flush=True)
    receipt(stage,dict(status='completed_protected_parameter_credit_probe',costs=costs,candidates=len(candidates),
        screen_eligible=sum(r['eligible'] for r in candidates),wall_s=time.perf_counter()-started))
    status=read(status_path);status['current'].update(state='protected_parameter_credit_complete',physical_diagnostic_running=False)
    status['completed'].append(dict(stage=stage.name,receipt_sha256=sha(stage/'receipt.json')))
    status['running']=None;status['active_experiment']=None;status['updated_utc']=datetime.now(timezone.utc).isoformat()
    write_json(status_path,status)


if __name__=='__main__':main()
