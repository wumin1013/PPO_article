"""Measure complete learned-function consequences in a small data-derived basis."""
import time
from datetime import datetime, timezone
import numpy as np
import torch
from learn_exit_local_credit import last_layer_features
from learn_constrained_entry_credit import solve_halfspace_qp
from learn_single_decision_credit import SCREEN, review, adoption_checks
from local_geometry_feedback_policy import LocalGeometryFeedbackPolicy
from probe_one_decision_credit import AMPLITUDE
from joint_runtime_v5 import environment_factory
from train_pilot import evaluate,save_torch
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


METRICS=('in_edge_outward_mm','out_edge_outward_mm','full_physical_reverse_deg','full_normalized_energy')
EPSILON=.5


def main():
    torch.set_num_threads(1)
    source=BASE/'matched_dense_corner_credit_learning_20260917'
    selection=read(source/'training_selection.json');model=source/selection['model']
    dense=BASE/'dense_corner_complete_credit_collection_20260917'
    sparse=BASE/'single_decision_complete_credit_probe_20260917'
    support_file=BASE/'matched_entry_interaction_feedback_20260916/window_training_data.npz'
    stage=register('closed_loop_parameter_credit_probe_20260917',
        [__file__,model,source/'receipt.json',source/'training_selection.json',
         dense/'receipt.json',dense/'results.json',dense/'queries.npz',
         sparse/'results.json',sparse/'queries.npz',support_file,
         BASE/'learn_exit_local_credit.py',BASE/'learn_constrained_entry_credit.py',
         BASE/'learn_single_decision_credit.py',BASE/'local_geometry_feedback_policy.py',
         BASE/'dense_credit_learned_xy_review_20260917/visual_check.json'],
        dict(basis_dimensions=5,parameter_perturbations=10,coefficient_amplitude=EPSILON,
             evaluation_actor_calls=18000,evaluation_physics_ticks=420000,wall_s=360,
             screen=SCREEN,new_single_action_branches=0),
        'Advance the64/64,12/12 learned candidate using complete closed-loop parameter '
        'credit instead of assuming a sum of isolated20ms finite effects. Reuse saved '
        'dense action consequences solely to construct four output-layer proposal directions '
        '(decrease total incoming excursion,outgoing excursion,reversal,curvature energy). '
        'Their data came from the preceding parent, so they are proposals, not claimed '
        'exact derivatives at the new actor. Same4177-state metric and known failed-axis '
        'projection; all three heads may change in each shape proposal. Add one feed-bias '
        'direction to expose whole-route time plateaus:latent feed bias0.2 per coefficient, '
        'not a cruise reward or deployed speed rule. Evaluate+/-0.5 coefficient for all '
        'five directions on the21 existing training paths. This applies each learned '
        'network change at every decision and measures its actual feedback consequences. '
        'Preserve every failure; no failure-prefix aggregate ranking. Baseline reuses '
        'selected complete training traces. No development/sealed evaluation in collection. '
        'Only actual complete parameter effects may drive the next bounded coordinated '
        'coefficient update. Stop if no feasible useful directions; do not repeat noise '
        'or learning-rate tuning. These perturbed policies are research candidates, '
        'not deployed intervention schedules or trajectory templates.')
    started=time.perf_counter();costs=dict(evaluation_actor_calls=0,evaluation_physics_ticks=0)
    saved=torch.load(model,map_location='cpu',weights_only=False)
    p=LocalGeometryFeedbackPolicy(mask_endpoint=True);p.load_state_dict(saved['policy'])
    raw=read(dense/'results.json');old=read(sparse/'results.json')
    with np.load(dense/'queries.npz') as z:qobs=z['observation'].copy()
    with np.load(sparse/'queries.npz') as z:sobs=z['observation'].copy()
    with np.load(support_file) as z:obs=z['observation'][:4177].copy();weights=z['weight'][:4177].copy()*3
    f=last_layer_features(p,obs);qf=last_layer_features(p,qobs);sf=last_layer_features(p,sobs)
    qs=raw['queries'];lookup={(r['query'],r['axis'],r['direction']):r for r in raw['records']}
    bases={r['path']:r for r in old['baselines']}
    failed_old=[r for r in old['records'] if not r['success']]
    failed_new=[r for r in raw['records'] if not r['success']]
    basis=[];audits=[]
    for metric in METRICS:
        sensitivity=[]
        for q in qs:
            row=[]
            for h in range(3):
                lo,hi=[lookup[q['query'],h,s] for s in (-1,1)]
                if lo['success'] and hi['success']:g=(hi['shape'][metric]-lo['shape'][metric])/(2*AMPLITUDE[h])
                elif lo['success']:g=(lo['shape'][metric]-bases[q['path']]['shape'][metric])/(-AMPLITUDE[h])
                elif hi['success']:g=(hi['shape'][metric]-bases[q['path']]['shape'][metric])/AMPLITUDE[h]
                else:g=0.
                row.append(g)
            sensitivity.append(row)
        sensitivity=np.asarray(sensitivity);heads=[];kkt=[]
        for h in range(3):
            fisher=f[:,h].T@(weights[:,None]*f[:,h])+.001*np.eye(f.shape[-1])
            gradient=-np.sum(qf[:,h]*(sensitivity[:,h]*AMPLITUDE[h]/16)[:,None],axis=0)
            constraints=[r['direction']*sf[r['query'],h] for r in failed_old if r['axis']==h]
            constraints += [r['direction']*qf[r['query'],h] for r in failed_new if r['axis']==h]
            direction,audit=solve_halfspace_qp(fisher,gradient,np.asarray(constraints).reshape(-1,f.shape[-1]))
            heads.append(direction);kkt.append(audit)
        direction=np.asarray(heads);support=np.concatenate([f,sf,qf])
        maximum=float(abs(np.einsum('nhf,hf->nh',support,direction)).max())
        direction=direction/max(maximum,1e-12)*AMPLITUDE[:,None];basis.append(direction)
        audits.append(dict(metric=metric,kkt=kkt,proposal_old_parent_derivative=float(np.sum(
            sensitivity*np.einsum('nhf,hf->nh',qf,direction))/16)))
    feed=np.zeros_like(basis[0]);feed[1,-1]=.2;basis.append(feed)
    names=list(METRICS)+['feed_bias'];basis=np.asarray(basis)
    np.savez_compressed(stage/'basis.npz',basis=basis,names=np.asarray(names),amplitude=EPSILON)
    write_json(stage/'basis_audit.json',dict(shape_directions=audits,feed_bias_per_unit=.2,
        learned_parameter_count=int(basis[0].size),basis_rank=int(np.linalg.matrix_rank(basis.reshape(5,-1)))))
    base_results=[selection['results'][i] for i in SCREEN]
    base_traces=[BASE/selection['traces'][i] for i in SCREEN]
    assert [r['original_path'] for r in base_results]==SCREEN
    baseline=review(base_results,base_traces)
    paths=read(BASE/'learning_preparation_v4/paths.json')['training']
    status_path=BASE/'当前执行状态.json';status=read(status_path)
    status['current'].update(state='closed_loop_parameter_credit_probe_running',training_running=False,
        physical_diagnostic_running=True,diagnostic_stage=stage.name)
    status['running']=stage.name;status['active_experiment']=stage.name
    status['updated_utc']=datetime.now(timezone.utc).isoformat();write_json(status_path,status)
    candidates=[]
    for i,name in enumerate(names):
        for sign in (-1,1):
            delta=sign*EPSILON*basis[i];p.load_state_dict(saved['policy'])
            with torch.no_grad():
                p.actor[-1].weight.add_(torch.tensor(delta[:,:-1],dtype=torch.float32))
                p.actor[-1].bias.add_(torch.tensor(delta[:,-1],dtype=torch.float32))
            tag=f'b{i}_{sign:+d}';save_torch(stage/f'{tag}.pt',dict(policy=p.state_dict(),
                architecture='local_geometry_feedback',mask_endpoint=True,parent_sha256=sha(model),basis=i,coefficient=sign*EPSILON))
            rs=evaluate(p,[np.asarray(paths[k]) for k in SCREEN],stage,tag,None,costs,started+360,environment_factory)
            assert costs['evaluation_actor_calls']<=18000 and costs['evaluation_physics_ticks']<=420000
            rs=[dict(r,original_path=k) for r,k in zip(rs,SCREEN)]
            traces=[stage/tag/f'{j}.npz' for j in range(len(SCREEN))]
            r=dict(review(rs,traces),basis=i,name=name,sign=sign,coefficient=sign*EPSILON,tag=tag,
                results=rs,traces=[str(t.relative_to(BASE)) for t in traces])
            r['checks']=adoption_checks(r,baseline);r['eligible']=all(r['checks'].values())
            candidates.append(r)
            write_json(stage/'results.json',dict(baseline=baseline,candidates=candidates,costs=costs,goal_complete=False))
            print('FUNCTION_CREDIT',name,sign,{k:r[k] for k in ('successes','full_time_s','entry_sum_mm','exit_sum_mm','full_reverse_sum_deg','eligible')},flush=True)
    receipt(stage,dict(status='completed_closed_loop_parameter_credit_probe',costs=costs,
        candidates=len(candidates),screen_eligible=sum(r['eligible'] for r in candidates),wall_s=time.perf_counter()-started))
    status=read(status_path);status['current'].update(state='closed_loop_parameter_credit_probe_complete',physical_diagnostic_running=False)
    status['completed'].append(dict(stage=stage.name,receipt_sha256=sha(stage/'receipt.json')))
    status['running']=None;status['active_experiment']=None;status['updated_utc']=datetime.now(timezone.utc).isoformat()
    status['next']='Use measured complete-function effects to solve a small coordinated parameter update with actual per-path shape constraints.'
    write_json(status_path,status)


if __name__=='__main__':main()
