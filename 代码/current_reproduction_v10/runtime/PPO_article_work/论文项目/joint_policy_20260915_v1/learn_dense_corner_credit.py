"""Matched temporal-support correction starting from the retained learned policy."""
import time
from datetime import datetime, timezone
import numpy as np
import torch
from learn_exit_local_credit import last_layer_features
from learn_constrained_entry_credit import solve_halfspace_qp
from learn_single_decision_credit import SCREEN, review, adoption_checks
from local_geometry_feedback_policy import LocalGeometryFeedbackPolicy
from probe_one_decision_credit import AMPLITUDE
from search_complete_return import read
from study_io import BASE, register, receipt, sha, write_json
from train_pilot import evaluate, save_torch
from joint_runtime_v5 import environment_factory


def main():
    torch.set_num_threads(1)
    parent=BASE/'exit_local_credit_parameter_learning_20260916';pool=parent/'training_1';model=parent/'candidate_1.pt'
    dense=BASE/'dense_corner_complete_credit_collection_20260917'
    sparse=BASE/'single_decision_complete_credit_probe_20260917'
    earlier=BASE/'matched_single_decision_credit_learning_20260917'
    support_file=BASE/'matched_entry_interaction_feedback_20260916/window_training_data.npz'
    assert read(dense/'receipt.json')['status']=='completed_dense_corner_credit_collection'
    dense_records=read(dense/'results.json')['records']
    failures=[r for r in dense_records if not r['success']]
    stage=register('matched_dense_corner_credit_learning_20260917',
        [__file__,model,parent/'receipt.json',pool/'summary.json',support_file,
         dense/'receipt.json',dense/'sensitivities.json',dense/'queries.npz',
         sparse/'receipt.json',sparse/'sensitivities.json',sparse/'queries.npz',
         earlier/'directions.npz',earlier/'screen_results.json',earlier/'receipt.json',
         BASE/'learn_exit_local_credit.py',BASE/'learn_constrained_entry_credit.py',
         BASE/'learn_single_decision_credit.py',BASE/'joint_runtime_v5.py'],
        dict(new_probe_decisions=0,evaluation_actor_calls=22000,evaluation_physics_ticks=500000,
             wall_s=450,steps=[.25,.5,1.],new_directions=2,screen=SCREEN),
        'Apply the diagnosed temporal-credit support repair to the retained strong actor. '
        'Exactly match prior independent-zero objective,4177-state metric,ridge.001,last291 '
        'parameters,amplitude,known path31 failed-axis exclusion,normalization and steps. '
        'Extend isolated-corner quadrature from4 states to ALL decisions. Other48 '
        'paths retain identical sparse credit. Densification discovers additional failures: '
        'complete-side derivatives use their successful sides, and original raw failure '
        'returns remain preserved. Both new arms use exactly these same dense gradients; '
        'dense_old_exclusions retains the old exclusion and dense_all_exclusions adds '
        'the newly observed failed-axis halfspaces. This separates safety-exclusion effects '
        'from dense credit. These are local exclusion observations, not safety certificates. '
        'Verify overlapping query observations and '
        'reproduce the old sparse direction numerically before evaluating the new one. '
        'Reuse all prior sparse candidate physics. Every new failure is explicitly retained. '
        'Screen21 paths for each arm with unchanged adoption_checks:each entry nonworse,entry+exit '
        'sums both>=10%better,no higher max exit or reversal,complete time<=102%,better '
        'full objective. At most one selected candidate across both arms receives remaining43 training routes '
        'and then12 development routes only if full training passes. True physical XY '
        'remains required. No executor/observation change, intervention deployment, '
        'sealed tests or final-method/independent-seed claims. Stop this direction if '
        'no step qualifies; do not extend the same training unchanged.')
    started=time.perf_counter();costs=dict(evaluation_actor_calls=0,evaluation_physics_ticks=0)
    status_path=BASE/'当前执行状态.json';status=read(status_path)
    status['current'].update(state='matched_dense_corner_credit_learning_running',training_running=True,
        physical_diagnostic_running=True,diagnostic_stage=stage.name)
    status['running']=stage.name;status['active_experiment']=stage.name
    status['updated_utc']=datetime.now(timezone.utc).isoformat();write_json(status_path,status)
    saved=torch.load(model,map_location='cpu',weights_only=False)
    p=LocalGeometryFeedbackPolicy(mask_endpoint=True);p.load_state_dict(saved['policy'])
    base_results=[dict(r,original_path=i) for i,r in enumerate(read(pool/'summary.json')['results'])]
    base_traces=[pool/f'{i}.npz' for i in range(64)]
    baseline=review(base_results,base_traces)
    screen_baseline=review([base_results[i] for i in SCREEN],[base_traces[i] for i in SCREEN])
    n=sum(r['decisions'] for r in base_results);assert n==4177
    with np.load(support_file) as z:obs=z['observation'][:n].copy();weights=z['weight'][:n].copy()*3
    with np.load(sparse/'queries.npz') as z:sobs=z['observation'].copy()
    with np.load(dense/'queries.npz') as z:dobs=z['observation'].copy()
    sr=read(sparse/'sensitivities.json')['records'];dr=read(dense/'sensitivities.json')['records']
    dense_lookup={(r['path'],r['anchor']):r['query'] for r in dr}
    for row in sr:
        if row['path']<16:assert np.array_equal(sobs[row['query']],dobs[dense_lookup[(row['path'],row['anchor'])]])
    f=last_layer_features(p,obs);sf=last_layer_features(p,sobs);df=last_layer_features(p,dobs)
    failed=[r for r in read(sparse/'results.json')['records'] if not r['success']]
    other=[i for i,r in enumerate(sr) if r['path']>=16]
    directions={};audits={}
    for name,records,qf in [('sparse',sr,sf),
            ('dense_old_exclusions',dr+[sr[i] for i in other],np.concatenate([df,sf[other]])),
            ('dense_all_exclusions',dr+[sr[i] for i in other],np.concatenate([df,sf[other]]))]:
        sensitivity=np.asarray([r['objectives']['independent_zero_return']['complete_side_gradient'] for r in records])
        qw=np.asarray([r['path_objective_gradient_weight'] for r in records]);heads=[];kkt=[]
        for head in range(3):
            fisher=f[:,head].T@(weights[:,None]*f[:,head])+.001*np.eye(f.shape[-1])
            gradient=np.sum(qf[:,head]*(qw*sensitivity[:,head]*AMPLITUDE[head])[:,None],axis=0)
            constraints=[r['direction']*sf[r['query'],head] for r in failed if r['axis']==head]
            if name=='dense_all_exclusions':
                constraints += [r['direction']*df[r['query'],head] for r in failures if r['axis']==head]
            constraints=np.asarray(constraints).reshape(-1,f.shape[-1])
            direction,audit=solve_halfspace_qp(fisher,gradient,constraints);heads.append(direction);kkt.append(audit)
        direction=np.asarray(heads)
        # Preserve the identical old support metric/normalization for a clean factor.
        support=np.concatenate([f,sf]);maximum=float(abs(np.einsum('nhf,hf->nh',support,direction)).max())
        direction=direction/max(maximum,1e-12)*AMPLITUDE[:,None];directions[name]=direction
        predicted=float(np.sum(qw[:,None]*np.einsum('nhf,hf->nh',qf,direction)*sensitivity))
        assert predicted>0
        audits[name]=dict(kkt=kkt,predicted_return_gain=predicted,queries=len(records),
            extra_dense_max_normalized_change=float(abs(np.einsum('nhf,hf->nh',df,direction)/AMPLITUDE).max()))
    with np.load(earlier/'directions.npz') as z:old_direction=z['independent_zero_return']
    assert abs(directions['sparse']-old_direction).max()<1e-12
    np.savez_compressed(stage/'directions.npz',**directions)
    write_json(stage/'direction_audit.json',dict(arms=audits,old_direction_reproduced=True,
        dense_sparse_cosine=float(np.sum(directions['dense_old_exclusions']*old_direction)/(np.linalg.norm(directions['dense_old_exclusions'])*np.linalg.norm(old_direction))),
        old_failed_axes=failed,new_failed_axes=failures))
    paths=read(BASE/'learning_preparation_v4/paths.json');training=[np.asarray(x) for x in paths['training']]
    def run(indices,tag,geometry=training):
        rs=evaluate(p,[geometry[i] for i in indices],stage,tag,None,costs,started+450,environment_factory)
        assert costs['evaluation_actor_calls']<=22000 and costs['evaluation_physics_ticks']<=500000
        return [dict(r,original_path=i) for r,i in zip(rs,indices)],[stage/tag/f'{i}.npz' for i in range(len(indices))]
    candidates=[];cache={}
    for arm in ('dense_old_exclusions','dense_all_exclusions'):
        for step in (.25,.5,1.):
            p.load_state_dict(saved['policy']);delta=directions[arm]*step
            with torch.no_grad():
                p.actor[-1].weight.add_(torch.tensor(delta[:,:-1],dtype=torch.float32))
                p.actor[-1].bias.add_(torch.tensor(delta[:,-1],dtype=torch.float32))
            name=f'{arm}_{step:g}.pt'
            save_torch(stage/name,dict(policy=p.state_dict(),architecture='local_geometry_feedback',mask_endpoint=True,
                observation_dim=72,parent_sha256=sha(model),learning_method='dense_nominal_complete_return_credit',step=step,arm=arm))
            rs,traces=run(SCREEN,f'{arm}_{step:g}');cache[name]=(rs,traces)
            row=dict(review(rs,traces),step=step,model=name,arm=arm);row['checks']=adoption_checks(row,screen_baseline)
            row['eligible']=all(row['checks'].values());candidates.append(row)
            write_json(stage/'screen_results.json',dict(baseline=screen_baseline,candidates=candidates,costs=costs))
            print('DENSE_LEARNING',arm,step,{k:row[k] for k in ('successes','entry_sum_mm','exit_sum_mm','full_reverse_sum_deg','eligible')},row['checks'],flush=True)
    chosen=None;full=None;dev=None
    eligible=[r for r in candidates if r['eligible']]
    if eligible:
        chosen=max(eligible,key=lambda r:r['independent_zero_return'])
        write_json(stage/'screen_selection.json',dict(selected=chosen,training_only=True))
        p.load_state_dict(torch.load(stage/chosen['model'],map_location='cpu',weights_only=False)['policy'])
        rs,traces=run([i for i in range(64) if i not in SCREEN],'remaining')
        old_rs,old_traces=cache[chosen['model']];pairs=sorted(zip(old_rs+rs,old_traces+traces),key=lambda v:v[0]['original_path'])
        rs,traces=map(list,zip(*pairs));full=review(rs,traces);full['checks']=adoption_checks(full,baseline)
        write_json(stage/'training_selection.json',dict(model=chosen['model'],training=full,results=rs,traces=[str(x.relative_to(BASE)) for x in traces]))
        if all(full['checks'].values()):
            rs,traces=run(list(range(12)),'development',[np.asarray(x) for x in paths['validation']]);dev=review(rs,traces)
    sparse_rows=[r for r in read(earlier/'screen_results.json')['candidates'] if r['arm']=='independent_zero_return']
    write_json(stage/'results.json',dict(baseline=baseline,screen=candidates,reused_sparse=sparse_rows,
        chosen=chosen,full_training=full,development=dev,costs=costs,goal_complete=False))
    receipt(stage,dict(status='completed_matched_dense_corner_credit_learning',costs=costs,
        screen_eligible=len(eligible),full_training_eligible=bool(full and all(full['checks'].values())),wall_s=time.perf_counter()-started))
    status_path=BASE/'当前执行状态.json';status=read(status_path)
    status['current'].update(state='dense_corner_credit_learning_complete',training_running=False,physical_diagnostic_running=False)
    status['running']=None;status['active_experiment']=None;status['observed_research_processes']=[]
    status['completed'].append(dict(stage=stage.name,receipt_sha256=sha(stage/'receipt.json')))
    status['next']='Review actual XY and dense credit correction; advance only a genuinely improved learned candidate or switch mechanism based on the result.'
    status['updated_utc']=datetime.now(timezone.utc).isoformat();write_json(status_path,status)


if __name__=='__main__':main()
