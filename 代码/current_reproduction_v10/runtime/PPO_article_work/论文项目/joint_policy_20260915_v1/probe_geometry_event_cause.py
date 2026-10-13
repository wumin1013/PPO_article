"""One-decision feature-event interventions; never a deployed controller."""
import itertools
import time
import numpy as np
import torch
from local_geometry_feedback_policy import LocalGeometryFeedbackPolicy
from probe_coordinated_suffix import reconstruct
from probe_one_decision_credit import score_complete
from validate_actions import difference_audit
from study_io import BASE, register, receipt, write_json
from search_complete_return import read


def main():
    torch.set_num_threads(1)
    audit=BASE/'preswing_sampled_geometry_event_audit_20260917'
    learned=BASE/'matched_single_decision_credit_learning_20260917'
    parent=BASE/'exit_local_credit_parameter_learning_20260916'
    stage=register('preswing_sampled_geometry_causal_probe_20260917',
        [__file__,audit/'receipt.json',audit/'results.json',parent/'candidate_1.pt',
         learned/'independent_zero_return_0.25.pt',learned/'independent_zero_return_0.5.pt',
         BASE/'probe_coordinated_suffix.py',BASE/'probe_one_decision_credit.py',
         BASE/'validate_actions.py',BASE/'local_geometry_feedback_policy.py'],
        dict(new_suffixes=13,actor_calls=2500,new_physics_ticks=65000,wall_s=120),
        'Physical conditional counterfactuals for sampled-tangent event attribution. '
        'Path13 quarter-step:at decision22 replace its10 sampled tangent vectors with the '
        'parent same-index vectors, recompute network action, and apply each of seven nonzero '
        'three-head combinations for that decision only. Resume unchanged updated actor. '
        'Reverse test:parent state/actor receives quarter-step tangent block once. Path31 '
        'half-step:replace tangent block at each of event25/28/44/55 separately, and at allfour; '
        'all other observations and physical dynamics unchanged. These off-policy observation '
        'interventions diagnose causality, not feasible sensing, learned corrections or '
        'deployable timing rules. Reuse original trajectories and failed baseline; include '
        'complete outcome/XY shape and failures, never compare failed-prefix efficiency. '
        'No training/development/sealed-test access; stop after13 suffixes.')
    started=time.perf_counter();costs=dict(actor_calls=0,new_physics_ticks=0,saved_prefix_geometry_ticks=0)
    policy=LocalGeometryFeedbackPolicy(mask_endpoint=True);records=[]
    cases=[]
    for bits in itertools.product((0,1),repeat=3):
        if any(bits):cases.append((13,'quarter','parent',[22],bits))
    cases.append((13,'parent','quarter',[22],(1,1,1)))
    for anchors in [[25],[28],[44],[55],[25,28,44,55]]:
        cases.append((31,'half','parent',anchors,(1,1,1)))
    for path,source,donor,anchors,bits in cases:
        with np.load(audit/f'p{path}_{source}.npz') as z:
            ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        with np.load(audit/f'p{path}_{donor}.npz') as z:
            donor_obs=z['observation'].copy()
        model=parent/'candidate_1.pt' if source=='parent' else learned/f'independent_zero_return_{.25 if source=="quarter" else .5:g}.pt'
        policy.load_state_dict(torch.load(model,map_location='cpu',weights_only=False)['policy'])
        first=min(anchors);env=reconstruct(ref,rows,actions,first*20,costs)
        chunks=[];held=[];count=first;interventions=[]
        while not env.done:
            assert time.perf_counter()-started<120
            assert costs['actor_calls']<2500 and costs['new_physics_ticks']<62880
            observation=env.observation();action=policy.act(observation,True)[0];costs['actor_calls']+=1
            if count in anchors:
                other=observation.copy();other[42:62]=donor_obs[count,42:62]
                altered=policy.act(other,True)[0];costs['actor_calls']+=1
                actual=np.where(bits,altered,action)
                interventions.append(dict(anchor=count,original=env.decode_action(action).tolist(),
                    intervened=env.decode_action(actual).tolist()))
                action=actual
            _,_,_,info=env.step(action);chunks.append(info['rows']);held.append(info['held_action']);count+=1
            costs['new_physics_ticks']+=len(info['rows'])
        full=np.vstack([rows[:first*20],*chunks]);all_actions=np.vstack([actions[:first],held])
        audit_result=difference_audit(ref[0],full)
        success=bool(env.success and audit_result['kinematic_pass'] and audit_result['rest_pass'])
        name=f'p{path}_{source}_'+','.join(map(str,anchors))+'_'+''.join(map(str,bits))+'.npz'
        np.savez_compressed(stage/name,reference=ref,rows=full,actions=all_actions)
        row=dict(path=path,source=source,donor=donor,anchors=anchors,bits=list(bits),success=success,
            reason=env.reason,full_time_s=len(full)*.001 if success else None,trace=name,
            interventions=interventions,audit=audit_result,geometry=env.tracker.snapshot(1.25),
            **score_complete(ref,full,all_actions,env))
        records.append(row)
        write_json(stage/'results.json',dict(records=records,costs=costs,goal_complete=False))
        print('EVENT_CAUSE',path,source,anchors,bits,success,
              {k:row['shape'][k] for k in ('in_edge_outward_mm','out_edge_outward_mm','full_physical_reverse_deg')} if row['shape'] else env.reason,flush=True)
    receipt(stage,dict(status='completed_conditional_geometry_event_cause_probe',costs=costs,
        complete_suffixes=sum(r['success'] for r in records),suffixes=len(records),wall_s=time.perf_counter()-started))


if __name__=='__main__':main()
