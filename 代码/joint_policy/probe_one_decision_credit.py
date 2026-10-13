"""Single-decision counterfactual credit across the complete nominal training pool."""
import time
import numpy as np
import torch
from local_geometry_feedback_policy import LocalGeometryFeedbackPolicy
from probe_coordinated_suffix import reconstruct
from complete_corner_metrics import complete_corner_metrics,coverage_costs
from bounded_shape_return import terminal_shape_cost
from validate_actions import difference_audit
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


AMPLITUDE=np.array([.005,.1,.1])


def score_complete(reference,rows,actions,env=None):
    length=float(np.linalg.norm(np.diff(reference,axis=0),axis=1).sum())
    base=-len(rows)*.001+length/100
    if env is not None and not env.success:
        base-=env.max_ticks*.001+1+max(0.,length-env.progress()[0])/20
    if len(reference)!=3:
        return dict(base_return=base,original_full_return=base,independent_zero_return=base,
                    original_shape_cost=0.,independent_shape_cost=0.,shape=None)
    _,old=terminal_shape_cost(reference,rows,actions,discrete_first_label=False)
    shape=complete_corner_metrics(reference,rows)
    original=coverage_costs(old,shape)['full_two_edges']['cost']
    q=(shape['full_normalized_energy']+
       5*(shape['in_edge_outward_mm']+shape['out_edge_outward_mm'])/1.25+
       2*shape['full_physical_reverse_deg']/abs(shape['turn_deg']))
    independent=float(-np.expm1(-q))
    assert 0<=original<=1 and 0<=independent<=1
    return dict(base_return=base,original_full_return=base-original,
        independent_zero_return=base-independent,original_shape_cost=original,
        independent_shape_cost=independent,shape=shape)


def main():
    torch.set_num_threads(1)
    parent=BASE/'exit_local_credit_parameter_learning_20260916'
    pool=parent/'training_1';initial=parent/'candidate_1.pt'
    stage=register('single_decision_complete_credit_probe_20260917',
        [__file__,initial,parent/'receipt.json',pool/'summary.json',
         BASE/'probe_coordinated_suffix.py',BASE/'bounded_shape_return.py',
         BASE/'complete_corner_metrics.py',BASE/'local_geometry_feedback_policy.py',
         BASE/'joint_runtime.py',BASE/'joint_runtime_v5.py',BASE/'validate_actions.py',
         BASE/'learning_preparation_v4/paths.json'],
        dict(new_queries=160,new_suffixes=960,actor_calls=60000,new_physics_ticks=1500000,
             wall_s=600,intervention_decisions=1,latent_amplitude=AMPLITUDE.tolist(),
             isolated_decision_quantiles=[.2,.4,.6,.8],other_decision_quantiles=[.33,.67],
             new_policy_training_decisions=0),
        'Mechanism-informed deterministic counterfactual credit collection across all64 '
        'original training paths. Reuse complete parent trajectories. At four evenly spaced '
        'decision quantiles per isolated corner and two per other path, perturb one head '
        'for exactly one20ms decision, then resume the unchanged nominal deterministic '
        'policy to full stop or true failure. Six antithetic axial branches share the exact '
        'state. This differs from prior60/120ms block credit and sparse-corner fits in '
        'duration and query coverage; do not claim a single-factor comparison to them. '
        'Score the same physics under original complete-extent max-edge shape cost and '
        'a declared objective repair: independent entry+exit excursions, zero positional '
        'and reversal allowances, unchanged energy/weights/bounded transform. The objective '
        'repair bundles these disclosed changes; time/failure charges and all physical '
        'constraints remain identical. Non-isolated paths still use original base return, '
        'a known objective-coverage limitation. Store raw full-return finite derivatives '
        'including failures separately from complete-side derivatives; retain every failed '
        'axis as a local exclusion observation. No learning, deployment or development '
        'evaluation in collection. Analyze signal coverage and feasibility before separately '
        'registering any parameter update; no automatic continuation of a failed fit family.')
    started=time.perf_counter();costs=dict(actor_calls=0,new_physics_ticks=0,saved_prefix_geometry_ticks=0)
    policy=LocalGeometryFeedbackPolicy(mask_endpoint=True)
    policy.load_state_dict(torch.load(initial,map_location='cpu',weights_only=False)['policy'])
    baseline=[];records=[];queries=[];query_observations=[]
    saved_results=read(pool/'summary.json')['results'];assert all(r['success'] for r in saved_results)
    for path in range(64):
        with np.load(pool/f'{path}.npz') as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        base=score_complete(ref,rows,actions);baseline.append(dict(path=path,decisions=len(actions),**base))
        fractions=[.2,.4,.6,.8] if len(ref)==3 else [.33,.67]
        anchors=sorted({int(q*(len(actions)-1)) for q in fractions})
        for anchor in anchors:
            frame=anchor*20;env=reconstruct(ref,rows,actions,frame,costs);initial_state=env.state()
            observation=env.observation();latent=policy.act(observation,True)[0];costs['actor_calls']+=1
            assert np.array_equal(env.decode_action(latent),actions[anchor])
            qid=len(queries);query_observations.append(observation)
            queries.append(dict(query=qid,path=path,anchor=anchor,
                path_objective_gradient_weight=len(actions)/(64*len(anchors))))
            for axis in range(3):
                for direction in (-1,1):
                    env.restore(initial_state);chunks=[];held=[];count=0
                    while not env.done:
                        assert time.perf_counter()-started<600
                        assert costs['actor_calls']<60000 and costs['new_physics_ticks']<1497880
                        action=policy.act(env.observation(),True)[0];costs['actor_calls']+=1
                        if count==0:action=action.astype(float);action[axis]+=direction*AMPLITUDE[axis]
                        _,_,_,info=env.step(action);chunks.append(info['rows']);held.append(info['held_action']);count+=1
                        costs['new_physics_ticks']+=len(info['rows'])
                    suffix=np.vstack(chunks);suffix_actions=np.asarray(held)
                    full=np.vstack([rows[:frame],suffix])
                    all_actions=np.vstack([actions[:anchor],suffix_actions]) if anchor else suffix_actions
                    audit=difference_audit(ref[0],full)
                    success=bool(env.success and audit['kinematic_pass'] and audit['rest_pass'])
                    value=score_complete(ref,full,all_actions,env)
                    name=f'q{qid}_h{axis}_{direction:+d}.npz'
                    np.savez_compressed(stage/name,reference=ref,suffix_rows=suffix,
                        suffix_actions=suffix_actions,baseline_path=path,anchor=anchor,
                        observation=observation,axis=axis,latent_offset=direction*AMPLITUDE[axis])
                    records.append(dict(query=qid,path=path,anchor=anchor,axis=axis,direction=direction,
                        latent_offset=float(direction*AMPLITUDE[axis]),success=success,reason=env.reason,
                        full_time_s=len(full)*.001 if success else None,trace=name,audit=audit,
                        geometry=env.tracker.snapshot(1.25),**value))
        np.savez_compressed(stage/'queries.npz',observation=np.asarray(query_observations))
        write_json(stage/'results.json',dict(baselines=baseline,queries=queries,records=records,costs=costs))
        print('ONE_DECISION',path,'queries',len(queries),'suffixes',len(records),
              'complete',sum(r['success'] for r in records),'calls',costs['actor_calls'],flush=True)
    sensitivities=[]
    for query in queries:
        qid=query['query'];base=baseline[query['path']];per_objective={}
        for objective in ('original_full_return','independent_zero_return'):
            raw=[];feasible=[];modes=[]
            for axis in range(3):
                pair=sorted([r for r in records if r['query']==qid and r['axis']==axis],key=lambda r:r['direction'])
                lo,hi=pair;raw.append((hi[objective]-lo[objective])/(2*AMPLITUDE[axis]))
                good=[r for r in pair if r['success']]
                if len(good)==2:value=raw[-1];mode='central_complete'
                elif len(good)==1:value=(good[0][objective]-base[objective])/good[0]['latent_offset'];mode='feasible_one_sided'
                else:value=0.;mode='both_sides_failed'
                feasible.append(value);modes.append(mode)
            per_objective[objective]=dict(raw_finite_gradient=raw,complete_side_gradient=feasible,modes=modes)
        sensitivities.append(dict(query,objectives=per_objective))
    write_json(stage/'sensitivities.json',dict(records=sensitivities,
        finite_difference_warning='Finite20ms offsets; discontinuous completion times and gate events need independent physical checks. Not exact differentiable Q.'))
    receipt(stage,dict(status='completed_nominal_single_decision_counterfactual_credit',costs=costs,
        queries=len(queries),suffixes=len(records),complete_suffixes=sum(r['success'] for r in records),
        wall_s=time.perf_counter()-started))


if __name__=='__main__':main()
