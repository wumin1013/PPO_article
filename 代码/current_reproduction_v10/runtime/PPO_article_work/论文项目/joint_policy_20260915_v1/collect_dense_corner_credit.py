"""Complete temporal coverage for nominal three-head action consequences."""
import time
from datetime import datetime, timezone
import numpy as np
import torch
from local_geometry_feedback_policy import LocalGeometryFeedbackPolicy
from probe_coordinated_suffix import reconstruct
from probe_one_decision_credit import AMPLITUDE, score_complete
from validate_actions import difference_audit
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha


def main():
    torch.set_num_threads(1)
    parent=BASE/'exit_local_credit_parameter_learning_20260916';pool=parent/'training_1'
    old=BASE/'single_decision_complete_credit_probe_20260917'
    stage=register('dense_corner_complete_credit_collection_20260917',
        [__file__,parent/'candidate_1.pt',parent/'receipt.json',pool/'summary.json',
         old/'receipt.json',old/'results.json',old/'sensitivities.json',old/'queries.npz',
         BASE/'probe_coordinated_suffix.py',BASE/'probe_one_decision_credit.py',
         BASE/'local_geometry_feedback_policy.py',BASE/'validate_actions.py',
         BASE/'single_decision_temporal_composition_audit_20260917/receipt.json',
         BASE/'factorial_head_credit_attribution_20260917/results.json'],
        dict(paths=list(range(16)),all_decisions_per_path=True,latent_amplitude=AMPLITUDE.tolist(),
             actor_calls=160000,new_physics_ticks=3800000,wall_s=1500,optimizer_updates=0),
        'Correct a demonstrated sign error from four-time quadrature using all decisions '
        'on the16 existing isolated training corners of the retained strong model. Reuse '
        'every original axial branch at matching path/anchor/head/sign; same finite amplitude, '
        'one20ms action and original deterministic closed-loop continuation. Reconstruct '
        'each saved prefix only once while caching exact dynamic states, verifying every '
        'recorded desired velocity, integrated position and decoded actor action. Save full '
        'physical suffixes and failure charges. This collection changes temporal support '
        'only relative to the sparse nominal probe. Other48 paths retain existing sparse '
        'credit in the subsequent matched correction. No current-head attribution claims '
        'are made from an episode-wide scalar. Stop at budget and analyze; no deployment, '
        'new geometry, development or sealed testing. A new correction may proceed only '
        'after complete collection and finite signal review, never from failed prefixes.')
    started=time.perf_counter();costs=dict(actor_calls=0,new_physics_ticks=0,saved_prefix_geometry_ticks=0)
    p=LocalGeometryFeedbackPolicy(mask_endpoint=True)
    p.load_state_dict(torch.load(parent/'candidate_1.pt',map_location='cpu',weights_only=False)['policy'])
    old_results=read(old/'results.json');lookup={(r['path'],r['anchor'],r['axis'],r['direction']):r for r in old_results['records']}
    records=[];queries=[];observations=[];new_count=0;reused_count=0
    status_path=BASE/'当前执行状态.json';status=read(status_path)
    status['current'].update(state='dense_corner_complete_credit_collection_running',training_running=False,
        physical_diagnostic_running=True,diagnostic_stage=stage.name)
    status['next']='Finish all-time nominal credit collection, compare matched sparse/dense learned updates using retained strong model and complete XY review.'
    status['updated_utc']=datetime.now(timezone.utc).isoformat();write_json(status_path,status)
    for path in range(16):
        with np.load(pool/f'{path}.npz') as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        obs=[];states=[]
        env=reconstruct(ref,rows,actions,(len(actions)-1)*20,costs,observations=obs,state_snapshots=states)
        obs.append(env.observation());states.append(env.state());assert len(obs)==len(actions)==len(states)
        for anchor,(observation,state) in enumerate(zip(obs,states)):
            env.restore(state);assert np.array_equal(observation,env.observation())
            latent=p.act(observation,True)[0];costs['actor_calls']+=1
            assert np.array_equal(env.decode_action(latent),actions[anchor])
            qid=len(queries);observations.append(observation)
            queries.append(dict(query=qid,path=path,anchor=anchor,path_objective_gradient_weight=1/64))
            for axis in range(3):
                for direction in (-1,1):
                    key=(path,anchor,axis,direction)
                    if key in lookup:
                        row=dict(lookup[key]);row.update(query=qid,reused=True,trace=str((old/row['trace']).relative_to(BASE)))
                        records.append(row);reused_count+=1;continue
                    env.restore(state);chunks=[];held=[];a=latent.astype(float).copy();a[axis]+=direction*AMPLITUDE[axis]
                    while True:
                        assert time.perf_counter()-started<1500
                        assert costs['actor_calls']<160000 and costs['new_physics_ticks']<3797880
                        _,_,_,info=env.step(a);chunks.append(info['rows']);held.append(info['held_action'])
                        costs['new_physics_ticks']+=len(info['rows'])
                        if env.done:break
                        a=p.act(env.observation(),True)[0];costs['actor_calls']+=1
                    suffix=np.vstack(chunks);suffix_actions=np.asarray(held)
                    full=np.vstack([rows[:anchor*20],suffix]);all_actions=np.vstack([actions[:anchor],suffix_actions])
                    audit=difference_audit(ref[0],full);success=bool(env.success and audit['kinematic_pass'] and audit['rest_pass'])
                    name=f'p{path}_a{anchor}_h{axis}_{direction:+d}.npz'
                    np.savez_compressed(stage/name,reference=ref,suffix_rows=suffix,suffix_actions=suffix_actions,
                        baseline_path=path,anchor=anchor,observation=observation,axis=axis,latent_offset=direction*AMPLITUDE[axis])
                    records.append(dict(query=qid,path=path,anchor=anchor,axis=axis,direction=direction,
                        latent_offset=float(direction*AMPLITUDE[axis]),success=success,reason=env.reason,
                        full_time_s=len(full)*.001 if success else None,audit=audit,reused=False,
                        trace=str((stage/name).relative_to(BASE)),**score_complete(ref,full,all_actions,env)))
                    new_count+=1
            if anchor%15==0:print('DENSE_AXIAL',path,anchor,len(actions),'new',new_count,'calls',costs['actor_calls'],flush=True)
        write_json(stage/'results.json',dict(records=records,queries=queries,costs=costs,completed_paths=path+1,
            new_suffixes=new_count,reused_suffixes=reused_count,goal_complete=False))
        np.savez_compressed(stage/'queries.npz',observation=np.asarray(observations))
    sensitivities=[]
    by_query={q['query']:[] for q in queries}
    for row in records:by_query[row['query']].append(row)
    baselines={r['path']:r for r in old_results['baselines']}
    for query in queries:
        branches=by_query[query['query']];values={}
        for objective in ('original_full_return','independent_zero_return'):
            raw=[];complete=[];modes=[]
            for axis in range(3):
                pair=sorted([r for r in branches if r['axis']==axis],key=lambda r:r['direction'])
                assert len(pair)==2
                raw.append((pair[1][objective]-pair[0][objective])/(2*AMPLITUDE[axis]))
                good=[r for r in pair if r['success']]
                if len(good)==2:g=raw[-1];mode='central_complete'
                elif len(good)==1:g=(good[0][objective]-baselines[query['path']][objective])/good[0]['latent_offset'];mode='feasible_one_sided'
                else:g=0.;mode='both_sides_failed'
                complete.append(g);modes.append(mode)
            values[objective]=dict(raw_finite_gradient=raw,complete_side_gradient=complete,modes=modes)
        sensitivities.append(dict(query,objectives=values))
    write_json(stage/'sensitivities.json',dict(records=sensitivities))
    receipt(stage,dict(status='completed_dense_corner_credit_collection',costs=costs,queries=len(queries),
        new_suffixes=new_count,reused_suffixes=reused_count,complete_suffixes=sum(r['success'] for r in records),
        wall_s=time.perf_counter()-started))
    status=read(status_path);status['current'].update(state='dense_corner_credit_complete',physical_diagnostic_running=False)
    status['completed'].append(dict(stage=stage.name,receipt_sha256=sha(stage/'receipt.json')))
    status['updated_utc']=datetime.now(timezone.utc).isoformat();write_json(status_path,status)
    print('DENSE_DONE',new_count,reused_count,costs,flush=True)


if __name__=='__main__':main()
