"""Verify complete-objective identity and actor invariance under old paired credit."""
import copy
import time
import numpy as np
import torch
from causal_complete_shape import causal_shape_pieces, CausalCompleteShape
from conditional_feedback_policy import ConditionalLocalGeometryPolicy
from probe_one_decision_credit import score_complete
from train_bounded_shape_ppo import update
from search_complete_return import read
from study_io import BASE, register, receipt, write_json
from train_pilot import save_torch


def main():
    torch.set_num_threads(1)
    replay=BASE/'preswing_learning_origin_replay_20260916'
    history=BASE/'local_geometry_continuous_reward_ppo_20260916'
    pool=BASE/'exit_local_credit_parameter_learning_20260916/training_1'
    probes=BASE/'single_decision_complete_credit_probe_20260917'
    stage=register('causal_complete_shape_accounting_audit_20260917',
        [__file__,BASE/'causal_complete_shape.py',BASE/'probe_one_decision_credit.py',
         BASE/'complete_corner_metrics.py',BASE/'geometric_turn_metrics.py',
         BASE/'train_bounded_shape_ppo.py',BASE/'conditional_feedback_policy.py',
         replay/'before_update_1.pt',replay/'receipt.json',history/'episode_results.json',
         pool/'summary.json',probes/'receipt.json',probes/'results.json'],
        dict(new_physics_ticks=0,new_training_decisions=0,optimizer_replays=2,wall_s=120),
        'Audit causal dense accounting for the already-declared independent-zero complete '
        'shape cost. Running spatial energy, independent side maxima and signed physical '
        'reversal are monotone; charge changes of1-exp(-Q) at the decision when information '
        'first becomes available. No retrospectively selected window or future normalization. '
        'Verify complete scalar identity on all64 retained training trajectories and the '
        'historical first batch, and exact prefix independence on32 reused single-decision '
        'branches. Keep all base-time/failure costs and isolated-only coverage limitation. '
        'Replay two same-batch/same-Adam/same-RNG PPO updates:terminal versus dense allocation '
        'with identical shape totals and original whole-episode paired actor advantages. '
        'Expected actor equality explicitly checks that accounting alone cannot fix this '
        'actor credit mechanism. Critic targets may differ. No candidate adoption or physics.')
    started=time.perf_counter();cases=[];max_error=0.;prefix_checks=[]
    baseline_arrays={}
    for path in range(64):
        with np.load(pool/f'{path}.npz') as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        pieces,acc=causal_shape_pieces(ref,rows,len(actions))
        expected=score_complete(ref,rows,actions)['independent_shape_cost']
        error=abs(pieces.sum()-expected);max_error=max(max_error,float(error))
        assert error<1e-9
        cases.append(dict(path=path,cost=float(pieces.sum()),error=float(error),
            nonzero_decisions=int((pieces>1e-12).sum()),decisions=len(actions)))
        if path<16:baseline_arrays[path]=(ref,rows,actions,pieces)
    data=read(probes/'results.json')
    for path in range(16):
        ref,rows,actions,base_pieces=baseline_arrays[path]
        first=next(q for q in data['queries'] if q['path']==path)
        anchor=first['anchor']
        for record in [r for r in data['records'] if r['query']==first['query'] and r['axis']==0]:
            with np.load(probes/record['trace']) as z:
                suffix=z['suffix_rows'];held=z['suffix_actions']
            full=np.vstack([rows[:anchor*20],suffix]);all_actions=np.vstack([actions[:anchor],held])
            pieces,acc=causal_shape_pieces(ref,full,len(all_actions))
            assert np.array_equal(pieces[:anchor],base_pieces[:anchor])
            error=abs(pieces.sum()-record['independent_shape_cost']);max_error=max(max_error,float(error))
            assert error<1e-9
            prefix_checks.append(dict(path=path,anchor=anchor,trace=record['trace'],prefix_exact=True,complete_cost_error=float(error)))
    before=torch.load(replay/'before_update_1.pt',map_location='cpu',weights_only=False)
    meta=read(history/'episode_results.json')['episodes']
    loaded=[]
    for i in before['episode_numbers']:
        with np.load(history/'episodes'/meta[i]['trace']) as z:d={k:z[k].copy() for k in z.files}
        pieces,acc=causal_shape_pieces(d['reference'],d['rows'],len(d['actions']))
        expected=score_complete(d['reference'],d['rows'],d['actions'])['independent_shape_cost']
        assert abs(pieces.sum()-expected)<1e-9
        loaded.append((d,pieces))
    states={};records=[];policy=ConditionalLocalGeometryPolicy(72,mask_endpoint=True)
    for allocation in ('terminal','causal_dense'):
        batch=[]
        for d,pieces in loaded:
            e={k:d[k].copy() for k in ('observation','latent','previous_observation','previous_latent','context_live','old_log','value','terminal')}
            e['reward']=d['base_reward'].copy()
            if allocation=='terminal':e['reward'][-1]-=pieces.sum()
            else:e['reward']-=pieces
            batch.append(e)
        for index in range(0,len(batch),4):
            group=batch[index:index+4]
            totals=np.asarray([e['reward'].sum() for e in group])
            advantages=(totals-(totals.sum()-totals)/3)/max(float(totals.std()),.02)
            for e,a in zip(group,advantages):e['paired_advantage']=np.full(len(e['reward']),a)
        policy.load_state_dict(before['policy'])
        optim={'actor':torch.optim.Adam(policy.actor_parameters(),lr=3e-5),
               'critic':torch.optim.Adam(policy.critic.parameters(),lr=.001)}
        for k,v in optim.items():v.load_state_dict(copy.deepcopy(before['optimizer'][k]))
        torch.set_rng_state(before['torch_rng'])
        statistics=update(policy,optim,batch,paired_advantage=True)
        states[allocation]=copy.deepcopy(policy.state_dict())
        save_torch(stage/f'{allocation}.pt',dict(policy=policy.state_dict(),diagnostic_only=True))
        records.append(dict(allocation=allocation,statistics=statistics,
            episode_returns=[float(e['reward'].sum()) for e in batch]))
    actor_equal=all(torch.equal(states['terminal'][k],states['causal_dense'][k]) for k in states['terminal'] if k.startswith('actor.'))
    critic_equal=all(torch.equal(states['terminal'][k],states['causal_dense'][k]) for k in states['terminal'] if k.startswith('critic.'))
    assert actor_equal
    assert time.perf_counter()-started<120
    write_json(stage/'results.json',dict(baselines=cases,prefix_checks=prefix_checks,optimizer_replays=records,
        actor_tensors_bitwise_equal=actor_equal,critic_tensors_bitwise_equal=critic_equal,
        max_complete_cost_error=max_error,goal_complete=False,
        implication='Causal allocation is valid but original paired actor overwrites reward-to-go credit. A learning remedy must change actor credit, not just reward timing.'))
    receipt(stage,dict(status='completed_causal_complete_shape_accounting_audit',new_physics_ticks=0,
        new_training_decisions=0,optimizer_replays=2,actor_bitwise_equal=actor_equal,
        max_complete_cost_error=max_error,wall_s=time.perf_counter()-started))
    print('CAUSAL_ACCOUNTING',max_error,'actor_equal',actor_equal,'critic_equal',critic_equal,flush=True)


if __name__=='__main__':main()
