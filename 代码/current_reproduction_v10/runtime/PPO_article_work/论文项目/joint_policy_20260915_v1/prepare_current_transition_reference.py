"""Fresh current-state metric and exact reuse of already evaluated policies."""
import argparse
import time
import numpy as np
import torch
from scipy.linalg import block_diag
from canonical_gate_state import CanonicalGatePolicy,checkpoint_environment_factory
from diagnose_local_basis_fit import feature_batches
from probe_coordinated_suffix import reconstruct
from probe_one_decision_credit import AMPLITUDE
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',default='canonical_gate_band_transition_measured_margin_learning_20260917')
    parser.add_argument('--stage',default='canonical_gate_band_bend_learning_reference_20260917')
    parser.add_argument('--comparisons',nargs='+')
    parser.add_argument('--diagnostic-model')
    args=parser.parse_args()
    torch.set_num_threads(1)
    parent=BASE/args.parent
    data=read(parent/'results.json')
    chosen=(next(c for c in data['candidates'] if c['model']==args.diagnostic_model)
            if args.diagnostic_model else data['chosen'])
    model=parent/chosen['model']
    metadata=torch.load(model,map_location='cpu',weights_only=False)
    if args.diagnostic_model:
        assert chosen['summary']['comparable_complete_pool']
        assert all(r['success'] and r['audit']['rest_pass'] and
                   np.all(np.asarray(r['audit']['peaks'])<=[100.,2000.,62500.]) for r in chosen['results'])
        assert args.comparisons
    else:
        assert data['full_candidate']['checks']['eligible']
    old_parent=BASE/'canonical_gate_band_monotone_block_learning_20260917'
    old=read(old_parent/'results.json')['chosen']
    failed=BASE/'canonical_gate_band_transition_credit_learned_update_20260917'
    failed_data=read(failed/'results.json')
    if args.comparisons:
        comparisons=[]
        for item in args.comparisons:
            other_model=BASE/item;folder=other_model.parent
            candidate=next(c for c in read(folder/'results.json')['candidates'] if c['model']==other_model.name)
            comparisons.append((folder,candidate))
        assert len(comparisons)==4
    else:
        comparisons=[(old_parent,old)]+[(failed,c) for c in failed_data['candidates']]+[
            (parent,next(c for c in data['candidates'] if c['step']==.5))]
    stage=register(args.stage,
        [__file__,model,parent/'results.json',parent/'receipt.json',BASE/'canonical_gate_state.py',
         BASE/'diagnose_local_basis_fit.py',BASE/'probe_coordinated_suffix.py']+
        [f/n for f,c in comparisons for n in (c['model'],'results.json','receipt.json')],
        dict(new_physics_ticks=0,saved_geometry_ticks=65000,replayed_policy_checks=3000,
             reused_complete_policy_variants=4,wall_s=90,diagnostic_center=bool(args.diagnostic_model)),
        ('Use an explicitly UNADOPTED center that completed every screen path '
         'under strict physical limits. Its failed shape/time adoption criteria '
         'remain failed; current399a remains the retained comparator. Recenter '
         'a local-model diagnostic on its saved actual states, without replaying '
         'completed paths or transferring old-center derivatives. ' if args.diagnostic_model else
         'The selected learned step passed64/64 but keeps concentrated curvature ')+
        'and insufficient cruise. Recenter the next mechanism experiment on its '
        'actual saved29-route state support, recomputing the parameter metric. '
        'Bind four already evaluated older policies by their EXACT coefficient '
        'differences from this new center, checking that every other parameter '
        'and runtime encoding is identical. These responses are validation '
        'directions only, not a new restricted optimization space or new '
        'physical experiments. The subsequent current-state transition credit '
        'will support the next declared mechanism experiment using the EXISTING '
        'bending-energy and shape quantities, without changing physical limits or '
        'deploying a geometric turn template. No old-center derivatives reused.')
    start=time.perf_counter();costs=dict(saved_prefix_geometry_ticks=0,new_physics_ticks=0,replayed_policy_checks=0)
    policy=CanonicalGatePolicy(256);policy.load_state_dict(metadata['policy'])
    factory=checkpoint_environment_factory(metadata['architecture']);obs=[];weights=[]
    ids=[r['original_path'] for r in chosen['results']]
    for trace in chosen['traces']:
        with np.load(BASE/trace) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        states=[];env=reconstruct(ref,rows,actions,(len(actions)-1)*20,costs,
                                 state_snapshots=states,env_factory=factory)
        states.append(env.state());env=factory(ref)
        for j,state in enumerate(states):
            env.restore(state);o=env.observation();costs['replayed_policy_checks']+=1
            assert np.array_equal(env.decode_action(policy.act(o,True)[0]),actions[j])
            obs.append(o);weights.append(1/(len(ids)*len(states)))
    obs=np.asarray(obs);weights=np.asarray(weights)
    features=feature_batches(policy,obs)*policy.residual_scale.numpy()[None,:,None]/AMPLITUDE[None,:,None]
    metric=block_diag(*[features[:,h].T@(weights[:,None]*features[:,h])/3+.001*np.eye(256) for h in range(3)])
    support=np.zeros((len(obs),3,768))
    for h in range(3):support[:,h,h*256:(h+1)*256]=features[:,h]
    basis=[];amplitudes=[];candidates=[]
    for j,(folder,c) in enumerate(comparisons):
        saved=torch.load(folder/c['model'],map_location='cpu',weights_only=False)
        assert saved['architecture']==metadata['architecture']
        assert set(saved['policy'])==set(metadata['policy'])
        for key in saved['policy']:
            if key!='residual_coefficients':assert torch.equal(saved['policy'][key],metadata['policy'][key]),key
        direction=(saved['policy']['residual_coefficients'].double()-metadata['policy']['residual_coefficients'].double()).numpy().ravel()
        norm=float(np.sqrt(direction@metric@direction));assert norm>0
        basis.append(direction/norm);amplitudes.append(norm)
        assert [r['original_path'] for r in c['results']]==ids and c['summary']['successes']==len(ids)
        candidates.append(dict(direction=j,sign=1,amplitude=norm,model=str((folder/c['model']).relative_to(BASE)),
            model_sha256=sha(folder/c['model']),**{k:c[k] for k in ('summary','metrics','results','traces')}))
    np.savez_compressed(stage/'basis.npz',basis=np.array(basis),amplitudes=amplitudes,metric=metric,
        support=support.reshape(-1,768),observation=obs,weights=weights)
    baseline={k:chosen[k] for k in ('summary','metrics','results','traces')}
    write_json(stage/'results.json',dict(parent_sha256=sha(model),parent_model=str(model.relative_to(BASE)),
        baseline=baseline,candidates=candidates,metric_names=data['metric_names'],
        diagnostic_center=bool(args.diagnostic_model),center_adopted=False if args.diagnostic_model else None,
        directions_are_exact_history_differences=True,new_policy_training=False,goal_complete=False))
    assert costs['saved_prefix_geometry_ticks']<=65000 and costs['replayed_policy_checks']<=3000
    receipt(stage,dict(status='prepared_current_transition_learning_reference',costs=costs,
        reused_variants=len(candidates),wall_s=time.perf_counter()-start,goal_complete=False))
    print('CURRENT_REFERENCE',sha(model),'norms',amplitudes,flush=True)


if __name__=='__main__':main()
