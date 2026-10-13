"""Whole-feedback-policy credit in a measured six-direction update subspace."""
import time
import numpy as np
import torch
from scipy.linalg import block_diag
from canonical_gate_state import CanonicalGatePolicy,checkpoint_environment_factory
from diagnose_local_basis_fit import feature_batches
from probe_coordinated_suffix import reconstruct
from probe_one_decision_credit import AMPLITUDE
from learn_single_decision_credit import review
from probe_direct_constraint_composition import metric_rows
from train_pilot import evaluate,save_torch
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    torch.set_num_threads(1)
    parent=BASE/'canonical_gate_band_monotone_block_learning_20260917'
    data=read(parent/'results.json');chosen=data['chosen'];model=parent/chosen['model']
    metadata=torch.load(model,map_location='cpu',weights_only=False)
    factory=checkpoint_environment_factory(metadata['architecture'])
    sources=[BASE/'canonical_gate_band_quadratic_error_step_solve_20260917',
             BASE/'canonical_gate_band_half_step_measured_parameter_proposal_20260917']
    ids=[r['original_path'] for r in chosen['results']];assert len(ids)==29
    stage=register('canonical_gate_band_coherent_parameter_credit_20260917',
        [__file__,model,parent/'results.json',parent/'receipt.json',BASE/'canonical_gate_state.py',
         BASE/'gate_band_state.py',BASE/'continuous_gate_guidance.py',BASE/'train_pilot.py',
         BASE/'diagnose_local_basis_fit.py',BASE/'probe_coordinated_suffix.py']+
        [p/n for p in sources for n in ('direction.npz','optimizer.json','receipt.json')],
        dict(parameter_directions=6,signs=[-1,1],training_paths=ids,new_policy_paths=348,
             maximum_parameter_metric_norm=.025,maximum_saved_state_action_change=.125,
             evaluation_actor_calls=30000,evaluation_physics_ticks=650000,
             saved_geometry_ticks=65000,wall_s=360,optimizer_updates=0),
        'Replace isolated single-action consequence composition with direct '
        'whole-feedback-policy parameter responses at the NEW monotone64/64 '
        'research center. Six interpretable directions come from the two '
        'previously realized learned updates, separated by their three output '
        'heads and orthogonalized using a freshly reconstructed current-policy '
        'state-action metric. No path-dependent runtime rule or random search '
        'direction is introduced. Each probe changes one parameter direction '
        'for the entire rollout, with all heads responding to real state. '
        'Use both signs with norm<=.025 and saved-state normalized action '
        'change<=.125. Evaluate all29 existing screening paths per variant, '
        'retaining incomplete paths as failures. Current baseline is reused. '
        'This is a bounded credit-method experiment, not six-dimensional '
        'state compression or a final policy. After the12 variants, inspect '
        'time/shape/cruise signals and subspace feasibility before any update. '
        'Do not automatically add directions or continue training if this '
        'subspace fails. No development, sealed tests or strong-model replacement.')
    start=time.perf_counter();costs=dict(evaluation_actor_calls=0,evaluation_physics_ticks=0,
        saved_prefix_geometry_ticks=0,new_physics_ticks=0)
    policy=CanonicalGatePolicy(256);policy.load_state_dict(metadata['policy'])
    observations=[];weights=[]
    for r,trace in zip(chosen['results'],chosen['traces']):
        with np.load(BASE/trace) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        states=[];env=reconstruct(ref,rows,actions,(len(actions)-1)*20,costs,
            state_snapshots=states,env_factory=factory)
        states.append(env.state());env=factory(ref)
        for j,state in enumerate(states):
            env.restore(state);observation=env.observation()
            assert np.array_equal(env.decode_action(policy.act(observation,True)[0]),actions[j])
            observations.append(observation);weights.append(1/(29*len(states)))
    observations=np.asarray(observations);weights=np.asarray(weights)
    features=feature_batches(policy,observations)*policy.residual_scale.detach().numpy()[None,:,None]/AMPLITUDE[None,:,None]
    metric=block_diag(*[features[:,h].T@(weights[:,None]*features[:,h])/3+.001*np.eye(256) for h in range(3)])
    support=np.zeros((len(observations),3,768))
    for h in range(3):support[:,h,h*256:(h+1)*256]=features[:,h]
    support=support.reshape(-1,768)
    updates=[]
    for folder in sources:
        with np.load(folder/'direction.npz') as z:updates.append(z['direction'].reshape(3,256).copy())
    basis=[];descriptions=[]
    for head in range(3):
        local=[]
        for index,update in enumerate(updates):
            vector=np.zeros(768);vector[head*256:(head+1)*256]=update[head]
            for previous in local:vector-=previous*(previous@metric@vector)
            norm=float(np.sqrt(vector@metric@vector));assert norm>1e-8
            vector/=norm;local.append(vector);basis.append(vector)
            descriptions.append(dict(head=head,source_update=index,pre_normalization_norm=norm))
    basis=np.asarray(basis)
    assert np.max(abs(basis@metric@basis.T-np.eye(6)))<1e-8
    amplitudes=np.minimum(.025,.125/np.maximum(abs(support@basis.T).max(axis=0),1e-30))
    np.savez_compressed(stage/'basis.npz',basis=basis,amplitudes=amplitudes,metric=metric,
        support=support,observation=observations,weights=weights)
    write_json(stage/'basis.json',dict(directions=descriptions,amplitudes=amplitudes.tolist(),
        current_parent_sha256=sha(model),current_saved_states=len(observations)))
    paths=[np.asarray(p) for p in read(BASE/'learning_preparation_v4/paths.json')['training']]
    names=data['metric_names'];baseline=dict(summary=chosen['summary'],metrics=chosen['metrics'],
        results=chosen['results'],traces=chosen['traces'])
    records=[]
    for index in range(6):
        for sign in (-1,1):
            policy.load_state_dict(metadata['policy'])
            delta=sign*amplitudes[index]*basis[index].reshape(3,256)
            with torch.no_grad():policy.residual_coefficients.add_(torch.tensor(delta,dtype=torch.float32))
            tag=f'direction{index}_{sign:+d}'
            save_torch(stage/f'{tag}.pt',dict(metadata,policy=policy.state_dict(),
                parent_sha256=sha(model),diagnostic_only=True,parameter_direction=index,
                parameter_displacement=float(sign*amplitudes[index])))
            rs=evaluate(policy,[paths[p] for p in ids],stage,tag,None,costs,start+360,factory)
            assert costs['evaluation_actor_calls']<=30000 and costs['evaluation_physics_ticks']<=650000
            assert costs['saved_prefix_geometry_ticks']<=65000
            rs=[dict(r,original_path=p) for r,p in zip(rs,ids)]
            ts=[stage/tag/f'{j}.npz' for j in range(len(ids))]
            summary=review(rs,ts);metrics=metric_rows(rs,ts,summary,names)
            record=dict(direction=index,sign=sign,amplitude=float(amplitudes[index]),
                model=f'{tag}.pt',summary=summary,metrics=metrics,results=rs,
                traces=[str(t.relative_to(BASE)) for t in ts])
            records.append(record)
            write_json(stage/'results.json',dict(parent_sha256=sha(model),parent_model=str(model.relative_to(BASE)),
                baseline=baseline,candidates=records,metric_names=names,costs=costs,
                representation_dimension_unchanged=True,diagnostic_only=True,goal_complete=False))
            print('COHERENT_PARAMETER_CREDIT',index,sign,summary['successes'],summary['entry_sum_mm'],
                summary['exit_sum_mm'],summary['full_reverse_sum_deg'],summary['full_time_s'],flush=True)
    receipt(stage,dict(status='completed_current_coherent_parameter_credit',variants=len(records),
        complete_paths=sum(r['summary']['successes'] for r in records),costs=costs,
        wall_s=time.perf_counter()-start,goal_complete=False))


if __name__=='__main__':main()
