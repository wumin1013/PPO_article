"""Current complete-path local models, reusing the completed two-corner pilot."""
import argparse
import time
import os
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy, checkpoint_environment_factory
from probe_coordinated_suffix import reconstruct
from closed_loop_transition_credit import transition_jacobian
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference',default='canonical_gate_band_coherent_parameter_credit_20260917')
    parser.add_argument('--stage',default='canonical_gate_band_complete_transition_credit_20260917')
    parser.add_argument('--resume-from')
    parser.add_argument('--smooth-actor-differences',action='store_true')
    parser.add_argument('--reuse-smooth-pilot')
    parser.add_argument('--paths',type=int,nargs='+')
    args=parser.parse_args()
    torch.set_num_threads(1)
    source=BASE/args.reference
    pilot=BASE/'canonical_gate_band_closed_loop_tangent_credit_20260917'
    data=read(source/'results.json');model=BASE/data['parent_model']
    metadata=torch.load(model,map_location='cpu',weights_only=False)
    assert sha(model)==data['parent_sha256']
    reuse_pilot=not args.smooth_actor_differences and sha(model)==read(pilot/'results.json')['parent_sha256']
    assert read(pilot/'receipt.json')['status']=='completed_closed_loop_tangent_credit_comparison'
    ids=[r['original_path'] for r in data['baseline']['results']]
    reference_ids=ids.copy()
    if args.paths:
        assert set(args.paths)<=set(ids) and len(set(args.paths))==len(args.paths)
        ids=list(args.paths)
    prior=None;extra_sources=[BASE/'terminal_duration_credit.py'];reused_records={}
    wall_budget=600;actor_budget=36000;physics_budget=1100000;saved_budget=65000
    smooth_pilot=None;smooth_records={}
    if args.smooth_actor_differences:
        wall_budget=900;actor_budget=45000
        extra_sources += [BASE/'smooth_credit_policy.py']
    if args.paths:
        assert len(ids)<=2 and not args.resume_from and not args.reuse_smooth_pilot
        wall_budget=180;actor_budget=4500;physics_budget=120000;saved_budget=10000
    if args.reuse_smooth_pilot:
        assert args.smooth_actor_differences
        smooth_pilot=BASE/args.reuse_smooth_pilot;smooth_data=read(smooth_pilot/'results.json')
        assert smooth_data['parent_sha256']==sha(model)
        assert read(smooth_pilot/'receipt.json')['status']=='completed_centered_smooth_credit_precision_audit'
        smooth_records={r['path']:r for r in smooth_data['records']
            if r['precision']=='centered_float64' and r['epsilon']==1e-4}
        extra_sources += [smooth_pilot/'results.json',smooth_pilot/'receipt.json']
    if args.resume_from:
        prior=BASE/args.resume_from;prior_data=read(prior/'results.json');prior_receipt=read(prior/'receipt.json')
        assert prior_data['parent_sha256']==sha(model)
        assert prior_receipt['status']=='failed_current_complete_transition_credit'
        assert read(prior/'registration.json')['source_sha256'][(source/'results.json').relative_to(BASE.parent.parent).as_posix()]==sha(source/'results.json')
        reused_records={r['path']:r for r in prior_data['records']}
        assert set(reused_records)<set(ids)
        assert read(prior/'registration.json')['budget'].get('smooth_actor_differences',False)==args.smooth_actor_differences
        extra_sources += [prior/'results.json',prior/'receipt.json',prior/'registration.json']
        wall_budget=420;actor_budget=16000;physics_budget=400000;saved_budget=30000
    stage=register(args.stage,
        [__file__,model,source/'results.json',source/'receipt.json',
         pilot/'results.json',pilot/'receipt.json',pilot/'path14_eps0.0001.npz',
         pilot/'path15_eps0.0001.npz',BASE/'closed_loop_transition_credit.py',
         BASE/'probe_closed_loop_tangent_credit.py',BASE/'canonical_gate_state.py',
         BASE/'gate_band_state.py',BASE/'joint_runtime.py',BASE/'probe_coordinated_suffix.py']+extra_sources+
         ([source/'basis.npz'] if (source/'basis.npz').exists() else []),
        dict(training_paths=ids,finite_difference_scale=1e-4,actor_calls=actor_budget,
             new_physics_ticks=physics_budget,saved_prefix_geometry_ticks=saved_budget,wall_s=wall_budget,
             completed_models_reused=sorted(reused_records),resume_from=args.resume_from,
             smooth_actor_differences=args.smooth_actor_differences,reused_smooth_models=sorted(smooth_records),
             prior_actual_costs=prior_receipt['costs'] if prior else None,
             reference_paths=reference_ids,diagnostic_subset=bool(args.paths),
             reused_ordinary_path_models=[14,15] if reuse_pilot else [],policy_updates=0),
        ('Correct the demonstrated precision-sensitive credit across the entire current '
         'screen, not only its most recently active constraints. Float64 observation '
         'and policy DIFFERENCES are centered on each exact deployed float32 action. '
         'The policy used in complete evaluation remains float32. Reuse the two '
         'completed current-parent smooth local models by hash. Same features, '
         'finite scale, physical executor and metric; no error-rate inflation. '
         if args.smooth_actor_differences else '')+
        ('Reuse the already completed bound path-model files from the interrupted collection, '
         'recording their exact hashes and original locations without repeating their physics. '
         'Only missing paths are collected under this separate remaining-work budget. The '
         'failed receipt and its actual costs remain unchanged. ' if prior else '')+
        'The bounded two-path pilot has stable local derivatives at two scales and '
        'heading errors about3.5%/6% of measured finite responses. Collect this '
        'different credit construction over the explicitly bound current29 training screen. '
        'Reuse exact ordinary transition Jacobians for paths14/15 ONLY if the '
        'parent checkpoint hash matches that pilot; otherwise recompute all. Include terminal '
        'state derivatives only when the actual same terminal initiation remains '
        'valid. Align already stopped tails solely for derivatives, never time '
        'scoring. Record one-sided physical-boundary derivatives explicitly and '
        'fail the model if both signs are invalid. Store local Jacobians and '
        'current policy features, allowing all shared parameters to be chained '
        'without additional physical probes. This is not an expansion of the '
        'failed six-history-direction search or additional training. No time '
        'derivative through discrete completion, no policy adoption, changed '
        'physics, development or sealed access. After completion compile and '
        'validate complete waveform responses against the registered reused variants '
        'before a bounded shared-parameter learning solve.')
    start=time.perf_counter();costs=dict(actor_calls=0,new_physics_ticks=0,saved_prefix_geometry_ticks=0)
    policy=CanonicalGatePolicy(256);policy.load_state_dict(metadata['policy'])
    if args.smooth_actor_differences:
        from smooth_credit_policy import CenteredSmoothActor
        smooth_actor=CenteredSmoothActor(policy)
    factory=checkpoint_environment_factory(metadata['architecture']);records=[]
    trace_by_path={r['original_path']:t for r,t in zip(data['baseline']['results'],data['baseline']['traces'])}
    assert set(trace_by_path)==set(reference_ids)
    def check():
        if time.perf_counter()-start>wall_budget-10:raise TimeoutError('Complete-transition credit wall budget')
        assert costs['actor_calls']<actor_budget and costs['new_physics_ticks']<physics_budget-2120
        assert costs['saved_prefix_geometry_ticks']<=saved_budget
    try:
        for path in ids:
            if path in smooth_records:
                old=smooth_records[path];old_file=smooth_pilot/old['file'];assert old_file.is_file()
                with np.load(old_file) as z:decisions=len(z['state_jacobian'])
                records.append(dict(path=path,file=os.path.relpath(old_file,stage),decisions=decisions,
                    derivative_epsilon=1e-4,derivative_precision='centered_float64',
                    one_sided=old['one_sided'],invalid_queries=old['invalid_queries'],
                    source_local_model_sha256=sha(old_file),reused_from_stage=smooth_pilot.name))
                continue
            if path in reused_records:
                old=reused_records[path];old_file=(prior/old['file']).resolve();assert old_file.is_relative_to(BASE)
                records.append(dict(old,file=os.path.relpath(old_file,stage),reused_from_stage=prior.name,reused_file_sha256=sha(old_file)))
                continue
            with np.load(BASE/trace_by_path[path]) as z:
                ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
            expected=np.asarray(read(BASE/'learning_preparation_v4/paths.json')['training'][path])
            assert np.array_equal(ref,expected),f'Path identity mismatch: {path}'
            states=[]
            env=reconstruct(ref,rows,actions,(len(actions)-1)*20,costs,
                            state_snapshots=states,env_factory=factory)
            states.append(env.state());env=factory(ref)
            reused=None
            if reuse_pilot and path in (14,15):
                with np.load(pilot/f'path{path}_eps0.0001.npz') as z:
                    reused=(z['state_jacobian'].copy(),z['local_row_jacobian'].copy())
                assert len(reused[0])==len(actions)-1
            sjs=[];rjs=[];features=[];observations=[];modes=[];invalid=[];offsets=[0]
            for k,state in enumerate(states):
                check();env.restore(state);env.observation_dtype=np.float32;obs=env.observation()
                latent=policy.act(obs,True)[0].astype(float);costs['actor_calls']+=1
                assert np.array_equal(env.decode_action(latent),actions[k])
                with torch.no_grad():
                    features.append(policy.residual_features(torch.tensor(obs)).numpy()*
                                    policy.residual_scale.numpy()[:,None])
                observations.append(obs)
                terminal=k==len(states)-1
                nominal_rows=rows[k*20:] if terminal else rows[k*20:(k+1)*20]
                assert bool(nominal_rows[:,12].all())==terminal
                if reused is not None and not terminal:
                    sj,rj=reused[0][k],reused[1][k];local_modes=['reused_central']*10;bad=[]
                else:
                    derivative_policy=policy
                    if args.smooth_actor_differences:
                        env.observation_dtype=np.float64;smooth_actor.center(env.observation(),latent)
                        costs['actor_calls']+=1;derivative_policy=smooth_actor
                    sj,rj,local_modes,bad=transition_jacobian(env,derivative_policy,state,latent,nominal_rows,
                        None if terminal else states[k+1],1e-4,terminal,costs,check)
                sjs.append(sj);rjs.append(rj);modes.extend(local_modes)
                invalid.extend(dict(decision=k,**r) for r in bad);offsets.append(offsets[-1]+len(rj))
            assert offsets[-1]==len(rows)
            name=f'path{path}_local_model.npz'
            np.savez_compressed(stage/name,reference=ref,rows=rows,actions=actions,
                state_jacobian=np.asarray(sjs),row_jacobian=np.concatenate(rjs),
                row_offsets=offsets,features=np.asarray(features),observations=np.asarray(observations))
            records.append(dict(path=path,file=name,decisions=len(states),
                derivative_epsilon=1e-4,
                derivative_precision='centered_float64' if args.smooth_actor_differences else 'original_float32',
                central=modes.count('central'),reused_central=modes.count('reused_central'),
                one_sided=modes.count('one_sided'),invalid_queries=invalid))
            write_json(stage/'results.json',dict(parent_sha256=sha(model),parent_model=data['parent_model'],
                records=records,costs=costs,requested_paths=ids,reference_paths=reference_ids,goal_complete=False))
            print('COMPLETE_TRANSITION_CREDIT',path,'decisions',len(states),'one_sided',modes.count('one_sided'),
                  'ticks',costs['new_physics_ticks'],flush=True)
        receipt(stage,dict(status='completed_current_complete_transition_credit',paths=len(records),
            costs=costs,wall_s=time.perf_counter()-start,policy_updates=0))
    except Exception as exc:
        receipt(stage,dict(status='failed_current_complete_transition_credit',paths=len(records),
            error=str(exc),costs=costs,wall_s=time.perf_counter()-start,policy_updates=0))
        raise


if __name__=='__main__':main()
