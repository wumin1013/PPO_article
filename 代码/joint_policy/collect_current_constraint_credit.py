"""Current-policy all-time credit for direct constrained parameter learning."""
import argparse
import time
from datetime import datetime,timezone
import numpy as np
import torch
from kernel_local_feedback_policy import KernelLocalFeedbackPolicy
from probe_coordinated_suffix import reconstruct
from probe_one_decision_credit import AMPLITUDE,score_complete
from audit_function_constraint_interference import command_segment_margin
from validate_actions import difference_audit
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage')
    parser.add_argument('--paths',type=int,nargs='+')
    parser.add_argument('--amplitude-scale',type=float,default=1.)
    parser.add_argument('--wall-s',type=float,default=1800.)
    parser.add_argument('--actor-budget',type=int,default=200000)
    parser.add_argument('--tick-budget',type=int,default=4800000)
    parser.add_argument('--continuous-screen-source')
    parser.add_argument('--arm',default='updated')
    parser.add_argument('--reuse-certificate')
    args=parser.parse_args()
    assert 0<args.amplitude_scale<=1
    if args.paths is not None or args.amplitude_scale!=1:
        assert args.stage,'Changed credit support requires a new registered stage'
    torch.set_num_threads(1)
    parent=BASE/'recentered_reverse_shape_credit_learning_20260917'
    selected=read(parent/'training_selection.json');model=parent/selected['model']
    selection_file=parent/'training_selection.json'
    policy_class=KernelLocalFeedbackPolicy
    current_factory=None
    if args.continuous_screen_source:
        assert args.stage and args.paths, 'A new sensor/parent requires explicit stage and paths'
        from continuous_sensor_runtime import environment_factory as continuous_factory
        parent=BASE/args.continuous_screen_source
        selection_file=parent/'results.json'
        candidate=next(r for r in read(selection_file)['candidates']
            if r.get('arm',r.get('tag'))==args.arm or r.get('model')==args.arm)
        selected=dict(model=candidate['model'],
            results={r['original_path']:r for r in candidate['results']},
            traces={r['original_path']:t for r,t in zip(candidate['results'],candidate['traces'])})
        model=parent/selected['model']
        metadata=torch.load(model,map_location='cpu',weights_only=False)
        current_factory=continuous_factory
        if metadata['architecture'] in ('canonical_gate_kernel','gate_band_canonical_kernel'):
            from canonical_gate_state import CanonicalGatePolicy,environment_factory as canonical_factory
            policy_class=CanonicalGatePolicy
            current_factory=canonical_factory
            if metadata['architecture']=='gate_band_canonical_kernel':
                from gate_band_state import environment_factory as gate_band_factory
                current_factory=gate_band_factory
        else:
            assert metadata['observation_encoding']=='continuous_centered_reference_mean_tangent_2mm'
    paths=args.paths if args.paths is not None else list(range(16))+[40]
    assert len(set(paths))==len(paths) and all(0<=p<64 for p in paths)
    amplitude=AMPLITUDE*args.amplitude_scale
    reuse={};reuse_sources=[];reuse_count=0;new_count=0
    if args.reuse_certificate:
        certificate=BASE/args.reuse_certificate;cert=read(certificate/'results.json')
        assert (certificate/'receipt.json').exists() and cert['parent_sha256']==sha(model)
        assert metadata['architecture']=='gate_band_canonical_kernel' and cert['same_policy_tensors']
        original=BASE/cert['source'];old_rows=read(original/'results.json')['records']
        lookup={(r['path'],r['anchor'],r['axis'],r['direction']):r for r in old_rows}
        for c in cert['records']:
            if not c['reusable']:continue
            key=(c['path'],c['anchor'],c['axis'],c['direction']);r=lookup[key]
            assert c['trace']==r['trace'] and sha(BASE/r['trace'])==c['trace_sha256']
            assert r['latent_offset']==c['direction']*amplitude[c['axis']]
            reuse[key]=r
        reuse_sources=[certificate/'results.json',certificate/'receipt.json',original/'results.json',
                       original/'receipt.json',BASE/'certify_gate_band_credit_reuse.py',BASE/'gate_band_state.py']
    query_count=0
    for path in paths:
        with np.load(BASE/selected['traces'][path]) as z:query_count+=len(z['actions'])
    stage=register(args.stage or 'current_dense_constraint_credit_collection_20260917',
        [__file__,model,selection_file,parent/'receipt.json',
         BASE/'constraint_normal_smoothness_credit_learning_20260917/optimizer.json',
         BASE/'headwise_smoothness_preserving_credit_learning_20260917/optimizer.json',
         BASE/'probe_coordinated_suffix.py',BASE/'probe_one_decision_credit.py',
         BASE/'kernel_local_feedback_policy.py',BASE/'audit_function_constraint_interference.py',
         BASE/'validate_actions.py']+([BASE/'continuous_sensor_runtime.py',
         BASE/'large_step_single_sensor_causal_probe_20260917/receipt.json',
         BASE/'canonical_gate_state.py',BASE/'continuous_gate_guidance.py',BASE/'joint_runtime.py'] if args.continuous_screen_source else [])+reuse_sources,
        dict(paths=paths,queries=query_count,axial_suffixes=6*query_count,latent_amplitude=amplitude.tolist(),
             actor_calls=args.actor_budget,new_physics_ticks=args.tick_budget,wall_s=args.wall_s,optimizer_updates=0,
             exact_reuse_certificate=args.reuse_certificate,certified_available_suffixes=len(reuse)),
        ('The gate-band state-support repair changes observation eligibility only. '
         'This collection uses its exact runtime and unchanged parent policy tensors. '
         'The attached trajectory-wise certificate proves equivalence for reusable '
         'old consequences by identical nominal prefixes and identical observation '
         'support at every branch decision; every accepted suffix hash is checked. '
         'Collect all other branches afresh. Recompute current nominal observations '
         'and subsequent parameter feature Jacobians; do not label uncertified old '
         'gradients as current. Explicit paths, counts and budgets apply. Same one '
         '20ms three-axis interventions, full braking/rest, original physical limits '
         'and coverage. Failed prefixes are not completed paths. Stop to analyze '
         'after this collection; no actor updates, development or sealed access.'
         if args.reuse_certificate else
        ('When checkpoint architecture is canonical_gate_kernel, this is the NEW '
         'canonical gate-state plus continuous-guide research branch. The explicit '
         '19-path budget and source override the earlier seven-route sensor-only '
         'description below. All initial state/action pairs and physical replay '
         'are verified with the actual checkpoint-specific runtime; previous '
         'sensor-only/point-guide consequences are not reused as current gradients. '
         if policy_class is not KernelLocalFeedbackPolicy else '')+
        ('NEW continuous-sensor learned parent: source, arm, paths and budget are explicit. '
         'This measures consequences under the fixed2mm mean-tangent observation, while '
         'all physical execution is inherited unchanged. It is not a repeat of old '
         'point-tangent credit, and old gradients are not reused as current evidence. '
         'This parent is an unpromoted29/29 research branch. Collect only the declared '
         'seven training routes at all their decision times; one perturbation scale '
         'previously validated in the original system, but recheck current whole-policy '
         'composition before learning. No promotion by inheritance. Earlier stage counts '
         'and parent descriptions below are historical; this explicit branch overrides them. '
         if args.continuous_screen_source else '')+
        ('Matched smaller-amplitude diagnostic: SAME parent, exact nominal decision states, '
         'executor and metrics. All decision times on explicitly listed TRAINING routes; '
         'only the single-action perturbation amplitude changes relative to the saved full '
         'collection. Compare summed credit with the already completed small full-policy '
         'composition. Do not reuse older large-amplitude slopes as infinitesimal credit. '
         'If the discrepancy persists, investigate event/resampling derivatives before '
         'using the gradients for another update. ' if args.amplitude_scale!=1 and not args.continuous_screen_source else '')+
        'Change the learning hypothesis after5/13/17 coarse parameter directions cannot '
        'provide enough simultaneous shape improvement while preserving every corner\'s '
        'energy and peak curvature. Collect CURRENT-parent consequences at ALL832 decision '
        'states of16 isolated training corners plus the previously interfered training '
        'path40. Six single20ms axis perturbations per exact state, then unchanged current '
        'actor to actual full completion or failure. Same action amplitudes as the original '
        'dense study; the parent is different, so old physical suffixes cannot be reused. '
        'Record both existing full shape metrics and a continuous command-segment distance '
        'enclosure for subsequent per-path constrained gradients in the full residual '
        'parameter space. Query count and amplitudes in the explicit budget override the '
        'original full-collection counts in this historical design description. '
        'Reconstruct saved states, never re-simulate nominal prefixes. '
        'The experiment bundles a new credit center and added physical-margin coverage; '
        'no one-factor superiority claim. Failed prefixes retain failure charges and are '
        'excluded from complete-side shape/time gradients. No labels, trajectory templates, '
        'runtime interventions, development or sealed data. No actor update until finite '
        'signal, gradient composition and feasible-parameter-space checks pass. Budget '
        'completion triggers analysis, not automatic unchanged training.'))
    started=time.perf_counter();costs=dict(actor_calls=0,new_physics_ticks=0,saved_prefix_geometry_ticks=0)
    policy=policy_class(256)
    policy.load_state_dict(torch.load(model,map_location='cpu',weights_only=False)['policy'])
    records=[];queries=[];observations=[];baselines=[]
    status_path=BASE/'当前执行状态.json';status=read(status_path)
    status['current'].update(state='current_dense_constraint_credit_running',training_running=False,
        physical_diagnostic_running=True,diagnostic_stage=stage.name)
    status['running']=stage.name;status['active_experiment']=stage.name
    status['updated_utc']=datetime.now(timezone.utc).isoformat();write_json(status_path,status)
    def save(completed_paths):
        write_json(stage/'results.json',dict(records=records,queries=queries,baselines=baselines,costs=costs,
            completed_paths=completed_paths,new_suffixes=new_count,reused_suffixes=reuse_count,goal_complete=False))
        np.savez_compressed(stage/'queries.npz',observation=np.asarray(observations))
    for path_number,path in enumerate(paths):
        trace=BASE/selected['traces'][path]
        with np.load(trace) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        assert selected['results'][path]['success']
        baselines.append(dict(path=path,trace=str(trace.relative_to(BASE)),decisions=len(actions),
            **score_complete(ref,rows,actions),segment_distance=command_segment_margin(ref,rows)))
        obs=[];states=[]
        env=reconstruct(ref,rows,actions,(len(actions)-1)*20,costs,observations=obs,state_snapshots=states,env_factory=current_factory)
        obs.append(env.observation());states.append(env.state());assert len(obs)==len(actions)==len(states)
        if args.continuous_screen_source:
            env=current_factory(ref)
            obs=[]
            for state in states:
                env.restore(state);obs.append(env.observation())
        for anchor,(observation,state) in enumerate(zip(obs,states)):
            env.restore(state);assert np.array_equal(observation,env.observation())
            latent=policy.act(observation,True)[0];costs['actor_calls']+=1
            assert np.array_equal(env.decode_action(latent),actions[anchor])
            qid=len(queries);observations.append(observation)
            queries.append(dict(query=qid,path=path,anchor=anchor))
            for axis in range(3):
                for sign in (-1,1):
                    key=(path,anchor,axis,sign)
                    if key in reuse:
                        records.append(dict(reuse[key],query=qid,reused=True,
                            exact_reuse_certificate=args.reuse_certificate))
                        reuse_count+=1
                        continue
                    env.restore(state);chunks=[];held=[];a=latent.astype(float).copy();a[axis]+=sign*amplitude[axis]
                    while True:
                        assert time.perf_counter()-started<args.wall_s
                        assert costs['actor_calls']<args.actor_budget and costs['new_physics_ticks']<args.tick_budget-2120
                        _,_,_,info=env.step(a);chunks.append(info['rows']);held.append(info['held_action'])
                        costs['new_physics_ticks']+=len(info['rows'])
                        if env.done:break
                        a=policy.act(env.observation(),True)[0];costs['actor_calls']+=1
                    suffix=np.vstack(chunks);suffix_actions=np.asarray(held)
                    full=np.vstack([rows[:anchor*20],suffix]);all_actions=np.vstack([actions[:anchor],suffix_actions])
                    audit=difference_audit(ref[0],full);success=bool(env.success and audit['kinematic_pass'] and audit['rest_pass'])
                    name=f'p{path}_a{anchor}_h{axis}_{sign:+d}.npz'
                    np.savez_compressed(stage/name,reference=ref,suffix_rows=suffix,suffix_actions=suffix_actions,
                        baseline_path=path,anchor=anchor,observation=observation,axis=axis,latent_offset=sign*amplitude[axis])
                    records.append(dict(query=qid,path=path,anchor=anchor,axis=axis,direction=sign,
                        latent_offset=float(sign*amplitude[axis]),success=success,failed_prefix=not success,
                        reason=env.reason,full_time_s=len(full)*.001 if success else None,audit=audit,
                        trace=str((stage/name).relative_to(BASE)),segment_distance=command_segment_margin(ref,full),
                        **score_complete(ref,full,all_actions,env)))
                    new_count+=1
            if anchor%15==0:
                save(path_number)
                print('CURRENT_DENSE',path,anchor,len(actions),'suffixes',len(records),'calls',costs['actor_calls'],
                    'reused',reuse_count,'new',new_count,
                    'failures',sum(not r['success'] for r in records),'wall',round(time.perf_counter()-started,1),flush=True)
        save(path_number+1)
    assert len(queries)==query_count and len(records)==6*query_count
    receipt(stage,dict(status='completed_current_dense_constraint_credit',costs=costs,
        queries=len(queries),suffixes=len(records),complete_suffixes=sum(r['success'] for r in records),
        new_suffixes=new_count,reused_suffixes=reuse_count,
        wall_s=time.perf_counter()-started))
    status=read(status_path);status['current'].update(state='current_dense_constraint_credit_complete',physical_diagnostic_running=False)
    status['completed'].append(dict(stage=stage.name,receipt_sha256=sha(stage/'receipt.json')))
    status['running']=None;status['active_experiment']=None;status['updated_utc']=datetime.now(timezone.utc).isoformat()
    status['next_action']='Audit current all-time metric gradients and direct constrained residual update feasibility before any physical candidate learning.'
    write_json(status_path,status)
    print('CURRENT_DENSE_COMPLETE',costs,flush=True)


if __name__=='__main__':main()
