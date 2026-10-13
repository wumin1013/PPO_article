"""One registered long development run, with per-decision recovery state."""
import argparse
import json
import math
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from study_io import BASE,ROOT,register,receipt,write_json,record_progress,sha
from joint_policy import JointPolicy
from smooth_policy_candidate import SmoothExplorationPolicy
from joint_runtime_v5 import environment_factory
from train_pilot import save_torch
from joint_runtime import DT,FILTER_TAU,wrap
from validate_actions import difference_audit


def recover_first_macro(env,old_directory):
    """Recover deterministic state from the 20 saved commands, without dynamics."""
    failed=json.loads((old_directory/'failure.json').read_text(encoding='utf-8'))
    assert failed['costs']['actor_calls']==1 and failed['costs']['physics_ticks']==20
    assert "KeyError: 'batch'" in failed['traceback']
    storage=json.loads((old_directory/'storage.json').read_text(encoding='utf-8'))
    external=Path(storage['directory'])
    old_rows=np.memmap(external/'rows.bin',mode='r',dtype='float64',shape=(1600000,13))[:20].copy()
    old_action=np.memmap(external/'actions.bin',mode='r',dtype='float64',shape=(80000,3))[0].copy()
    env.held=old_action.copy()
    env.decisions=1
    exact_desired=True
    for row in old_rows:
        assert row[12]==0.  # No terminal tail in this precisely bounded recovery.
        s,_=env.progress()
        target,_=env.sample_reference(s+env.held[2])
        delta=target-env.position
        beta=math.atan2(delta[1],delta[0]) if np.linalg.norm(delta)>1e-12 else env.heading
        increment=(1-math.exp(-DT/FILTER_TAU))*wrap(beta+env.held[0]-env.heading)
        env.heading=wrap(env.heading+increment)
        expected=env.held[1]*np.array([math.cos(env.heading),math.sin(env.heading)])
        exact_desired &= np.array_equal(expected,row[8:10])
        assert np.array_equal(env.position+DT*row[2:4],row[:2])
        env.position=row[:2].copy()
        env.velocity=row[2:4].copy()
        env.acceleration=row[4:6].copy()
        env.tracker.update(env.position)
        env.ticks+=1
        rest=np.linalg.norm(env.velocity)<=.001 and np.linalg.norm(env.acceleration)<=.01
        env.rest_ticks=env.rest_ticks+1 if rest else 0
    assert exact_desired and env.tracker.inside and not env.done
    return old_rows,old_action,dict(prior_costs=failed['costs'],saved_commands_read=20,
        guidance_reconstructions=20,geometry_command_checks=20,new_physics_ticks=0,new_actor_calls=0,
        desired_vectors_and_position_updates_bitwise_equal=True,
        rng_note='Original RNG was not saved; deterministic evaluation uses no random draws. Execution state is reconstructed, not an original complete checkpoint.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',required=True)
    parser.add_argument('--model-stage',required=True)
    parser.add_argument('--case',choices=['mixed_lengths_large','dense_corners_large','closed_line_arc_large'],required=True)
    parser.add_argument('--recover-stage')
    parser.add_argument('--plan',default='long_transfer_diagnostic_v6_plan.md')
    parser.add_argument('--wall-s',type=float,default=5400.)
    args=parser.parse_args()
    if not 0 < args.wall_s <= 5400:
        raise ValueError('Bounded positive evaluation time required')
    plan=BASE/args.plan
    if plan.parent != BASE or not plan.is_file():
        raise ValueError('Use an existing local evaluation plan')
    model_dir=BASE/args.model_stage
    completed=json.loads((model_dir/'receipt.json').read_text(encoding='utf-8'))
    registration=json.loads((model_dir/'registration.json').read_text(encoding='utf-8'))
    selection=json.loads((model_dir/'selection.json').read_text(encoding='utf-8'))
    if selection['selected']['tag']!='final_validation':
        raise ValueError('This bounded evaluator requires the selected final checkpoint')
    model=model_dir/'final.pt'
    if sha(model)!=completed['final_sha256']:
        raise ValueError('Completed model bytes differ')
    prior=ROOT/'论文项目/revision_20260913_preview_efficiency_v2/conventional_ts_v1/sampling_reserve_v1/cases'/args.case/'full/evaluation.json'
    old=json.loads(prior.read_text(encoding='utf-8'))
    reference=ROOT/old['reference']['file']
    if sha(reference)!=old['reference']['sha256']:
        raise ValueError('Reference bytes differ')
    sources=[Path(__file__),model,model_dir/'receipt.json',model_dir/'registration.json',model_dir/'selection.json',reference,
        BASE/'s2_frozen_command_new_geometry_v1/receipt.json']
    sources += [BASE/f for f in ['joint_policy.py','joint_runtime.py','joint_runtime_v2.py','joint_runtime_v3.py',
        'joint_runtime_v5.py','terminal_candidate_runtime_v1.py','geometry_spec.py','validate_actions.py',
        'train_pilot.py','study_io.py','task_contract.md','smooth_policy_candidate.py']]
    sources.append(plan)
    for filename in ['joint_runtime.py','joint_runtime_v2.py','joint_runtime_v3.py',
                     'joint_runtime_v5.py','terminal_candidate_runtime_v1.py','geometry_spec.py','validate_actions.py']:
        path=BASE/filename
        assert registration['source_sha256'][path.relative_to(ROOT).as_posix()]==sha(path),filename
    preflight_file=ROOT/'论文项目/revision_20260910_continued/root_cause_iteration_v1/experiment_preflight_v1.py'
    from joint_runtime import load_source
    check=load_source('long_stage_guard',preflight_file).check_new_stage(BASE/args.stage,
        ['registration.json','receipt.json','trace.npz','evaluation.json','failure.json'],
        {p.relative_to(ROOT).as_posix():sha(p) for p in (model,reference)})
    sources.append(preflight_file)
    if args.recover_stage:
        old_dir=BASE/args.recover_stage
        if (old_dir/'receipt.json').exists():
            raise ValueError('Completed nodes cannot be recovered')
        old_registration=json.loads((old_dir/'registration.json').read_text(encoding='utf-8'))
        for source in [model,reference]:
            assert old_registration['source_sha256'][source.relative_to(ROOT).as_posix()]==sha(source)
        sources += [old_dir/'failure.json',old_dir/'registration.json',old_dir/'storage.json']
    stage=register(args.stage,sources,dict(model_loads=1,development_runs=1,actor_calls_max=80000,
        physics_ticks_max=1600000,wall_s=args.wall_s,training_decisions=0,case=args.case),
        'Measure long-reference transfer of one completed selected policy; keep all failures, no retraining or sealed test.')
    write_json(stage/'preflight.json',check)
    external=Path('D:/PPO_article_joint_runs/rolling_states')/args.stage
    external.mkdir(parents=True,exist_ok=False)
    write_json(stage/'storage.json',dict(directory=str(external),trace_dtype='float64',trace_columns=13,
        actions_columns=3,automatic_resume=False))
    costs=dict(model_loads=0,actor_calls=0,physics_ticks=0,training_decisions=0)
    started=time.perf_counter()
    try:
        torch.set_num_threads(1)
        incoming=torch.load(model,map_location='cpu',weights_only=False)
        costs['model_loads']+=1
        cls=SmoothExplorationPolicy if incoming.get('sampler')=='gsde8' else JointPolicy
        policy=cls(incoming['observation_dim'],mode=incoming['mode'])
        policy.load_state_dict(incoming['policy'])
        policy.eval()
        fixed_d=registration['budget']['fixed_d'] if incoming['mode']=='fixed' else None
        ref=np.loadtxt(reference,delimiter=',',skiprows=1)
        env=environment_factory(ref,fixed_d=fixed_d)
        if env.max_ticks>1600000:
            raise ValueError('Task time exceeds this run budget')
        rows=np.memmap(external/'rows.bin',mode='w+',dtype='float64',shape=(1600000,13))
        actions=np.memmap(external/'actions.bin',mode='w+',dtype='float64',shape=(80000,3))
        n=0
        base_actions=0
        recovered=None
        if args.recover_stage:
            prefix,first_action,recovered=recover_first_macro(env,old_dir)
            rows[:20]=prefix
            actions[0]=first_action
            n,base_actions=20,1
            rows.flush()
            actions.flush()
            write_json(stage/'prefix_reconstruction.json',recovered)
            save_torch(external/'resume_1.pt',dict(environment=env.state(),costs=costs,
                trace_rows=n,action_rows=base_actions,recovered=recovered,phase='prefix_reconstructed'))
            write_json(external/'resume_pointer.json',dict(decision=1,slot=1,trace_rows=20,
                phase='prefix_reconstructed',complete=False))
        while not env.done:
            if costs['actor_calls']>=80000 or time.perf_counter()-started>args.wall_s:
                raise TimeoutError('Registered development budget exhausted')
            latent,_,_=policy.act(env.observation(),deterministic=True)
            costs['actor_calls']+=1
            _,_,_,info=env.step(latent)
            new=n+len(info['rows'])
            if new>1600000:
                raise RuntimeError('Physics budget exhausted')
            rows[n:new]=info['rows']
            actions[base_actions+costs['actor_calls']-1]=info['held_action']
            rows.flush()
            actions.flush()
            n=new
            costs['physics_ticks']+=len(info['rows'])
            decision=base_actions+costs['actor_calls']
            save_torch(external/f'resume_{decision%2}.pt',dict(environment=env.state(),costs=costs,
                trace_rows=n,action_rows=decision,model_sha256=completed['final_sha256'],reference_sha256=old['reference']['sha256'],
                torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),recovered=recovered,phase='executing'))
            write_json(external/'resume_pointer.json',dict(decision=decision,slot=decision%2,
                trace_rows=n,phase='executing',complete=False))
            if decision%1000==0:
                value=dict(costs=costs,progress=env.progress()[0]/env.corridor.arcs[-1],elapsed_s=env.ticks*.001,
                    wall_s=time.perf_counter()-started)
                write_json(stage/'progress.json',value)
                print(value,flush=True)
        audit=difference_audit(ref[0],rows[:n])
        for key,value in incoming['policy'].items():
            assert torch.equal(value,policy.state_dict()[key]),key
        success=bool(env.success and audit['kinematic_pass'] and audit['rest_pass'])
        result=dict(case=args.case,mode=incoming['mode'],fixed_d=fixed_d,model_stage=args.model_stage,
            success=success,reason=env.reason,completion_time_s=n*.001 if success else None,
            elapsed_s=n*.001,geometry=env.tracker.snapshot(1.25),audit=audit,costs=costs,
            trace_rows=n,action_rows=base_actions+costs['actor_calls'],development_case=True,recovered=recovered)
        np.savez_compressed(stage/'trace.npz',reference=ref,rows=np.asarray(rows[:n]),actions=np.asarray(actions[:base_actions+costs['actor_calls']]))
        write_json(stage/'evaluation.json',result)
        receipt(stage,dict(status='long_development_evaluation_completed',success=success,costs=costs,
            evaluation_sha256=sha(stage/'evaluation.json'),trace_sha256=sha(stage/'trace.npz'),
            wall_s=time.perf_counter()-started,goal_complete=False))
        write_json(external/'resume_pointer.json',dict(decision=base_actions+costs['actor_calls'],trace_rows=n,
            phase='complete',complete=True,automatic_resume_forbidden=True))
        record_progress(args.stage,[stage/'receipt.json'],
            '一项新所选模型长路径开发评价已完成，成败以对应evaluation为准，S2与旧v6结果直接复用；按已注册计划核配对另一组并分析新保存结果，不重复执行。20页英文工作稿仍缺最终图件、独立种子、三类最终长路径与封存测试，总目标未完成。')
        print(result,flush=True)
    except Exception:
        write_json(stage/'failure.json',dict(traceback=traceback.format_exc(),costs=costs,
            wall_s=time.perf_counter()-started,automatic_retry=False))
        raise


if __name__=='__main__':
    main()
