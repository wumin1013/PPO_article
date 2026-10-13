"""Complete rest-to-rest evaluation; all exceptions and failed prefixes persist."""
import time
import numpy as np
from common import sha, write_json, utc
from metrics import metrics


def rollout(policy, reference, output, *, fixed_d=None, dt=.001, wall_s=300., tolerance=1.25, collect=False, runtime_version='v2'):
    from runtime_bootstrap import setup
    setup()
    from execution_settings import configure_dt
    configure_dt(dt)
    from runtime_configuration import configure_runtime
    configure_runtime(runtime_version, tolerance=tolerance)
    from compact_farvector_runtime import FarVectorRuntime
    from direct_two_action_control import DirectTwoRuntime
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.with_suffix('.json').exists():raise FileExistsError(output)
    started=time.perf_counter(); chunks=[]; observations=[]; actions=[]; error=None; success=False; reason=None
    observed_positions=[];last_report=started;attempted_actor_calls=0;completed_decisions=0
    env=None
    initialization_wall_s=0.; actor_wall_s=0.; executor_wall_s=0.
    try:
        env=(FarVectorRuntime(reference,potential_speed=100.) if fixed_d is None else
             DirectTwoRuntime(reference,preview_rule_mm=fixed_d,potential_speed=100.))
        initialization_wall_s=time.perf_counter()-started
        # Common generous completion budget; slow valid teachers must not be
        # rejected merely by the historical length/20 deadline.
        env.env.max_ticks=int(round((5.+np.linalg.norm(np.diff(reference,axis=0),axis=1).sum()/5.)/dt))
        original_update=env.tracker.update
        def observe(position):
            observed_positions.append(np.asarray(position).copy())
            return original_update(position)
        env.tracker.update=observe
        while not env.done:
            if time.perf_counter()-started>wall_s:raise TimeoutError('Registered evaluation wall limit')
            obs=env.observation()
            attempted_actor_calls+=1
            action_started=time.perf_counter()
            if hasattr(policy,'action_for'):action=policy.action_for(env)
            else:action=policy.act(obs,True)[0]
            actor_wall_s+=time.perf_counter()-action_started
            observations.append(obs);actions.append(action)
            executor_started=time.perf_counter()
            _,_,_,info=env.step(action)
            executor_wall_s+=time.perf_counter()-executor_started
            chunks.append(info['rows']);completed_decisions+=1
            if time.perf_counter()-last_report>30:
                print('EVALUATING',output.stem,env.ticks,env.tracker.frontier,flush=True);last_report=time.perf_counter()
        success=env.success;reason=env.reason
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc);reason='exception'
        frame=exc.__traceback__
        while frame is not None:
            if frame.tb_frame.f_code.co_name=='step' and frame.tb_frame.f_code.co_filename.endswith('joint_runtime.py'):
                partial=frame.tb_frame.f_locals.get('rows',[])
                if len(partial) and sum(len(c) for c in chunks)+len(partial)<=len(observed_positions):
                    chunks.append(np.asarray(partial))
            frame=frame.tb_next
    rows=np.vstack(chunks) if chunks else np.empty((0,13))
    positions=np.vstack([reference[0],np.asarray(observed_positions).reshape(-1,2)])
    data=dict(reference=reference,rows=rows,positions=positions,observations=np.asarray(observations),actions=np.asarray(actions))
    np.savez_compressed(output,**data)
    try:
        audit=metrics(reference,positions,dt,tolerance,declared_success=success,reason=reason)
    except Exception as exc:
        audit=dict(success=False,complete_time_s=None,reason='independent_audit_exception',
                   audit_error=type(exc).__name__+': '+str(exc),samples=len(positions),elapsed_prefix_s=(len(positions)-1)*dt,dt_s=dt)
    audit.update(error=error,executor_reason=reason,wall_s=time.perf_counter()-started,trace_sha256=sha(output),
                 actor_calls=attempted_actor_calls,submitted_actions=len(actions),completed_decisions=completed_decisions,
                 actor_call_accounting='All attempted policy invocations, including the final failing decision.',
                 created_utc=utc(),fixed_preview_mm=fixed_d,runtime_version=runtime_version,
                 partial_decision_position_samples_retained=len(observed_positions)-len(rows))
    audit['guide_calls'] = getattr(env, 'v3_guide_calls', 0) if env else 0
    audit['guide_interventions'] = getattr(env, 'v3_guide_interventions', 0) if env else 0
    audit['cost'] = dict(initialization_wall_s=initialization_wall_s,actor_wall_s=actor_wall_s,
        executor_wall_s=executor_wall_s,scope='Shared-host measured wall time; excludes file/audit cost from actor and executor totals; no hard real-time claim.')
    audit['action_source'] = 'deterministic_geometric_teacher' if hasattr(policy,'action_for') else 'trained_actor_final_or_registered_checkpoint'
    write_json(output.with_suffix('.json'),audit)
    print(output.stem,audit['success'],audit['complete_time_s'],audit['reason'],flush=True)
    return (audit,data) if collect else audit
