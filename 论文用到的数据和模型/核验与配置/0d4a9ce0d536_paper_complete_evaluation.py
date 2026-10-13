"""Complete-path evaluation shared by finite paper studies; failures stay failures."""
import time
import numpy as np
from complete_motion_metrics import complete_motion_metrics
from validate_actions import difference_audit
from study_io import sha, write_json


def evaluate(policy, env, file, deadline, *, max_calls=100000, prefix=None, bind_policy=True):
    reference=env.corridor.reference
    if bind_policy and hasattr(policy,'bind'): policy.bind(env)
    chunks=[]; observations=[]; actions=[]; latents=[]; executed=[]
    if prefix is not None:
        # Only an exactly replayed, whole-decision prefix may be continued.
        assert not bind_policy, 'Binding would erase the replayed causal memory'
        assert len(prefix['rows']) == env.ticks == 20*env.decisions
        assert len(prefix['actions']) == env.decisions
        assert np.array_equal(prefix['reference'],reference)
        assert np.array_equal(prefix['rows'][-1,:2],env.position)
        chunks=[prefix['rows']];observations=list(prefix['observations'])
        actions=list(prefix['actions']);latents=list(prefix['latents'])
        executed=[prefix['executed_actions_1ms']]
    error=None; started=time.perf_counter(); last=started
    try:
        while not env.done:
            if time.perf_counter()>=deadline or len(actions)>=max_calls:
                raise TimeoutError('Registered complete-evaluation limit')
            x=env.observation(); latent=policy.act(x,True)[0]
            if hasattr(env,'pending_pulse'): assert not np.any(env.pending_pulse)
            _,_,_,info=env.step(latent)
            rows=info['rows']; chunks.append(rows); observations.append(x)
            actions.append(info['held_action']); latents.append(latent)
            executed.append(info.get('executed_actions_1ms',np.tile(info['held_action'],(len(rows),1))))
            if time.perf_counter()-last>40:
                last=time.perf_counter()
                print('EVALUATING',file.stem,'ticks',env.ticks,'gates',env.tracker.frontier,flush=True)
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
    rows=np.vstack(chunks) if chunks else np.empty((0,13))
    held=np.vstack(executed) if executed else np.empty((0,3))
    physical=difference_audit(reference[0],rows) if len(rows) else None
    geometry=env.tracker.snapshot(1.25)
    success=bool(error is None and env.success and geometry['geometry_complete'] and physical['kinematic_pass'] and physical['rest_pass'])
    change=np.flatnonzero(np.any(held[1:]!=held[:-1],axis=1))+1
    change=change[(rows[change,12]==0)&(rows[change-1,12]==0)] if len(change) else change
    assert np.all(change%20==0)
    np.savez_compressed(file,reference=reference,rows=rows,observations=np.asarray(observations),
        actions=np.asarray(actions),latents=np.asarray(latents),executed_actions_1ms=held)
    result=dict(success=success,reason=env.reason,error=error,
        complete_time_s=env.ticks*.001 if success else None,
        elapsed_prefix_or_complete_s=env.ticks*.001,geometry=geometry,audit=physical,
        ordinary_20ms_hold=True,motion=complete_motion_metrics(reference,rows) if len(rows) else None,
        motion_scope='complete' if success else 'failure_prefix_only',trace_sha256=sha(file),
        actor_calls=len(actions),physics_ticks=len(rows),wall_s=time.perf_counter()-started)
    write_json(file.with_suffix('.json'),result)
    return result
