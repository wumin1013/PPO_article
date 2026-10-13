"""Training-only handling of known physical rejection, without relaxing physics."""
import numpy as np

KNOWN_REJECTIONS={
    'Inductive backup failed numerical certification',
    'Input lacks both recursive certificates',
    'Turn-release numerical transition failed unchanged internal limits',
}


def sampled_step(env, action, dt, record_failure):
    """Keep accepted partial XY/V samples when the executor rejects a decision.

    A rejected transition is terminal with no value bootstrap. The caller gives
    it the same failure cost as any other physical failure. Unexpected errors
    still propagate. Full evaluation continues to use the unwrapped executor.
    """
    tracker=env.tracker;update=tracker.update;rows=[];start_tick=env.ticks
    had_callback='update' in vars(tracker)
    def witnessed(position):
        result=update(position)
        rows.append(np.r_[position,env.velocity].copy())
        return result
    tracker.update=witnessed
    rejected=None
    try:
        result=env.step(action)
    except ValueError as exc:
        if str(exc) not in KNOWN_REJECTIONS:raise
        rejected=str(exc)
    finally:
        # Remove our temporary instance callback before serializing state.
        if had_callback:tracker.update=update
        else:del tracker.update
    accepted=np.array(rows).reshape(-1,4)
    assert len(accepted)==env.ticks-start_tick
    if rejected is None:
        assert np.array_equal(accepted,result[3]['rows'][:,:4])
        return result
    env.env.done=True;env.env.success=False;env.env.reason='physical_certification_rejection'
    record_failure(dict(error=rejected,actual_accepted_ticks=len(accepted),start_tick=start_tick,
        failed_tick=env.ticks,action=np.asarray(action).copy(),accepted_xyv=accepted,
        rejected_state=env.state(),dt_s=dt))
    info=dict(rows=accepted,elapsed_s=len(accepted)*dt,success=False,reason=env.reason,
        held_action=env.held.copy(),gates=env.tracker.frontier,terminal_tail=False,
        physical_rejection=rejected)
    return env.observation(),0.,True,info
