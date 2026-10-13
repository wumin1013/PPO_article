"""Set and verify physical resolution, including repeated API calls."""
import math
from runtime_bootstrap import setup


def configure_dt(dt):
    if dt not in (.001,.0005):raise ValueError('Only registered 1 and 0.5 ms resolutions are supported')
    setup()
    import joint_runtime
    import turn_aware_projector as turn
    joint_runtime.DT=dt;joint_runtime.MACRO_TICKS=round(.02/dt)
    turn.DT=dt;turn.ORIGINAL.DT=dt
    turn.ORIGINAL.MAX_RELEASE_TERMS=math.ceil(turn.ORIGINAL.AMAX_INTERNAL/(turn.ORIGINAL.JMAX_INTERNAL*dt))
    original=getattr(turn.TurnAwareProjector,'_jmp_base_class',turn.TurnAwareProjector)
    class ResolutionProjector(original):
        _jmp_base_class=original
        def enforce_kinematic_constraints(self,v,a,j,dt_override=None):
            return super().enforce_kinematic_constraints(v,a,j,dt=dt)
    turn.TurnAwareProjector=ResolutionProjector
    assert joint_runtime.DT==dt and abs(joint_runtime.MACRO_TICKS*dt-.02)<1e-12
