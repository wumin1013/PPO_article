"""A finite initial tangential ramp; original normal and subsequent tracking."""
import ast
from pathlib import Path
import textwrap
import numpy as np

LAUNCH_S = .12


def install(*,omega,dt=.001,tolerance=1.25):
    from runtime_physical_time_v5 import install as install_base
    settings=install_base(dt=dt,tolerance=tolerance)
    import joint_runtime as joint
    if omega not in (40.,60.,80.):raise ValueError('Unregistered launch bandwidth')
    if omega!=40.:
        def demand(self,velocity,acceleration,desired,rate):
            velocity,acceleration,jerk,diagnostics=joint.tracking.compute_tracking_demand(
                velocity,acceleration,desired,rate)
            t=self.ticks*joint.DT
            target_speed=float(np.linalg.norm(desired))
            if t<LAUNCH_S and target_speed>1e-9:
                direction=desired/target_speed
                extra=(omega**2-1600.)*float(np.dot(desired-velocity,direction))
                extra-=(2.*omega-80.)*float(np.dot(acceleration,direction))
                # Smoothly recover the unchanged demand before normal traversal.
                weight=(1.-(t/LAUNCH_S)**2)**2
                jerk=jerk+weight*extra*direction
            return velocity,acceleration,jerk,diagnostics
        joint.JointRuntime._launch_tracking_demand=demand
        # Read the actual immutable class source rather than inspect() on a
        # compiled v5 patch, whose line numbers intentionally start at one.
        source=Path(joint.__file__).read_text('utf-8-sig')
        tree=ast.parse(source)
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='JointRuntime')
        node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='step')
        step=textwrap.dedent(ast.get_source_segment(source,node))
        changes={'range(MACRO_TICKS+2100)':'range(MACRO_TICKS+round(2.1/DT))',
                 'tracking.compute_tracking_demand(old_v,old_a,desired,rate)':
                 'self._launch_tracking_demand(old_v,old_a,desired,rate)'}
        for old,new in changes.items():
            assert step.count(old)==1,old
            step=step.replace(old,new)
        namespace={}
        exec(compile(step,joint.__file__,'exec'),joint.JointRuntime.step.__globals__,namespace)
        namespace['step']._physical_time_v5=True
        joint.JointRuntime.step=namespace['step']
    settings.update(version='v9_finite_launch',launch_omega_s_inv=omega,
        launch_window_s=LAUNCH_S,launch_blend='(1-(t/0.12)^2)^2 for 0<=t<0.12',
        subsequent_tracking_omega_s_inv=40.,normal_tracking_unchanged=True,
        kcm_unchanged=True,terminal_brake_unchanged=True,shared_by=['JNN','FIXED'])
    return settings
