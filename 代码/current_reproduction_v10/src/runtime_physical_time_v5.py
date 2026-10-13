"""Versioned physical-time corrections on the v4e executor.

The archived runtime files stay byte-identical. Only two audited expressions
are replaced in memory; guide, projector and completion rules are retained.
"""
import inspect
import textwrap

VERSION = 'v5_physical_time'


def _replace_function(function, old, new):
    source = textwrap.dedent(inspect.getsource(function))
    if source.count(old) != 1:
        raise RuntimeError('Runtime patch source does not match the audited expression')
    namespace = {}
    exec(compile(source.replace(old, new), inspect.getsourcefile(function), 'exec'),
         function.__globals__, namespace)
    return namespace[function.__name__]


def install(*, dt=.001, tolerance=1.25):
    from runtime_resolution_fixed_v4e import install as install_base
    result = install_base(dt=dt, tolerance=tolerance)
    import joint_runtime as joint
    import direct_compact_runtime as direct
    if not getattr(joint.JointRuntime.step, '_physical_time_v5', False):
        step = _replace_function(joint.JointRuntime.step,
            'range(MACRO_TICKS+2100)', 'range(MACRO_TICKS+round(2.1/DT))')
        step._physical_time_v5 = True
        joint.JointRuntime.step = step
    if not getattr(direct.DirectCompactRuntime.step, '_physical_time_v5', False):
        direct.joint_runtime_v5 = joint
        step = _replace_function(direct.DirectCompactRuntime.step,
            'self.env.max_ticks*.001', 'self.env.max_ticks*joint_runtime_v5.DT')
        step._physical_time_v5 = True
        direct.DirectCompactRuntime.step = step
    result.update(version=VERSION, terminal_iteration_allowance_s=2.1,
                  failure_reward_budget='max_ticks * current integration dt',
                  one_ms_commands_unchanged_from_v4e=True)
    return result
