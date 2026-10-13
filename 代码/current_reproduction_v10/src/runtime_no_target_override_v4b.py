"""Install the explicitly versioned v4b executor without mutating source files."""
VERSION='v4b_no_target_override'


def install(*,dt=.001,tolerance=1.25):
    """Keep source witness/local blending; return the unmodified guide target.

    Call before environment construction. The substitution persists when the
    existing evaluate.rollout calls configure_runtime('v3') again. Preview is
    still the third learned action, or rollout(fixed_d=...) for a fixed-preview
    teacher; steering and feed retain the common action decoder.
    """
    from runtime_bootstrap import setup
    setup()
    from execution_settings import configure_dt
    configure_dt(dt)
    import runtime_configuration as runtime
    from replay_no_target_override_v4b import no_target_override
    runtime._ordered_target=no_target_override
    runtime.configure_runtime('v3',tolerance=tolerance)
    return dict(version=VERSION,dt_s=dt,tolerance_mm=tolerance,
        source_witness_initialization=True,local_disjoint_blending=True,target_override=False,
        fixed_preview_interface='evaluate.rollout(..., fixed_d=preview_mm, runtime_version="v3")',
        adaptive_preview_interface='policy raw normalized action[2]; common decoder maps to 0.8--12 mm',
        heading_filter_tau_s=.012)
