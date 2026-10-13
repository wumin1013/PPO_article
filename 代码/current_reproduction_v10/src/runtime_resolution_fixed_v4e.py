"""Hold geometric and control parameters fixed while changing integration dt."""
VERSION = 'v4e_fixed_physical_guide_width'
GUIDE_TRANSITION_WIDTH_MM = .1


def install(*, dt=.001, tolerance=1.25):
    from terminal_completion_v4d import install as install_v4d
    result = install_v4d(dt=dt, tolerance=tolerance)
    import continuous_gate_guidance as guide
    # The historical module sets width = 100 * DT during its first import.
    # Pin its physical nominal value so import order and dt cannot change it.
    guide.GATE_TRANSITION_WIDTH_MM = GUIDE_TRANSITION_WIDTH_MM
    result.update(version=VERSION, dt_s=dt,
                  guide_transition_width_mm=GUIDE_TRANSITION_WIDTH_MM,
                  control_period_s=.02, heading_filter_tau_s=.012,
                  numerical_change_only_from_v4d_at_1ms=True)
    return result
