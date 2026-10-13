"""Finite launch response with independently verified terminal candidate selection."""
from pathlib import Path
import sys
import launch_tracking

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'terminal_research'))
import terminal_candidate
LAST_TERMINAL=None


def install(*,omega,dt=.001,tolerance=1.25):
    global LAST_TERMINAL
    LAST_TERMINAL=None
    settings=launch_tracking.install(omega=omega,dt=dt,tolerance=tolerance)
    from common import ROOT as BASE
    binding=terminal_candidate.install(vendor_dir=BASE/'literature/ruckig_vendor')
    import joint_runtime as joint
    previous=joint.JointRuntime._consider_terminal
    def considered(env):
        global LAST_TERMINAL
        previous(env)
        if hasattr(env,'v9_terminal_diagnostics'):
            LAST_TERMINAL=env.v9_terminal_diagnostics
    joint.JointRuntime._consider_terminal=considered
    settings.update(version='v9_launch_and_verified_terminal',terminal_binding=binding,
        terminal_brake_unchanged=False,one_ms_commands_unchanged_from_v4e=False,
        numerical_change_only_from_v4d_at_1ms=False)
    return settings
