"""Single common feed-target tracking change on the frozen full-batch study.

Keep original angular projection, v2 terminal brake, state, reward, network,
initializers and 64-decision PPO batches. Only scalar feed accelerations start
releasing before a reachable speed target. This module installs its wrapper
inside the current process and restores all bindings on context exit.
"""
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path

import preview_full_batch_study as full
from preview_target_tracking import project_feed_target
from src.environment import cnc_env, kinematics

BASE_OUT = full.core.old.ROOT/'论文项目/revision_20260910_continued/full_batch_v1'
OUT = full.core.old.ROOT/'论文项目/revision_20260910_continued/target_tracking_v1'
METHODS, MILESTONES = full.METHODS, full.MILESTONES
BASE_CONFIG, BASE_SOURCE, BASE_INSTALL, BASE_MAIN = (
    full.configuration, full.source_state, full.install, full.main)
ORIGINAL_APPLY = kinematics.apply_kinematic_constraints
_saved_bindings = None


def apply_target_tracking_constraints(
    prev_vel, prev_acc, prev_ang_vel, prev_ang_acc, vel_action, ang_vel_action,
    dt, MAX_VEL, MAX_ACC, MAX_JERK, MAX_ANG_VEL, MAX_ANG_ACC, MAX_ANG_JERK,
):
    """Preserve original angular channel exactly for identical arguments."""
    original = ORIGINAL_APPLY(prev_vel, prev_acc, prev_ang_vel, prev_ang_acc,
        vel_action, ang_vel_action, dt, MAX_VEL, MAX_ACC, MAX_JERK,
        MAX_ANG_VEL, MAX_ANG_ACC, MAX_ANG_JERK)
    feed = project_feed_target(prev_vel, prev_acc, vel_action,
        dt=dt, max_vel=MAX_VEL, max_acc=MAX_ACC, max_jerk=MAX_JERK)
    return (*feed, *original[3:])


def configuration(candidate, method, seed):
    cfg = BASE_CONFIG(candidate, method, seed)
    cfg['revision'].update(study='anticipatory-feed-target-tracking-v1',
        candidate_change='Only common scalar feed-target tracking: release acceleration before the current reachable speed target',
        unchanged='Original angular KCM, v2 terminal brake, local-corner state, reward, positive-feed PPO, full64 batches, initialization and optimizer settings',
        feed_tracking=dict(implementation='preview_target_tracking.project_feed_target',
            location='cnc_env.apply_kinematic_constraints feed output only; original angular function retained',
            velocity_target_unit='mm/s', sampling_period_s=.001,
            target_rule='Cap target-directed acceleration by discrete acceleration-release margin, then original hard-bound projection',
            limits='Original scalar feed velocity/acceleration/jerk bounds; no output clipping or positive feed floor',
            transient_scope='A newly unreachable target may still be crossed while inherited acceleration is released',
            not_a_learned_preview_rule=True,
            terminal_brake='Frozen v2 plan_terminal_brake, unchanged'))
    return cfg


def source_state():
    value = BASE_SOURCE()
    for name in ('preview_target_tracking.py', 'preview_target_tracking_study.py'):
        path = Path(__file__).with_name(name)
        value['source_files'][path.relative_to(full.core.old.ROOT).as_posix()] = full.core.old.sha(path)
    value['source_sha256'] = hashlib.sha256(json.dumps(value['source_files'], sort_keys=True).encode()).hexdigest()
    return value


def prepare():
    full.core.old.check_protocol()
    parent_path = BASE_OUT/'protocol.json'
    parent = full.core.old.read(parent_path)
    if parent['source'] != BASE_SOURCE():
        raise ValueError('Frozen full-batch source changed; do not rewrite its protocol')
    for method in ('fixed3', 'adaptive'):
        path = full.local.positive.PRIOR/f'preview_time_{method}_seed101/checkpoint.pt'
        if full.core.old.sha(path) != parent['initializers'][method]['checkpoint_sha256']:
            raise ValueError('Frozen sensor initializer changed')
    if full.core.old.sha(full.core.old.DATA/'manifest.json') != parent['manifest_sha256']:
        raise ValueError('Frozen development geometry changed')
    value = copy.deepcopy(parent)
    value.update(version='anticipatory-feed-target-tracking-v1', source=source_state(),
        parent_protocol_file=parent_path.relative_to(full.core.old.ROOT).as_posix(),
        parent_protocol_sha256=full.core.old.sha(parent_path),
        main_change='Only common feed-target tracking projection; no other training/control change',
        unchanged='Full-batch64 schedule, all state/reward/PPO parameters and original v2 terminal brake',
        initialization='Same corresponding sensor seed101 priors as full_batch_v1; fixed grid shares fixed3 prior',
        interpretation='Public feed tracking correction; independent preview benefit still requires adaptive versus equally corrected fixed controls',
        terminal_brake_version='Original v2; v3 is not installed')
    target = OUT/'protocol.json'
    if target.exists() and full.core.old.read(target) != value:
        raise ValueError('Preserve registered target-tracking source/protocol')
    if not target.exists():
        full.core.old.write(target, value)
    return value


def install():
    """Bind only this candidate; repeated calls are idempotent, never chained."""
    global _saved_bindings
    if _saved_bindings is not None:
        if cnc_env.apply_kinematic_constraints is not apply_target_tracking_constraints or full.OUT != OUT:
            raise RuntimeError('Candidate bindings changed during installation')
        return full
    if cnc_env.apply_kinematic_constraints is not ORIGINAL_APPLY or full.OUT != BASE_OUT:
        raise RuntimeError('Start a fresh process; another candidate is already installed')
    names = ('OUT', 'configuration', 'source_state', 'prepare')
    core_names = (*names, 'MacroEnvironment', 'make_agent')
    _saved_bindings = dict(full={name: getattr(full, name) for name in names},
        core={name: getattr(full.core, name) for name in core_names})
    full.OUT, full.configuration, full.source_state, full.prepare = OUT, configuration, source_state, prepare
    cnc_env.apply_kinematic_constraints = apply_target_tracking_constraints
    BASE_INSTALL()
    return full


def uninstall():
    global _saved_bindings
    if _saved_bindings is None:
        return
    for name, value in _saved_bindings['full'].items(): setattr(full, name, value)
    for name, value in _saved_bindings['core'].items(): setattr(full.core, name, value)
    cnc_env.apply_kinematic_constraints = ORIGINAL_APPLY
    _saved_bindings = None


@contextmanager
def installed():
    previous = _saved_bindings is not None
    install()
    try:
        yield full
    finally:
        if not previous:
            uninstall()


def main():
    # Original parser and dispatch resolve their full-module bindings, which
    # are scoped above. Training, selection, snapshot and evaluation functions
    # remain the exact frozen implementations, including their guards.
    with installed():
        BASE_MAIN()


if __name__ == '__main__': main()
