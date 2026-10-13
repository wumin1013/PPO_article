"""Integration check: candidate evaluation cannot advance the real tracker."""
import copy
import json
import numpy as np
from common import ROOT, sha, utc, write_json
from terminal_completion_v4d import install, consider_verified_terminal
from replay_strong_checkpoint_v4 import paths
from strong_policy_v4 import load_strong_policy


def equal(a, b):
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def main():
    install()
    import joint_runtime
    from compact_farvector_runtime import FarVectorRuntime
    actor, _ = load_strong_policy(ROOT / 'results/strong_fit_v4/pilot_seed101/adaptive/final.pt')
    results = []
    for row in paths():
        if row['historical_split'] not in ('train', 'validation'):
            continue
        env = FarVectorRuntime(np.loadtxt(ROOT / row['reference'], delimiter=',', skiprows=1), potential_speed=100.)
        calls = []
        original_update = env.tracker.update
        def observed(point):
            calls.append(np.asarray(point).copy())
            return original_update(point)
        env.tracker.update = observed
        checks = []
        def checked_consider(core):
            before = copy.deepcopy({k: v for k, v in vars(core.tracker).items() if k not in ('corridor', 'update')})
            n = len(calls)
            consider_verified_terminal(core)
            after = {k: v for k, v in vars(core.tracker).items() if k not in ('corridor', 'update')}
            assert n == len(calls), 'Candidate trace entered actual command recorder'
            assert equal(before, after), 'Candidate trace advanced actual tracker'
            checks.append(dict(tick=core.ticks, brake_accepted=core.brake is not None,
                               remaining_gates=len(core.corridor.gates)-core.tracker.frontier))
        joint_runtime.JointRuntime._consider_terminal = checked_consider
        while not env.done:
            env.step(actor.act(env.observation(), True)[0])
        assert len(calls) == env.ticks
        assert env.success
        results.append(dict(path=row['id'], checks=len(checks), accepted=[c for c in checks if c['brake_accepted']],
                            recorder_samples=len(calls), physics_ticks=env.ticks, success=env.success))
    write_json(ROOT / 'audit/terminal_isolation_v4d.json', dict(created_utc=utc(),
        module_sha256=sha(ROOT / 'src/terminal_completion_v4d.py'),
        check_sha256=sha(__file__), results=results, all_passed=True,
        original_tracker_unchanged_by_every_candidate=True, speculative_samples_never_recorded=True))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    import torch
    torch.set_num_threads(1)
    main()
