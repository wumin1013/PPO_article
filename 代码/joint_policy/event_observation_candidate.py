"""Compact local event geometry for learning, with no execution-side rules."""
import math
import numpy as np
from compact_observation_candidate import compact_observation
from joint_runtime_v5 import JointRuntimeV5


def event_observation(env, include_far_point=False):
    # Reuse the audited motion, anchor, task, and contact encoding.
    common = compact_observation(env)
    if env.tracker.frontier < len(env.corridor.gates):
        index = env.corridor.gates[env.tracker.frontier].vertex_index
        length = env.corridor.lengths[index]
        if index+1 < len(env.corridor.tangents):
            a, b = env.corridor.tangents[index:index+2]
            turn = math.atan2(a[0]*b[1]-a[1]*b[0], a@b)
        else:
            turn = 0.
    else:
        length, turn = 0., 0.
    geometry = np.array([math.log1p(length/15.)/4., turn/math.pi], dtype=np.float32)
    parts = [common[:9], geometry]
    if include_far_point:
        parts.append(common[11:13])  # Actual 15-mm reference query, no fitted curve.
    return np.concatenate(parts+[common[13:]])


class LearningDesignRuntime(JointRuntimeV5):
    def __init__(self, *args, representation='original72', reward_mode='original', **kwargs):
        if representation not in ('original72', 'events16', 'events_far18'):
            raise ValueError(representation)
        if reward_mode not in ('original', 'unshaped'):
            raise ValueError(reward_mode)
        self.representation, self.reward_mode = representation, reward_mode
        super().__init__(*args, **kwargs)

    def observation(self):
        if self.representation == 'original72':
            return super().observation()
        return event_observation(self, self.representation == 'events_far18')

    def potential(self):
        return 0. if self.reward_mode == 'unshaped' else super().potential()


def factory(reference, *, fixed_d=None, max_time=None, representation='original72', reward_mode='original'):
    length = np.linalg.norm(np.diff(reference, axis=0), axis=1).sum()
    return LearningDesignRuntime(reference, fixed_d=fixed_d, max_time=max(5.,float(length)/20.+1.),
        closed=bool(np.linalg.norm(reference[0]-reference[-1])<1e-9),
        representation=representation, reward_mode=reward_mode)
