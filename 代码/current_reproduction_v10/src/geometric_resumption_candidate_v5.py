"""One versioned extension to the frozen recoverable controller's empty set.

The parent remains byte-frozen. A single extra common-executor action aims at
the next ordered gate; its complete stopping continuation is checked as usual.
"""
import inspect
import math
import textwrap
import numpy as np
import recoverable_action_candidate_v5 as parent
from physical_teacher_v5 import guide_angle

VERSION = 'geometric_resumption_candidate_v5_4'


def geometric_action(env):
    target = (env.corridor.gates[env.tracker.frontier].point if
              env.tracker.frontier < len(env.corridor.gates) else env.corridor.reference[-1])
    difference = target-env.position
    bearing = math.atan2(difference[1], difference[0])
    wrap = lambda angle: (angle+math.pi) % (2*math.pi)-math.pi
    beta = guide_angle(env.env, 0., 20., .8)
    steering = float(np.clip(wrap(bearing-beta), -math.pi/2, math.pi/2))
    feed = 0. if abs(wrap(bearing-env.heading)) > .05 else 20.
    return np.array([steering/(math.pi/2), feed/50.-1., -1.])


def extended_step():
    source = textwrap.dedent(inspect.getsource(parent.RecoverableActionCandidate._step))
    old = "            source = 'certified_stop'\n"
    new = """            restore(self.clone, state)
            geometric = evaluate(geometric_action(self.clone), 25)
            if geometric[2]['feasible']:
                selected = geometric
                source = 'geometric_resumption'
            else:
                source = 'certified_stop'
"""
    if source.count(old) != 1:
        raise ValueError('Frozen recovery branch no longer matches the registered extension')
    namespace = dict(parent.__dict__)
    namespace['geometric_action'] = geometric_action
    output = {}
    exec(compile(source.replace(old, new), __file__, 'exec'), namespace, output)
    return output['_step']


class GeometricResumptionCandidate(parent.RecoverableActionCandidate):
    _step = extended_step()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counts['geometric_resumption'] = 0

    def close(self):
        result = super().close()
        result['version'] = VERSION
        return result
