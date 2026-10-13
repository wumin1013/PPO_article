"""Three fixed geometric recovery requests over the frozen v5_4 controller."""
import inspect
import textwrap
import types
import geometric_resumption_candidate_v5 as parent

VERSION = 'geometric_efficiency_candidates_v5_5'
CANDIDATES = [
    dict(id='feed40_heading005', feed_mm_s=40., heading_tolerance_rad=.05),
    dict(id='feed20_heading015', feed_mm_s=20., heading_tolerance_rad=.15),
    dict(id='feed40_heading015', feed_mm_s=40., heading_tolerance_rad=.15),
]


def geometric_request(spec):
    source = textwrap.dedent(inspect.getsource(parent.geometric_action))
    replacements = [
        ('guide_angle(env.env, 0., 20., .8)', 'guide_angle(env.env, 0., RECOVERY_FEED, .8)'),
        ('feed = 0. if abs(wrap(bearing-env.heading)) > .05 else 20.',
         'feed = 0. if abs(wrap(bearing-env.heading)) > HEADING_THRESHOLD else RECOVERY_FEED'),
    ]
    for old, new in replacements:
        assert source.count(old) == 1
        source = source.replace(old, new)
    namespace = dict(parent.__dict__)
    namespace.update(RECOVERY_FEED=spec['feed_mm_s'], HEADING_THRESHOLD=spec['heading_tolerance_rad'])
    output = {}
    exec(compile(source, __file__, 'exec'), namespace, output)
    return output['geometric_action']


def controller_class(spec):
    assert spec in CANDIDATES
    original = parent.GeometricResumptionCandidate._step
    namespace = dict(original.__globals__)
    namespace['geometric_action'] = geometric_request(spec)
    step = types.FunctionType(original.__code__, namespace, original.__name__,
                              original.__defaults__, original.__closure__)

    class EfficiencyCandidate(parent.GeometricResumptionCandidate):
        _step = step

        def close(self):
            result = super().close()
            result.update(version=VERSION, candidate=spec.copy())
            return result

    return EfficiencyCandidate
