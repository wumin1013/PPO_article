"""Evaluate the same Bernstein curvature bound in an orthonormal chord frame.

The geometric invariants, curve/control points, peak lower samples, stopping
tolerances and iteration budget are unchanged. Local coordinates avoid loose
axis-aligned interval cross products on nearly straight diagonal transitions.
"""
import numpy as np

_INSTALLED = False


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    from baselines import legacy_s2
    module_factory = legacy_s2._module

    def factory(name, path):
        module = module_factory(name, path)
        if name == 'jmp_legacy_s2_geometry':
            native = module._peak_enclosure
            def local_enclosure(derivatives, direction):
                direction = np.asarray(direction, float)
                tangent = direction / np.linalg.norm(direction)
                rotation = np.asarray([[tangent[0], tangent[1]], [-tangent[1], tangent[0]]])
                local = [np.asarray(control) @ rotation.T for control in derivatives]
                return native(local, np.asarray([1., 0.]))
            module._peak_enclosure = local_enclosure
        return module
    legacy_s2._module = factory
    _INSTALLED = True
