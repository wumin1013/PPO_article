"""Shared normalization for matched shape/energy objectives and selection."""
import numpy as np


def normalized_regrets(shape_values, energy_value, ideals):
    assert all(0. < ideals[name] < 1. for name in ('shape', 'energy'))
    return np.r_[(np.asarray(shape_values) - ideals['shape']) / (1. - ideals['shape']),
                 (energy_value - ideals['energy']) / (1. - ideals['energy'])]
