import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from preview_direct_swap import ReplacedPreview
import preview_direct_research as direct


def test_replacement_preserves_motion_and_changes_executed_preview():
    cfg = direct.config_for('direct', 101)
    env, initial = direct.make_env(cfg, 'unit_reference', reference=np.array([[0.,0.],[10.,0.],[10.,10.]]))
    proxy = ReplacedPreview(env, 2.3)
    np.testing.assert_array_equal(proxy.state, initial)
    action = np.array([0., .05, 1.])
    next_obs, _, _, info = proxy.step(action)
    np.testing.assert_array_equal(action, [0.,.05,1.])
    assert next_obs.shape == (20,)
    assert info['control_status']['direct_selected_mm'] == 2.3


def test_replacement_rejects_fixed_policy_and_out_of_bounds():
    cfg = direct.config_for('fixed_mid', 101)
    env, _ = direct.make_env(cfg, 'unit_reference', reference=np.array([[0.,0.],[10.,0.]]))
    with pytest.raises(ValueError):
        ReplacedPreview(env, 2.3)
    env.action_space_dim = 3
    with pytest.raises(ValueError):
        ReplacedPreview(env, 5.)
