"""The gain follow-up must change only the shared guidance setting."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import preview_direct_gain as candidate
import preview_direct_research as direct


def test_all_methods_only_change_registered_shared_gain():
    for variant in direct.PILOT_VARIANTS:
        original = candidate.BASE_CONFIG(variant, 101)
        updated = candidate.config_for(variant, 101)
        authority = updated['reward_weights']['control_authority']
        assert authority['tangent_blend'] == .015
        assert authority['warmup_episodes'] == 0
        updated['reward_weights']['control_authority'] = original['reward_weights']['control_authority']
        updated['revision'].pop('shared_guidance_gain_per_s')
        updated['revision'].pop('shared_guidance_blend')
        updated['revision']['study'] = original['revision']['study']
        assert updated == original


def test_gain_is_executed_and_policy_still_controls_full_distance():
    cfg = candidate.config_for('direct', 101)
    reference = np.array([[0., 0.], [10., 0.], [10., 10.]])
    env, _ = direct.make_env(cfg, 'synthetic_unit_test', reference=reference)
    for u, expected in [(0., .8), (1., 4.6), (.5, 2.7)]:
        _, _, _, info = env.step([0., .05, u])
        assert np.isclose(env._lookahead_dist_active, expected)
        assert info['control_status']['blend_active'] == .015


def test_import_does_not_retarget_previous_study():
    assert direct.OUT.name == 'preview_direct'
