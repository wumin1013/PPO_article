from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'PPO_project'), str(ROOT / 'PPO_project/tools')]
from preview_continuation_intervention import replace_preview, compare


@pytest.mark.parametrize('distance', [None, 1.2, 3., 4.])
def test_intervention_preserves_feed_logit_and_original_action(distance):
    original = np.array([0., -3.4, .25])
    action = replace_preview(original, distance)
    np.testing.assert_array_equal(action[:2], original[:2])
    np.testing.assert_array_equal(original, [0., -3.4, .25])
    assert action[2] == pytest.approx(.25 if distance is None else (distance - .8) / 3.8)


def test_time_comparison_excludes_unfinished_time_without_losing_count():
    rows = []
    for label in ['learned', 'fixed_D1.2', 'fixed_D3', 'fixed_D4']:
        for path, success, elapsed in [('a', True, 4. if label == 'learned' else 5.),
                                        ('b', label != 'learned', .1)]:
            rows.append(dict(intervention=label, path_id=path, success=success, elapsed_time_s=elapsed))
    for result in compare(rows):
        assert result['common_success_paths'] == ['a']
        assert result['learned_successes'] == 1
        assert result['intervention_successes'] == 2
        assert result['learned_time_gain_percent'] == pytest.approx(20.)
    for row in rows:
        row['success'] = False
    assert all(r['learned_time_gain_percent'] is None for r in compare(rows))
