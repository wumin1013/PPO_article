"""Bounded matched screen selection and scope tests; synthetic data only."""
from pathlib import Path
import sys
import copy

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT)]
import preview_target_tracking_matched as study


def node(step, successes=3, elapsed=5.):
    return dict(step=step, checkpoint=f'model{step}', checkpoint_sha256=str(step),
        rows=[dict(path_id=path, success=i < successes, elapsed_time_s=elapsed)
              for i, path in enumerate(study.audit.PATH_IDS)])


def test_step0_can_win_identically_for_every_group():
    selected = study.choose([node(0, elapsed=4.), node(256), node(512), node(1024)])
    assert selected['selected_step'] == 0 and selected['complete_total_time_s'] == 12.


def test_failure_duration_cannot_win_or_break_partial_success_ties():
    selected = study.choose([node(0, 2, 100.), node(256, 2, .01), node(512, 1, .001), node(1024, 0, .001)])
    assert selected['selected_step'] == 0 and selected['complete_total_time_s'] is None


def test_full_success_count_precedes_faster_failed_trace():
    selected = study.choose([node(0, 2, .1), node(256, elapsed=3.), node(512, elapsed=4.), node(1024)])
    assert selected['selected_step'] == 256


@pytest.mark.parametrize('step,smoke', [(2048,False),(4096,False),(32,False),(64,True),(None,False)])
def test_only_registered_missing_budget_allowed(step,smoke):
    with pytest.raises(ValueError):
        study.validate_request(step,smoke)


def test_missing_milestone_is_rejected():
    with pytest.raises(ValueError):
        study.choose([node(0),node(256),node(512)])


def test_configuration_forbids_adaptive_training_or_new_seed():
    for method,seed in [('adaptive',101),('fixed3',11)]:
        with pytest.raises(ValueError):
            study.configuration('preview_time',method,seed)


def test_common_config_removes_only_method_and_corresponding_initializer():
    cfg={'revision':{'variant':'adaptive','active_action_dims':[1,2],'init_checkpoint':'a',
        'init_checkpoint_sha256':'sha','reward':'same'},'ppo':{'epochs':4}}
    before=copy.deepcopy(cfg)
    shared=study.common_config(cfg)
    assert cfg==before and shared=={'revision':{'reward':'same'},'ppo':{'epochs':4}}
    changed=copy.deepcopy(cfg);changed['ppo']['epochs']=5
    assert study.common_config(changed)!=shared


def test_scoped_runner_and_feed_bindings_restore_after_error():
    full_out=study.full.OUT; core_env=study.core.MacroEnvironment
    original=study.audit.cnc_env.apply_kinematic_constraints
    with pytest.raises(RuntimeError,match='test failure'):
        with study.installed():
            assert study.full.OUT==study.OUT
            assert study.audit.cnc_env.apply_kinematic_constraints is study.target.apply_target_tracking_constraints
            raise RuntimeError('test failure')
    assert study.full.OUT==full_out and study.core.MacroEnvironment is core_env
    assert study.audit.cnc_env.apply_kinematic_constraints is original
