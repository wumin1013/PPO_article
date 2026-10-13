"""Coverage memory, unchanged initialization, real learning and exact resume."""
import copy
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'PPO_project'), str(ROOT/'PPO_project/tools')]
import preview_coverage_frontier_study as study
from src.utils.ordered_coverage import OrderedCoverageTracker
from test_preview_full_batch import assert_tree_equal

REFERENCE = np.asarray([[0., 0.], [8., 0.], [8., 5.]])


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(observation='on', method='adaptive', seed=901):
    return study.configuration('preview_time', method, seed, observation)


def environment(cfg=None, path='unit'):
    return study.CoverageFrontierEnvironment(cfg or config(), path, reference=REFERENCE)


@pytest.mark.parametrize('projected,covered,tolerance,expected',[
    (0., 1.24, 1.25, -.0992), (2., 1., 1.25, .08), (100., 0., 1.25, 1.),
    (0., 100., 1.25, -1.), (13., 13., 1.25, 0.), (2., 1., .625, .16)])
def test_signed_feature_limits_and_tolerance_scaling(projected, covered, tolerance, expected):
    assert study.frontier_observation(projected, covered, tolerance) == pytest.approx(expected)


@pytest.mark.parametrize('values',[(0.,0.,0.),(0.,0.,-1.),(np.nan,0.,1.),(0.,np.inf,1.)])
def test_invalid_feature_is_rejected(values):
    with pytest.raises(ValueError): study.frontier_observation(*values)


def test_equal_physical_state_with_different_coverage_history_changes_only_slot21():
    first, second = environment(), environment()
    trackers = []
    for history in ([REFERENCE[1], REFERENCE[2]], [REFERENCE[2]]):
        tracker = OrderedCoverageTracker(REFERENCE, tolerance_mm=1.25,
            initial_position=REFERENCE[0], witness_spacing_mm=.01)
        for position in history: tracker.update(position)
        trackers.append(tracker)
    assert trackers[0].covered_s_mm > trackers[1].covered_s_mm
    for env, tracker in zip((first, second), trackers):
        env.inner.raw.current_position = REFERENCE[-1].copy()
        env.inner.env.env.coverage = tracker
        env._sense()
    assert first.state.shape == second.state.shape == (26,)
    assert first.state[21] != second.state[21]
    np.testing.assert_array_equal(np.delete(first.state,21), np.delete(second.state,21))


def test_initial_and_repeated_observation_read_never_updates_coverage_or_filter(monkeypatch):
    env = environment()
    tracker = env.inner.env.env.coverage
    raw = env.inner.raw
    before = copy.deepcopy(vars(tracker))
    observation = env.state.copy()
    trace = copy.deepcopy(env.trace[-1])
    reference = (raw._theta_ref_filtered, raw._theta_ref_filter_update_count,
                 raw._theta_ref_filter_update_step)
    def forbidden(*args): raise AssertionError('Observation must not advance coverage')
    monkeypatch.setattr(tracker, 'update', forbidden)
    for _ in range(3): env._sense()
    np.testing.assert_array_equal(env.state, observation)
    assert env.trace[-1] == trace
    assert reference == (raw._theta_ref_filtered,raw._theta_ref_filter_update_count,raw._theta_ref_filter_update_step)
    for key, value in before.items(): assert_tree_equal(value, getattr(tracker,key))
    assert env.state[21] < 0  # Initial matching radius is covered without reward.


def test_step_reads_frontier_after_exactly20_physical_coverage_updates(monkeypatch):
    env = environment()
    tracker = env.inner.env.env.coverage
    original = tracker.update
    seen = []
    def update(position):
        result = original(position)
        seen.append(result['covered_s_mm'])
        return result
    monkeypatch.setattr(tracker, 'update', update)
    env.step(np.asarray([0.,-2.8,.58]))
    assert len(seen) == env.physics_steps == 20
    row = env.trace[-1]
    assert row['frontier_covered_s_mm'] == seen[-1]
    assert env.state[21] == np.float32(study.frontier_observation(
        row['frontier_projected_s_mm'],seen[-1],env.inner.raw.half_epsilon))


@pytest.mark.parametrize('method',study.METHODS)
def test_same_initializer_migration_seed_and_numerical_config(method):
    on, off = config('on',method), config('off',method)
    assert on['seed'] == on['revision']['seed'] == on['experiment']['seed'] == 901
    assert on['revision']['init_checkpoint'] == off['revision']['init_checkpoint']
    for key in ('environment','reward_weights','ppo','kinematic_constraints'):
        assert on[key] == off[key] == study.BASE_CONFIG('preview_time',method,901)[key]
    for key in ('adaptive_preview','active_action_dims','fixed_distance_mm'):
        assert on['revision'][key] == off['revision'][key]
    env = type('Env',(),dict(observation_dim=26))()
    old_agent = study.full.local.make_agent(on,env)
    migrated = study.make_agent(on,env)
    paired = study.make_agent(off,env)
    states = torch.as_tensor(np.random.default_rng(71).normal(0,.1,(12,26)).astype('float32'))
    states[:,21] = 0.
    new_states = states.clone(); new_states[:,21] = torch.linspace(-1.,1.,12)
    old_agent.critic.eval(); migrated.critic.eval()
    with torch.no_grad():
        for old_value,new_value in zip(old_agent.actor(states),migrated.actor(new_states)):
            torch.testing.assert_close(old_value,new_value,rtol=0.,atol=0.)
        torch.testing.assert_close(old_agent.critic(states),migrated.critic(new_states),rtol=0.,atol=0.)
    for name in ('actor','critic'):
        assert_tree_equal(getattr(migrated,name).state_dict(),getattr(paired,name).state_dict())
    assert not migrated.actor_optimizer.state and not migrated.critic_optimizer.state
    assert all(p.requires_grad for p in migrated.actor.parameters())


def test_observation_on_off_with_migrated_agent_has_identical_initial_closed_loop():
    first, second = environment(config('on')), environment(config('off'))
    first_agent, second_agent = study.make_agent(first.cfg,first), study.make_agent(second.cfg,second)
    for _ in range(32):
        a = first_agent.take_action(first.state,deterministic=True)
        b = second_agent.take_action(second.state,deterministic=True)
        np.testing.assert_array_equal(a,b)
        s1,r1,d1,_ = first.step(a); s2,r2,d2,_ = second.step(b)
        assert (r1,d1) == (r2,d2)
        np.testing.assert_array_equal(np.delete(s1,21),np.delete(s2,21))
        assert second.state[21] == 0.
    for key in ('x_mm','y_mm','v_mm_s','a_mm_s2','heading_rad','lookahead_mm'):
        np.testing.assert_array_equal([r[key] for r in first.trace],[r[key] for r in second.trace])


def test_64_real_macro_decisions_update_new_actor_and_critic_inputs(tmp_path):
    cfg = config()
    trainer = study.BASE_TRAINER(cfg,tmp_path,environment,study.make_agent,'real-frontier')
    trainer.advance(32)
    assert trainer.agent.logged_steps == 0 and len(trainer.buffer['states']) == 32
    assert torch.count_nonzero(trainer.agent.actor.shared_fc1.weight[:,21]) == 0
    trainer.advance(64)
    assert trainer.agent.logged_steps == 64 and len(trainer.updates) == 1
    # Terminal transitions may end before20ticks; the unchanged trainer then
    # resets and fills the same on-policy64batch across completed episodes.
    assert 64 <= trainer.physics_before + trainer.env.physics_steps <= 1280
    for weight in (trainer.agent.actor.shared_fc1.weight,trainer.agent.critic.fc1.weight):
        assert float(weight.grad[:,21].norm()) > 0
        assert float(weight[:,21].detach().norm()) > 0
    assert trainer.updates[0]['active_action_dims'] == [1,2]
    assert trainer.updates[0]['old_log_probability_max_absolute_error'] < 1e-4


def test_real_environment_pending_buffer_restores_after_new_column_update(tmp_path):
    cfg = config()
    def trainer(path):
        return study.BASE_TRAINER(cfg,path,environment,study.make_agent,'frontier-resume')
    first = trainer(tmp_path/'first'); first.advance(67)
    expected_rng = (torch.get_rng_state(),np.random.get_state(),random.getstate())
    second = trainer(tmp_path/'second'); second.advance(17)
    second = trainer(tmp_path/'second'); second.advance(67)
    third = trainer(tmp_path/'second')
    for name in ('actor','critic','actor_optimizer','critic_optimizer'):
        assert_tree_equal(getattr(first.agent,name).state_dict(),getattr(third.agent,name).state_dict())
    assert_tree_equal(first.buffer,third.buffer)
    assert_tree_equal(first.updates,third.updates)
    assert_tree_equal(first.env.state,third.env.state)
    assert_tree_equal(first.env.inner.env.env.coverage.snapshot(),third.env.inner.env.env.coverage.snapshot())
    assert_tree_equal(expected_rng,(torch.get_rng_state(),np.random.get_state(),random.getstate()))


def test_scoped_install_restores_after_exception(monkeypatch,tmp_path):
    monkeypatch.setattr(study,'OUT',tmp_path/'candidate')
    full_names = ('OUT','configuration','source_state','prepare','FullBatchTrainer')
    core_names = ('OUT','configuration','source_state','prepare','MacroEnvironment','make_agent')
    previous_full = {k:getattr(study.full,k) for k in full_names}
    previous_core = {k:getattr(study.full.core,k) for k in core_names}
    with pytest.raises(RuntimeError,match='deliberate'):
        with study.installed('off'):
            assert study.full.core.MacroEnvironment is study.CoverageFrontierEnvironment
            assert study.full.core.make_agent is study.make_agent
            assert study.full.configuration('preview_time','adaptive',901)['revision']['frontier_observation'] == 'off'
            with pytest.raises(RuntimeError,match='nest'):
                with study.installed('on'): pass
            raise RuntimeError('deliberate')
    assert all(getattr(study.full,k) is v for k,v in previous_full.items())
    assert all(getattr(study.full.core,k) is v for k,v in previous_core.items())


def test_prepare_preserves_frozen_parent_and_registers_gated_budget(monkeypatch,tmp_path):
    monkeypatch.setattr(study,'OUT',tmp_path/'candidate')
    protocol = study.prepare()
    assert protocol == study.prepare()
    assert protocol['smoke_decisions'] == 32 and protocol['screening_budget'] == 1024
    assert protocol['extension_budget'] == 4096 and protocol['test_access'] is False
    for observation in study.OBSERVATIONS:
        assert study.old.read(study.OUT/observation/'protocol.json')['source'] == protocol['source']
    monkeypatch.setattr(study,'BASE_SOURCE',lambda:{'source_sha256':'changed'})
    with pytest.raises(ValueError,match='full_batch source changed'): study.prepare()


@pytest.mark.parametrize('decisions,stage',[(16,'smoke'),(0,'smoke'),(32,'development'),(2048,'development'),(4096,'development'),(1024,'formal')])
def test_budget_rejects_unregistered_or_ungated_requests(monkeypatch,tmp_path,decisions,stage):
    monkeypatch.setattr(study,'OUT',tmp_path/'candidate')
    with pytest.raises(ValueError): study.train('on','adaptive',decisions,stage)
    with pytest.raises(ValueError): study.evaluate('on','adaptive',decisions,stage)


def test_selection_prefers_complete_success_then_time_without_one_percent_rule():
    base = dict(successes=3,count=3,complete_time_s=10.,step=0)
    faster = dict(base,complete_time_s=9.999,step=256)
    failed = dict(successes=2,count=3,complete_time_s=None,step=512)
    assert study.choose([base,failed,faster]) == faster
    assert study.strictly_better(faster,base)
    assert not study.strictly_better(dict(base,complete_time_s=10.-1e-10),base)
    assert not study.strictly_better(failed,dict(failed,step=1))


def test_empty_screen_is_incomplete_and_cannot_open_gate(monkeypatch,tmp_path):
    monkeypatch.setattr(study,'OUT',tmp_path/'candidate')
    report = study.select()
    assert report['missing_chains'] and not report['screen_ready']
    assert not report['complete_grid'] and not report['extension_gate_passed']
    assert not (study.OUT/'advance_gate.json').exists()


def screen_fixture(monkeypatch,tmp_path,complete_grid=False,lock_time=11.):
    monkeypatch.setattr(study,'ROOT',tmp_path)
    monkeypatch.setattr(study,'OUT',tmp_path/'candidate')
    source = dict(source_sha256='unit-frontier-source')
    monkeypatch.setattr(study,'prepare',lambda:dict(source=source))
    monkeypatch.setattr(study,'source_state',lambda:source)
    def populate(methods,observations=study.OBSERVATIONS):
        for observation in observations:
            for method in methods:
                run = study.run_directory(observation,method)
                study.old.write(run/'training_summary.json',dict(decisions=1024))
                for step in (0,256,512,1024):
                    snapshot = run/'checkpoints'/f'step{step}'
                    snapshot.mkdir(parents=True,exist_ok=True)
                    (snapshot/'checkpoint.pt').write_bytes(f'{observation}_{method}_{step}'.encode())
                    study.old.write(snapshot/'validation/summary.json',dict(unit=True))
    populate(('adaptive','fixed3'))
    if complete_grid: populate(('fixed1p2','fixed4'),('on',))
    def validation(snapshot,output=None):
        if output is not None:
            total = 10. if Path(output).name == 'learned' else lock_time
        elif snapshot.name == 'step0': total = 20.
        elif 'adaptive' not in str(snapshot): total = 12.
        elif snapshot.parents[3].name == 'on': total = 10.
        else: total = 11.
        return [dict(success=True,elapsed_time_s=total/3,path_id=f'p{i}') for i in range(3)]
    monkeypatch.setattr(study,'checked_validation',validation)
    return populate,validation


def test_screen_can_finish_without_short_long_grid_but_gate_stays_closed(monkeypatch,tmp_path):
    populate,_ = screen_fixture(monkeypatch,tmp_path)
    screen = study.select()
    assert screen['screen_ready'] and screen['screen_eligible_for_fixed_grid']
    assert len(screen['missing_chains']) == 2 and not screen['extension_gate_passed']
    assert screen['not_scheduled_chains'] == ['off_fixed1p2','off_fixed4']
    populate(('fixed1p2','fixed4'),('on',))
    full = study.select()
    assert full['complete_grid'] and full['candidate_better']
    assert not full['d_effect_supported'] and not full['extension_gate_passed']
    assert (study.OUT/'selection_budget1024_screen.json').exists()
    assert (study.OUT/'selection_budget1024_full.json').exists()


@pytest.mark.parametrize('lock_time,passed',[(11.,True),(10.,False),(9.,False)])
def test_extension_requires_best_locked_D_comparison_and_bound_files(monkeypatch,tmp_path,lock_time,passed):
    _,validation = screen_fixture(monkeypatch,tmp_path,complete_grid=True,lock_time=lock_time)
    report = study.select()
    selected = report['groups']['on_adaptive']['selected']
    folder = study.OUT/'intervention'
    snapshot = study.ROOT/selected['checkpoint']
    groups = {label:study.performance(validation(snapshot,folder/label))
              for label in ('learned','fixed_D1.2','fixed_D3','fixed_D4')}
    evidence = dict(binding=dict(checkpoint_sha256=selected['checkpoint_sha256'],
        source_sha256='unit-frontier-source',observation='on'),groups=groups)
    study.old.write(folder/'summary.json',evidence)
    final = study.select(intervention=folder)
    assert final['extension_gate_passed'] == passed
    assert (study.OUT/'advance_gate.json').exists() == passed
    if passed:
        study._request('on','adaptive',2048,'development')
        study.old.write(folder/'summary.json',dict(evidence,tampered=True))
        with pytest.raises(ValueError,match='evidence changed'):
            study._request('on','adaptive',4096,'development')
    else:
        with pytest.raises(ValueError,match='requires passed'):
            study._request('on','adaptive',2048,'development')


def test_trainer_entry_adds_diagnostics_without_changing_snapshot_manifest(monkeypatch,tmp_path):
    cfg = config('off')
    snapshot = tmp_path/'snapshot'
    snapshot.mkdir()
    agent = study.make_agent(cfg,type('Env',(),dict(observation_dim=26))())
    torch.save(dict(actor=agent.actor.state_dict(),critic=agent.critic.state_dict(),
        source_sha256='unit-source'),snapshot/'checkpoint.pt')
    study.old.write(snapshot/'training_summary.json',dict(optimized_decisions=0,optimizer_updates=0))
    study.old.write(snapshot/'snapshot_manifest.json',dict(files={
        name:study.old.sha(snapshot/name) for name in ('checkpoint.pt','training_summary.json')}))
    before = (snapshot/'snapshot_manifest.json').read_bytes()
    monkeypatch.setattr(study,'BASE_TRAIN',lambda *args:snapshot)
    assert study.train('off','adaptive',32,'smoke') == snapshot
    diagnostics = study.old.read(snapshot/'frontier_diagnostics.json')
    assert diagnostics['actor_input_column21_l2'] == diagnostics['critic_input_column21_l2'] == 0.
    assert diagnostics['optimized_decisions'] == diagnostics['optimizer_updates'] == 0
    assert before == (snapshot/'snapshot_manifest.json').read_bytes()
