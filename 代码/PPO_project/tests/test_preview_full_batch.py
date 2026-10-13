"""Meaningful on-policy boundary, density and exact-resume regression tests."""
import copy
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'PPO_project'), str(ROOT/'PPO_project/tools')]
import preview_full_batch_study as full
from preview_positive_feed_ppo import PositiveFeedPPO
from src.utils.rl_utils import compute_advantage


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class ShortEpisodes:
    """Deterministic replayable episodes, with multiple ends in every batch."""
    def __init__(self, cfg, path_id):
        self.path_id = path_id; self.limit = 3 if path_id == 'three' else 5
        self.state = np.zeros(26, dtype=np.float32); self.state[0] = self.limit/10
        self.observation_dim = 26; self.physics_steps = self.command_steps = 0
        self.done = False; self.result = None
        self.trace = [dict(timestamp_s=0., x_mm=0., y_mm=0.)]

    def step(self, action):
        assert not self.done
        self.physics_steps += 1; self.command_steps += 1
        self.state[1] += float(action[1])*.005
        self.state[2] = self.physics_steps/10
        self.state[3] = float(action[2])
        self.done = self.physics_steps == self.limit
        reward = float(.2 + .01*action[1] + (.3 if self.done else 0.))
        self.trace.append(dict(timestamp_s=self.physics_steps*.02, x_mm=float(self.state[1]), y_mm=float(self.state[3])))
        if self.done:
            self.result = dict(success=True, termination_reason='completed', failure_reasons=[])
        return self.state.copy(), reward, self.done, {}


def config():
    return dict(seed=101, revision=dict(training_path_ids=['three', 'five'], variant='adaptive'),
                ppo=dict(epochs=2))


def agent_factory(cfg, env):
    return PositiveFeedPPO(state_dim=26, hidden_dim=16, action_dim=3,
        active_action_dims=(1, 2), epochs=2, eps=.1, ent_coef=.0003, device=torch.device('cpu'))


def trainer(path, cfg=None):
    return full.FullBatchTrainer(cfg or config(), path, ShortEpisodes, agent_factory, source_hash='unit-source')


def assert_tree_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, np.ndarray):
        np.testing.assert_array_equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first: assert_tree_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for a, b in zip(first, second): assert_tree_equal(a, b)
    else:
        assert first == second


def test_done_records_boundary_without_short_update(tmp_path):
    run = trainer(tmp_path)
    run.advance(3)
    assert run.episode == 1
    assert run.updates == [] and len(run.buffer['states']) == 3
    assert run.buffer['dones'] == [False, False, True]
    assert run.buffer['terminated'] == [False, False, True]
    run.advance(64)
    assert len(run.updates) == 1
    assert run.updates[0]['transition_count'] == 64
    assert run.updates[0]['terminated_count'] == 16
    assert run.updates[0]['old_log_probability_max_absolute_error'] < 1e-4
    assert len(run.buffer['states']) == 0


def test_gae_does_not_leak_across_episode_boundaries():
    result = compute_advantage(.9, 1., torch.tensor([[1.], [2.], [100.], [200.]]),
                               torch.tensor([[0.], [1.], [0.], [1.]]))
    torch.testing.assert_close(result, torch.tensor([[2.8], [2.], [280.], [200.]]))


def test_zero_step_and_partial_milestone_preserve_actor_and_buffer(tmp_path):
    run = trainer(tmp_path)
    before = copy.deepcopy(run.agent.actor.state_dict())
    run.advance(0)
    run.advance(31)
    assert_tree_equal(before, run.agent.actor.state_dict())
    saved = torch.load(tmp_path/'checkpoint.pt', weights_only=False)
    assert saved['optimized_decisions'] == 0 and len(saved['buffer']['states']) == 31
    assert len(saved['episodes']) > 1
    restored = trainer(tmp_path)
    assert_tree_equal(run.buffer, restored.buffer)
    assert_tree_equal(run.env.state, restored.env.state)


def test_interrupted_cross_episode_buffer_matches_uninterrupted_training(tmp_path):
    original = trainer(tmp_path/'uninterrupted')
    original.advance(128)
    expected = {name: copy.deepcopy(getattr(original.agent, name).state_dict()) for name in
                ('actor', 'critic', 'actor_optimizer', 'critic_optimizer')}
    expected_rng = (torch.get_rng_state(), np.random.get_state(), random.getstate())
    continued = trainer(tmp_path/'continued')
    continued.advance(17)  # nonempty buffer spanning four completed episodes
    continued = trainer(tmp_path/'continued')
    continued.advance(67)  # updated actor plus three pending new-policy draws
    continued = trainer(tmp_path/'continued')
    continued.advance(128)
    for name in expected: assert_tree_equal(expected[name], getattr(continued.agent, name).state_dict())
    assert_tree_equal(expected_rng, (torch.get_rng_state(), np.random.get_state(), random.getstate()))
    assert_tree_equal(original.updates, continued.updates)
    assert_tree_equal(original.episodes, continued.episodes)
    assert_tree_equal(original.env.state, continued.env.state)


def test_logs_recovered_from_authoritative_checkpoint_preserve_extra_bytes(tmp_path):
    run = trainer(tmp_path); run.advance(64)
    expected = (tmp_path/'updates.jsonl').read_bytes()
    extra = expected+b'{"interrupted_extra_record": true}\n'
    (tmp_path/'updates.jsonl').write_bytes(extra)
    (tmp_path/'episodes.jsonl').write_bytes(b'')
    restored = trainer(tmp_path)
    assert (tmp_path/'updates.jsonl').read_bytes() == expected
    assert any(p.read_bytes() == extra for p in (tmp_path/'recovery').iterdir())
    assert len((tmp_path/'episodes.jsonl').read_text(encoding='utf-8').splitlines()) == restored.episode


def test_orphan_trace_after_interrupted_terminal_commit_is_preserved(tmp_path, monkeypatch):
    run = trainer(tmp_path); run.advance(0)
    def interrupt(): raise RuntimeError('simulated crash before commit')
    monkeypatch.setattr(run, 'save', interrupt)
    with pytest.raises(RuntimeError, match='simulated crash'):
        run.advance(3)
    orphan = (tmp_path/'train_episode_0000.npz').read_bytes()
    restored = trainer(tmp_path); restored.advance(3)
    assert restored.episode == 1 and len(restored.buffer['states']) == 3
    assert any(path.read_bytes() == orphan for path in (tmp_path/'recovery').iterdir())


def test_changed_source_or_configuration_cannot_resume(tmp_path):
    run = trainer(tmp_path); run.advance(5)
    with pytest.raises(ValueError, match='source mismatch'):
        full.FullBatchTrainer(config(), tmp_path, ShortEpisodes, agent_factory, source_hash='changed')
    changed = config(); changed['seed'] = 102
    with pytest.raises(ValueError, match='config mismatch'):
        trainer(tmp_path, changed)


@pytest.mark.parametrize('method', full.METHODS)
def test_only_training_schedule_changes_and_shared_initializer_is_retained(method):
    before = full.local.configuration('preview_time', method, 101)
    after = full.configuration('preview_time', method, 101)
    for key in before:
        if key != 'revision': assert before[key] == after[key]
    for key in ('init_checkpoint', 'init_checkpoint_sha256', 'active_action_dims', 'fixed_distance_mm',
                'corner_observation', 'feed_logit_std', 'pretraining_macro_decisions'):
        assert before['revision'][key] == after['revision'][key]
    assert after['revision']['validation_milestones'] == [0, 256, 512, 1024, 2048, 4096]


def test_real_environment_replay_is_exact_with_pending_buffer(tmp_path):
    cfg = full.configuration('preview_time', 'adaptive', 901)
    reference = np.array([[0., 0.], [8., 0.], [8., 5.]])
    factory = lambda c, path: full.local.LocalCornerEnvironment(c, path, reference=reference)
    first = full.FullBatchTrainer(cfg, tmp_path, factory, agent_factory, source_hash='real-smoke')
    first.advance(9)
    expected_state = first.env.state.copy()
    second = full.FullBatchTrainer(cfg, tmp_path, factory, agent_factory, source_hash='real-smoke')
    np.testing.assert_array_equal(second.env.state, expected_state)
    assert_tree_equal(first.buffer, second.buffer)
    assert second.env.physics_steps == first.env.physics_steps


def test_snapshot_provenance_zero_step_and_self_contained_resume(tmp_path, monkeypatch):
    out = tmp_path/'study'
    monkeypatch.setattr(full, 'OUT', out)
    real_trainer = full.FullBatchTrainer
    monkeypatch.setattr(full, 'FullBatchTrainer', lambda cfg, run, source_hash:
        real_trainer(cfg, run, ShortEpisodes, agent_factory, source_hash))
    def configuration(candidate, method, seed):
        cfg = config(); cfg['seed'] = seed; cfg['revision']['variant'] = method
        return cfg
    monkeypatch.setattr(full, 'configuration', configuration)
    script = ROOT/'PPO_project/tools/preview_full_batch_study.py'
    protocol = dict(source=dict(source_sha256='unit-source', source_files={
        'PPO_project/tools/preview_full_batch_study.py': full.core.old.sha(script)}))
    full.core.old.write(out/'protocol.json', protocol)
    monkeypatch.setattr(full, 'prepare', lambda: protocol)
    initial = full.train('adaptive', 901, 0, 'smoke')
    initial_bytes = (initial/'checkpoint.pt').read_bytes()
    assert full.core.old.read(initial/'training_summary.json')['optimizer_updates'] == 0
    assert full.core.old.read(initial/'provenance.json')['protocol_sha256'] == full.core.old.sha(out/'protocol.json')
    middle = full.train('adaptive', 901, 17, 'smoke')
    final = full.train('adaptive', 901, 64, 'smoke')
    assert full.train('adaptive', 901, 0, 'smoke') == initial
    assert (initial/'checkpoint.pt').read_bytes() == initial_bytes
    metadata = full.core.old.read(middle/'snapshot_manifest.json')
    assert all(full.core.old.sha(middle/name) == digest for name, digest in metadata['files'].items())
    assert len(list(middle.glob('train_episode_*.npz'))) > 0
    # Snapshot contains all completed traces, optimizers, pending old-policy
    # transitions and current replay actions; its original mutable run is not
    # needed for recovery. Advance a copied independent snapshot directory.
    import shutil
    clone = tmp_path/'snapshot_clone'
    shutil.copytree(middle, clone)
    restored = real_trainer(configuration('preview_time', 'adaptive', 901), clone,
                           ShortEpisodes, agent_factory, 'unit-source')
    restored.advance(64)
    target = torch.load(final/'checkpoint.pt', weights_only=False)
    for key in ('actor', 'critic', 'actor_optimizer', 'critic_optimizer'):
        assert_tree_equal(getattr(restored.agent, key).state_dict(), target[key])
    (initial/'episodes.jsonl').write_text('changed\n', encoding='utf-8')
    with pytest.raises(ValueError, match='snapshot artifact changed'):
        full.train('adaptive', 901, 0, 'smoke')
