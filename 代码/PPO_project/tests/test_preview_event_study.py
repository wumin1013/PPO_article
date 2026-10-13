"""Independent environment, resume and artifact-gate checks for event preview."""
import importlib
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "PPO_project/tools"))


@pytest.fixture
def event_study(monkeypatch):
    tool = importlib.import_module("preview_event_study")
    paths = [{"id": "train_probe", "split": "train"}] + [
        {"id": f"validation_{i}", "split": "validation"} for i in range(3)] + [{"id": "test_unopened", "split": "test"}]
    monkeypatch.setattr(tool, "manifest", lambda: {"paths": paths})
    monkeypatch.setattr(tool, "points", lambda _: np.array([[0., 0.], [50., 0.]]))
    torch.set_num_threads(1)
    return tool


def env_for(tool, variant="hold50"):
    cfg = tool.config_for(variant, 101)
    return cfg, tool.EventPreviewEnvironment(cfg, "train_probe", reference=np.array([[0., 0.], [50., 0.]]))


def test_hold50_changes_distance_only_on_ticks_zero_and_fifty(event_study):
    _, env = env_for(event_study)
    events = []
    for tick in range(52):
        requested = .2 if tick == 0 else .9 if tick == 50 else .55
        _, _, _, info = env.step([0., .05, requested])
        if info["control_status"]["preview_event"]:
            events.append(tick)
        expected = .2 if tick < 50 else .9
        assert env.held_preview == pytest.approx(expected)
        assert env.raw._lookahead_dist_active == pytest.approx(.8+3.8*expected)
    assert events == [0, 50]


def test_twenty_five_action_replay_restores_holding_phase(event_study):
    cfg, original = env_for(event_study)
    actions = np.random.default_rng(25).uniform([-.1,.03,0.], [.1,.07,1.], (65,3))
    for action in actions[:25]:
        original.step(action)
    restored = event_study.EventPreviewEnvironment(cfg, "train_probe", reference=np.array([[0.,0.],[50.,0.]]))
    for action in actions[:25]:
        restored.step(action)
    assert original.tick == restored.tick == 25
    assert original.state[-5] == pytest.approx(.5)
    assert not restored.preview_decision
    np.testing.assert_array_equal(original.state, restored.state)
    for action in actions[25:]:
        np.testing.assert_array_equal(original.step(action)[0], restored.step(action)[0])


def test_fixed_preview_never_scores_or_executes_third_action(event_study):
    cfg, env = env_for(event_study, "fixed2p3")
    assert cfg["revision"]["active_action_dims"] == [0,1]
    agent = event_study.make_agent(cfg, env)
    for requested in (0.,1.,.3):
        action, metadata = agent.take_action(env.state, preview_decision=True, held_preview=env.held_preview, return_info=True)
        np.testing.assert_array_equal(metadata["action_mask"], [1.,1.,0.])
        action[2] = requested
        _, _, _, info = env.step(action)
        assert not info["control_status"]["preview_event"]
        assert env.raw._lookahead_dist_active == pytest.approx(2.3)
        assert env.held_preview == pytest.approx((2.3-.8)/3.8)


def test_twenty_six_state_suffix_has_registered_units_and_matching_space(event_study):
    _, env = env_for(event_study)
    env.tick = 25
    env.held_preview = .25
    env.raw.heading = -.4
    env.raw._theta_ref_filtered = .2
    env.raw.angular_acc = .25 * env.raw.MAX_ANG_ACC
    obs = env._augment(env.env.state)
    np.testing.assert_allclose(obs[-6:], [.25,.5,.6/math.pi,.25,5.,.4/math.pi], rtol=1e-6, atol=1e-7)
    assert obs.shape == (26,)
    assert env.observation_space.shape == (26,)
    assert env.observation_space.contains(obs)


def test_actual_smoke_checkpoint_resume_preserves_phase(event_study, monkeypatch, tmp_path):
    tool = event_study
    cfg = tool.config_for("hold50", 907)
    cfg["ppo"].update(hidden_dim=16, epochs=1)
    monkeypatch.setattr(tool, "config_for", lambda variant, seed: cfg)
    monkeypatch.setattr(tool, "OUT", tmp_path / "study")
    monkeypatch.setattr(tool, "DATA", tmp_path / "data")
    tool.DATA.mkdir()
    tool.write(tool.DATA / "manifest.json", tool.manifest())
    source = {"source_files": {"PPO_project/src/environment/cnc_env.py": tool.sha(ROOT / "PPO_project/src/environment/cnc_env.py")}, "source_sha256": "fixture_source"}
    monkeypatch.setattr(tool, "source_state", lambda: source)
    run = tool.train("hold50", 907, "smoke", 25)
    saved = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert saved["steps"] == 25
    assert len(saved["episode_actions"]) == 25
    assert saved["observation"][-5] == pytest.approx(.5)
    tool.train("hold50", 907, "smoke", 50)
    restored = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert restored["steps"] == 50
    assert len(restored["episode_actions"]) == 50
    assert restored["observation"][-5] == pytest.approx(0.)


def fake_run(tool, monkeypatch, tmp_path, variant="hold50", seed=101, stage="pilot"):
    monkeypatch.setattr(tool, "OUT", tmp_path / "study")
    monkeypatch.setattr(tool, "DATA", tmp_path / "data")
    tool.DATA.mkdir()
    tool.write(tool.DATA / "manifest.json", tool.manifest())
    monkeypatch.setattr(tool, "source_state", lambda: {"source_sha256": "fixture_source"})
    monkeypatch.setattr(tool, "check_protocol", lambda: None)
    cfg = tool.config_for(variant, seed)
    run = tool.OUT / stage / f"{variant}_seed{seed}"
    run.mkdir(parents=True)
    tool.write(tool.OUT / "protocol.json", {"role": "test fixture"})
    tool.write(tool.OUT / "selection.json", {"selected_fixed": "fixed2p3", "selected_hold": "hold50"})
    (run / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    torch.save({"steps": tool.PILOT_STEPS if stage == "pilot" else tool.FORMAL_STEPS}, run / "checkpoint.pt")
    tool.write(run / "provenance.json", {"source_sha256": "fixture_source", "config_sha256": tool.sha(run / "config.yaml"), "manifest_sha256": tool.sha(tool.DATA / "manifest.json")})
    tool.write(run / "training_summary.json", {"steps": tool.PILOT_STEPS if stage == "pilot" else tool.FORMAL_STEPS,
        "checkpoint_sha256": tool.sha(run / "checkpoint.pt")})
    return run


def validation_rows(tool, run):
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
    return [dict(path_id=f"validation_{i}", split="validation", regime="nominal", success=True,
        variant=cfg["revision"]["variant"], effective_variant=cfg["revision"]["effective_variant"], seed=cfg["seed"],
        training_steps=checkpoint["steps"],
        path_completed=True, ordered_progress=1., elapsed_time_s=1.,
        checkpoint_sha256=tool.sha(run / "checkpoint.pt"), config_sha256=tool.sha(run / "config.yaml"),
        source_sha256="fixture_source", manifest_sha256=tool.sha(tool.DATA / "manifest.json")) for i in range(3)]


def test_test_split_rejected_before_any_path_access_without_freeze(event_study, monkeypatch, tmp_path):
    tool = event_study
    run = fake_run(tool, monkeypatch, tmp_path)
    def forbidden(_):
        pytest.fail("Test geometry was accessed without frozen artifacts")
    monkeypatch.setattr(tool, "points", forbidden)
    with pytest.raises((FileNotFoundError, ValueError)):
        tool.evaluate(run, "test")
    assert not (run / "evaluation_test_nominal").exists()


def test_rank_rejects_stale_validation_model_metadata(event_study, monkeypatch, tmp_path):
    tool = event_study
    run = fake_run(tool, monkeypatch, tmp_path)
    rows = validation_rows(tool, run)
    rows[0]["checkpoint_sha256"] = "stale_model"
    tool.write(run / "evaluation_validation_nominal/summary.json", {"rows": rows})
    with pytest.raises(ValueError, match="(?i)validation|metadata|checkpoint|model|hash|stale|misbound"):
        tool.rank("hold50")


def test_freeze_rejects_wrong_validation_path_set(event_study, monkeypatch, tmp_path):
    tool = event_study
    monkeypatch.setattr(tool, "FORMAL", ("state_direct",))
    monkeypatch.setattr(tool, "SEEDS", (11,))
    run = fake_run(tool, monkeypatch, tmp_path, "state_direct", 11, "formal")
    rows = validation_rows(tool, run)
    rows[0]["path_id"] = "test_unopened"
    tool.write(run / "evaluation_validation_nominal/summary.json", {"rows": rows})
    with pytest.raises(ValueError, match="(?i)validation|path|metadata"):
        tool.freeze()
    assert not (tool.OUT / "freeze.json").exists()


def test_cached_validation_rejects_changed_source_metadata(event_study, monkeypatch, tmp_path):
    tool = event_study
    run = fake_run(tool, monkeypatch, tmp_path)
    rows = validation_rows(tool, run)
    for row in rows:
        row["source_sha256"] = "another_source"
        tool.write(run / "evaluation_validation_nominal" / (row["path_id"]+".json"), row)
    def forbidden(*args, **kwargs):
        pytest.fail("Stale-cache check must reject before running a new policy")
    monkeypatch.setattr(tool, "make_agent", forbidden)
    with pytest.raises(ValueError, match="(?i)metadata|source|changed|hash|stale|misbound"):
        tool.evaluate(run, "validation")


def test_freeze_checks_checkpoint_budget_not_only_summary(event_study, monkeypatch, tmp_path):
    tool = event_study
    monkeypatch.setattr(tool, "FORMAL", ("state_direct",))
    monkeypatch.setattr(tool, "SEEDS", (11,))
    run = fake_run(tool, monkeypatch, tmp_path, "state_direct", 11, "formal")
    torch.save({"steps": tool.FORMAL_STEPS-1}, run / "checkpoint.pt")
    tool.write(run / "training_summary.json", {"steps": tool.FORMAL_STEPS, "checkpoint_sha256": tool.sha(run / "checkpoint.pt")})
    tool.write(run / "evaluation_validation_nominal/summary.json", {"rows": validation_rows(tool, run)})
    with pytest.raises(ValueError, match="(?i)budget|incomplete|steps|checkpoint"):
        tool.freeze()
    assert not (tool.OUT / "freeze.json").exists()


def test_import_does_not_activate_or_rebind_predecessor_studies():
    code = """import sys
sys.path.insert(0,'PPO_project/tools')
import preview_direct_explore as previous
from src.utils import revision_metrics as metrics
s=previous.study
before=(s.OUT,s.config_for,s.make_agent,s.source_state,s.protocol_definition,s.v2.make_env,metrics.ordered_projection)
import preview_event_study
after=(s.OUT,s.config_for,s.make_agent,s.source_state,s.protocol_definition,s.v2.make_env,metrics.ordered_projection)
assert before==after, 'event import mutates predecessor study registration'
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
