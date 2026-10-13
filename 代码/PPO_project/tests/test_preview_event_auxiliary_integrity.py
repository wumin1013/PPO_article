"""Independent auxiliary-tool tests using offline fixtures, never held-out paths."""
from __future__ import annotations

import builtins
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import preview_event_report as report
import preview_event_nnc_transfer as transfer
import preview_event_inference as inference
import verify_preview_event_delivery as delivery


def row(method, seed, path, elapsed, success=True):
    return dict(variant=method, seed=seed, path_id=path, elapsed_time_s=elapsed,
        success=success, failure_reasons=[] if success else ["timeout"],
        contour_error_max_mm=.4, contour_error_rmse_mm=.2, inference_median_us=50.)


def test_invalid_trace_remains_a_failure_with_missing_metrics_not_zeros():
    invalid = dict(variant="selected_hold", seed=22, path_id="a", success=False,
                   valid_trace=False, failure_reasons=["invalid_trace"])
    result = report.summarize([row("selected_hold", 11, "a", 10.), invalid])
    assert result["n"] == 2 and result["failures"] == 1
    assert result["failure_reasons"] == {"invalid_trace": 1}
    assert result["successful_seeds"] == [11]
    assert result["all_time_s"]["n"] == 1
    assert result["all_time_s"]["mean"] == 10.
    assert result["all_peak_error_mm"]["n"] == 1


def test_suite_statistic_uses_one_total_per_independent_seed():
    rows = [row("selected_fixed", s, p, 10.) for s in [11, 22, 33] for p in ["a", "b"]]
    rows += [row("selected_hold", 11, "a", 1.), row("selected_hold", 11, "b", 19.),
             row("selected_hold", 22, "a", 7.), row("selected_hold", 22, "b", 9.),
             row("selected_hold", 33, "a", .1, False), row("selected_hold", 33, "b", 1.)]
    pairs = report.paired(rows, ["selected_fixed", "selected_hold"], ["a", "b"], [11, 22, 33])
    suite = next(p for p in pairs if p["path_id"] == "all_paths")
    assert suite["improvement_pct"]["n"] == 2  # Not four trajectories or six runs.
    assert suite["improvement_pct"]["mean"] == pytest.approx(10.)
    assert suite["improvement_pct"]["std"] == pytest.approx(np.sqrt(200.))
    assert [p["candidate_time_s"] for p in suite["pairs"]] == [20., 16.]
    assert suite["excluded"] == [{"seed": 33, "reason": "not_jointly_successful"}]


def report_fixture(tmp_path, monkeypatch):
    study = tmp_path / "study"
    data = tmp_path / "data"
    study.mkdir()
    data.mkdir()
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(report, "STUDY", study)
    monkeypatch.setattr(report, "DATA", data)
    protocol = dict(formal_variants=["legacy_direct", "state_direct", "selected_fixed", "selected_hold"],
                    formal_seeds=[11, 22, 33, 44, 55], formal_steps=262144, source_sha256="saved-source")
    report.write(study / "protocol.json", protocol)
    report.write(study / "selection.json", {"selected_fixed": "fixed3", "selected_hold": "hold50"})
    report.write(data / "manifest.json", {"paths": [{"id": "test_fixture_no_geometry", "split": "test"}]})
    frozen = dict(source_sha256="saved-source", protocol_sha256=report.sha(study / "protocol.json"),
        selection_sha256=report.sha(study / "selection.json"), manifest_sha256=report.sha(data / "manifest.json"),
        models={}, configs={})
    for variant in protocol["formal_variants"]:
        for seed in protocol["formal_seeds"]:
            key = f"formal/{variant}_seed{seed}"
            run = study / key
            run.mkdir(parents=True)
            (run / "checkpoint.pt").write_bytes(key.encode())
            (run / "config.yaml").write_text(key, encoding="utf-8")
            frozen["models"][key] = report.sha(run / "checkpoint.pt")
            frozen["configs"][key] = report.sha(run / "config.yaml")
    report.write(study / "freeze.json", frozen)
    return study, frozen


@pytest.mark.parametrize("tamper", ["missing_model", "missing_config", "unreported_last_checkpoint"])
def test_partial_test_report_still_requires_entire_freeze(tmp_path, monkeypatch, tamper):
    study, frozen = report_fixture(tmp_path, monkeypatch)
    if tamper == "missing_model":
        frozen["models"].pop(next(iter(frozen["models"])))
        report.write(study / "freeze.json", frozen)
    elif tamper == "missing_config":
        frozen["configs"].pop(next(iter(frozen["configs"])))
        report.write(study / "freeze.json", frozen)
    else:
        key = sorted(frozen["models"])[-1]
        (study / key / "checkpoint.pt").write_bytes(b"changed but no summary exists")
    with pytest.raises(ValueError):
        report.collect("formal", "test", "nominal", allow_partial=True)


def test_direct_test_worker_rejects_gate_failure_before_archived_imports(tmp_path, monkeypatch):
    job = tmp_path / "request.json"
    transfer.write(job, dict(split="test", seed=11, paths=[], event_freeze_sha256="unavailable",
        event_source_sha256="fixture", manifest_sha256="fixture", script_sha256=transfer.sha(transfer.__file__)))
    invocations = []
    def rejected_gate(command, **kwargs):
        invocations.append(command)
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="No complete event freeze")
    monkeypatch.setattr(transfer.subprocess, "run", rejected_gate)
    original_import = builtins.__import__
    def guard(name, *args, **kwargs):
        if name == "preview_unified_baselines":
            pytest.fail("An unregistered test request reached archived policy imports")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guard)
    original_path = sys.path.copy()
    try:
        with pytest.raises((RuntimeError, ValueError, subprocess.CalledProcessError)):
            transfer.archive_worker(job)
    finally:
        sys.path[:] = original_path
    assert invocations and any("--verify-freeze-only" in command for command in invocations)


@pytest.mark.parametrize("budget_failure", [False, True])
def test_shared_gate_checks_training_and_validation_for_all_models(tmp_path, monkeypatch, budget_failure):
    directory, _ = report_fixture(tmp_path, monkeypatch)
    protocol = report.read(directory / "protocol.json")
    train_calls, validation_calls = [], []
    def verify_training(run, budget):
        train_calls.append((run, budget))
        if budget_failure and len(train_calls) == 20:
            raise ValueError("Last registered model budget incomplete")
    fixture = SimpleNamespace(OUT=directory, DATA=tmp_path / "data", FORMAL=protocol["formal_variants"],
        SEEDS=protocol["formal_seeds"], FORMAL_STEPS=protocol["formal_steps"], check_protocol=lambda: None,
        source_state=lambda: {"source_sha256": "saved-source"}, verify_training=verify_training,
        validation_rows=lambda run: validation_calls.append(run))
    if budget_failure:
        with pytest.raises(ValueError, match="budget incomplete"):
            transfer.require_freeze(fixture)
        assert len(validation_calls) == 19
    else:
        transfer.require_freeze(fixture)
        assert len(validation_calls) == 20
    assert len(train_calls) == 20
    assert all(budget == 262144 for _, budget in train_calls)


def test_worker_rejects_rehashed_but_wrong_registered_geometry(tmp_path, monkeypatch):
    monkeypatch.setattr(transfer, "ROOT", tmp_path)
    directory = tmp_path / "PPO_project/data/preview_event_v1"
    directory.mkdir(parents=True)
    reference = directory / "reference.csv"
    np.savetxt(reference, [[0., 0.], [1., 0.]], delimiter=",", header="x_mm,y_mm", comments="")
    manifest = directory / "manifest.json"
    transfer.write(manifest, {"paths": [{"id": "offline_validation", "split": "validation",
        "reference_file": "reference.csv", "reference_sha256": transfer.sha(reference)}]})
    copied = tmp_path / "wrong_reference.csv"
    np.savetxt(copied, [[0., 0.], [1.01, 0.]], delimiter=",", header="x_mm,y_mm", comments="")
    model_run = tmp_path / "model"
    model_run.mkdir()
    (model_run / "config.yaml").write_text(yaml.safe_dump({"environment": {}, "revision": {}}), encoding="utf-8")
    fake_old = SimpleNamespace(ROOT=transfer.REPLAY,
        baseline_runs=lambda: [{"baseline": "nnc", "seed": 11, "run": model_run}],
        transferred_rollout=lambda *a, **k: pytest.fail("Mislabeled reference reached an archived policy"))
    monkeypatch.setitem(sys.modules, "preview_unified_baselines", fake_old)
    job = tmp_path / "request.json"
    transfer.write(job, dict(split="validation", seed=11, script_sha256=transfer.sha(transfer.__file__),
        manifest_sha256=transfer.sha(manifest), common=dict(epsilon=2.5, dt_s=.001, max_steps=10, limits={}),
        paths=[dict(id="offline_validation", reference_file=str(copied), reference_sha256=transfer.sha(copied))]))
    original_path = sys.path.copy()
    try:
        with pytest.raises(ValueError, match="registered split geometry"):
            transfer.archive_worker(job)
    finally:
        sys.path[:] = original_path
    assert not (tmp_path / "offline_validation.npz").exists()


def test_existing_worker_outputs_are_rejected_before_imports(tmp_path, monkeypatch):
    job = tmp_path / "request.json"
    manifest = transfer.ROOT / "PPO_project/data/preview_event_v1/manifest.json"
    paths = [{"id": p["id"]} for p in transfer.read(manifest)["paths"] if p["split"] == "validation"]
    transfer.write(job, dict(split="validation", paths=paths, manifest_sha256=transfer.sha(manifest),
                            script_sha256=transfer.sha(transfer.__file__)))
    target = tmp_path / (paths[0]["id"] + ".npz")
    target.write_bytes(b"preserve this result")
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        transfer.archive_worker(job)
    assert target.read_bytes() == before


@pytest.mark.parametrize("changed_worker_script", [False, True])
def test_transfer_executes_saved_snapshot_and_verifies_worker_identity(tmp_path, monkeypatch, changed_worker_script):
    import prepare_auxiliary_replay
    study = inference.study
    data = tmp_path / "data"
    data.mkdir()
    transfer.write(data / "manifest.json", {"paths": [{"id": "offline_validation_fixture", "split": "validation"}]})
    monkeypatch.setattr(study, "OUT", tmp_path / "study")
    monkeypatch.setattr(study, "DATA", data)
    monkeypatch.setattr(study, "check_protocol", lambda: None)
    monkeypatch.setattr(study, "source_state", lambda: {"source_sha256": "event-fixture"})
    monkeypatch.setattr(study, "manifest", lambda: transfer.read(data / "manifest.json"))
    reference = np.array([[0., 0.], [.1, 0.], [.2, 0.]])
    monkeypatch.setattr(study, "points", lambda _: reference.copy())
    monkeypatch.setattr(study, "config_for", lambda *a: dict(environment=dict(epsilon=.2, interpolation_period=.001),
        revision=dict(episode_max_steps=100), kinematic_constraints={"MAX_VEL": 100., "MAX_ACC": 2000., "MAX_JERK": 62500.}))
    monkeypatch.setattr(prepare_auxiliary_replay, "verify", lambda _: {"source_bytes_verified": True})
    dest = study.OUT / "nnc_transfer/validation_fixture"
    launched = []
    def simulated_archived_process(command, **kwargs):
        launched.append(command)
        snapshot = dest / "evaluation_adapter.py"
        assert Path(command[1]).resolve() == snapshot.resolve()
        assert snapshot.read_bytes() == Path(transfer.__file__).read_bytes()
        request = transfer.read(dest / "request.json")
        assert request["script_sha256"] == transfer.sha(snapshot)
        for spec in request["paths"]:
            file = dest / (spec["id"] + ".npz")
            np.savez_compressed(file, columns=np.array(["timestamp_s", "x_mm", "y_mm", "heading_rad"]),
                                data=np.array([[0., 0., 0., 0.], [.1, .1, 0., 0.], [.2, .2, 0., 0.]]))
            transfer.write(dest / (spec["id"] + ".worker.json"), dict(reason="success", runtime={},
                request_sha256=transfer.sha(dest / "request.json"), trace_sha256=transfer.sha(file),
                script_sha256="different-worker" if changed_worker_script else transfer.sha(snapshot)))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(transfer.subprocess, "run", simulated_archived_process)
    monkeypatch.setattr(sys, "argv", ["preview_event_nnc_transfer.py", "--split", "validation", "--seed", "11", "--output", str(dest)])
    if changed_worker_script:
        with pytest.raises(ValueError):
            transfer.main()
        assert not (dest / "summary.json").exists()
    else:
        transfer.main()
        result = transfer.read(dest / "offline_validation_fixture.json")
        assert result["metrics"]["witness_spacing_mm"] == .01
        assert result["metrics"]["initial_velocity_supplied"]
        assert result["metrics"]["initial_acceleration_supplied"]
    trace = dest / "offline_validation_fixture.npz"
    previous = trace.read_bytes()
    with pytest.raises(FileExistsError):
        transfer.main()
    assert trace.read_bytes() == previous and len(launched) == 1


def test_delivery_comparator_reports_missing_nested_field():
    mismatch = delivery.differences({"success": False}, {"success": False, "constraints": {"feed": {"peak": 1.}}})
    assert mismatch and any("constraints" in field for field in mismatch)


def test_inference_freeze_rejection_precedes_state_generation(tmp_path, monkeypatch):
    destination = tmp_path / "never_created"
    monkeypatch.setattr(sys, "argv", ["preview_event_inference.py", "--repeats", "100", "--output", str(destination)])
    def deny(study):
        raise ValueError("Unfrozen")
    monkeypatch.setattr(inference, "require_freeze", deny)
    monkeypatch.setattr(inference.study, "EventPreviewEnvironment", lambda *a, **k: pytest.fail("Unfrozen benchmark created states"))
    with pytest.raises(ValueError, match="Unfrozen"):
        inference.main()
    assert not destination.exists()


def test_serial_inference_saves_calls_per_model_without_treating_calls_as_seeds(tmp_path, monkeypatch):
    study = inference.study
    variants, seeds = ("legacy_direct", "state_direct", "selected_fixed", "selected_hold"), (11, 22)
    monkeypatch.setattr(study, "OUT", tmp_path)
    monkeypatch.setattr(study, "FORMAL", variants)
    monkeypatch.setattr(study, "SEEDS", seeds)
    monkeypatch.setattr(inference, "require_freeze", lambda _: None)
    monkeypatch.setattr(study, "source_state", lambda: {"source_sha256": "fixture"})
    monkeypatch.setattr(study, "config_for", lambda *a: {"revision": {"training_path_ids": ["training_fixture"]}})
    paths = []
    class Environment:
        def __init__(self, cfg, path_id):
            paths.append(path_id)
            self.state = np.arange(26, dtype=np.float32)
        def step(self, action):
            return self.state.copy(), 0., False, {}
    monkeypatch.setattr(study, "EventPreviewEnvironment", Environment)
    calls = []
    class Actor:
        def __init__(self, dim):
            self.dim = dim
            self.calls = 0
            calls.append(self)
        def load_state_dict(self, state):
            assert state == {}
        def eval(self):
            return self
        def __call__(self, tensor):
            assert tensor.shape == (1, self.dim)
            self.calls += 1
            return torch.zeros((1, 3)), None
    monkeypatch.setattr(study, "make_agent", lambda cfg, env: SimpleNamespace(actor=Actor(env.observation_dim)))
    frozen = {"models": {}, "configs": {}}
    for variant in variants:
        for seed in seeds:
            key = f"formal/{variant}_seed{seed}"
            run = tmp_path / key
            run.mkdir(parents=True)
            cfg = {"revision": {"observation_dim": 20 if variant == "legacy_direct" else 26}}
            (run / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
            torch.save({"actor": {}}, run / "checkpoint.pt")
            frozen["models"][key] = study.sha(run / "checkpoint.pt")
            frozen["configs"][key] = study.sha(run / "config.yaml")
    study.write(tmp_path / "freeze.json", frozen)
    destination = tmp_path / "timing"
    monkeypatch.setattr(sys, "argv", ["preview_event_inference.py", "--repeats", "100", "--output", str(destination)])
    inference.main()
    summary = study.read(destination / "summary.json")
    assert paths == ["training_fixture"]
    assert len(summary["rows"]) == len(variants) * len(seeds)
    assert "Independent trained model" in summary["statistical_unit"]
    assert all(actor.calls == 300 for actor in calls)
    with np.load(destination / "latencies.npz") as samples:
        assert len(samples.files) == 8
        assert all(samples[key].shape == (100,) for key in samples.files)
    assert summary["states_sha256"] == study.sha(destination / "common_training_states.npy")
    assert summary["samples_sha256"] == study.sha(destination / "latencies.npz")
    with pytest.raises(FileExistsError):
        inference.main()
