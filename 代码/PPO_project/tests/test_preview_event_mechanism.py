"""Saved-data mechanism checks; no policy or environment execution."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "PPO_project/tools"))
import preview_event_mechanism as mechanism


def example(complete=True, crossed=True):
    trace = {field: np.zeros(5) for field in mechanism.FIELDS.values()}
    trace.update(timestamp_s=np.arange(5)*.001, lookahead_mm=np.array([4., 3., 3., 3., 3.]),
                 v_mm_s=np.array([0., 5., 5., 5., 5.]), preview_event=np.array([0., 1., 0., 1., 0.]),
                 kcm_intervention=np.array([0., .2, 2.4, .5, .1]), reward=np.array([0., .01, .01, .01, 10.]))
    trace[mechanism.FIELDS["progress"]] = np.array([np.nan, .90, .990, .995, 1. if complete else .996]) if crossed else np.array([np.nan, .1, .2, .3, .4])
    trace[mechanism.FIELDS["coverage"]][-1] = int(complete)
    for key in ("r_progress", "r_track", "r_smooth", "r_time", "r_terminal"):
        trace["reward_components__"+key] = np.array([np.nan, 0., 0., 0., 0.])
    trace["reward_components__r_terminal"][-1] = 10. if complete else -20.
    result = dict(variant="selected_hold", seed=11, path_id="mixed_lengths_validation", split="validation",
                  success=complete, path_completed=complete, failure_reasons=[] if complete else ["timeout"])
    cfg = {"revision": {"preview_period_steps": 2, "adaptive_preview": True, "preview_bounds_mm": [.8, 4.6]}}
    return trace, result, cfg


def test_initial_row_is_excluded_and_last_one_percent_uses_ordered_progress():
    trace, result, cfg = example()
    row = mechanism.trajectory_statistics(trace, result, cfg)
    assert row["distance_range_mm"] == 0 and row["distance_std_mm"] == 0
    assert row["preview_event_count"] == 2
    assert row["first_sample_at_99pct_s"] == pytest.approx(.002)
    assert row["last_1pct_completion_time_s"] == pytest.approx(.002)
    assert row["kcm_score_max"] == 2.4  # Never clamp this score to one.


@pytest.mark.parametrize("crossed", [True, False])
def test_incomplete_tail_is_not_filled_with_zero_or_claimed_complete(crossed):
    trace, result, cfg = example(complete=False, crossed=crossed)
    row = mechanism.trajectory_statistics(trace, result, cfg)
    assert row["last_1pct_completion_time_s"] is None
    assert row["observed_time_after_99pct_s"] == (pytest.approx(.002) if crossed else None)
    assert row["last_1pct_status"] == ("terminated_without_completion" if crossed else "threshold_not_reached")


def test_invalid_event_count_and_backwards_ordered_progress_are_rejected():
    trace, result, cfg = example()
    trace["preview_event"][2] = 1.
    with pytest.raises(ValueError, match="scheduled"):
        mechanism.trajectory_statistics(trace, result, cfg)
    trace, result, cfg = example()
    trace[mechanism.FIELDS["progress"]][3] = .2
    with pytest.raises(ValueError, match="monotone"):
        mechanism.trajectory_statistics(trace, result, cfg)


def test_csv_preserves_initial_nan_and_all_reward_fields(tmp_path):
    trace, _, _ = example()
    trace["reward_components__extra_observed_term"] = np.array([np.nan, .1, .2, .3, .4])
    path = tmp_path / "trace.csv"
    mechanism.export_trace(path, trace)
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["is_initial_row"] == "1"
    assert np.isnan(float(rows[0]["reward_components__extra_observed_term"]))
    assert float(rows[-1]["reward_components__extra_observed_term"]) == .4


def test_unfrozen_test_is_blocked_before_any_test_artifact_read(tmp_path):
    study, data = tmp_path / "study", tmp_path / "data"
    study.mkdir(); data.mkdir()
    manifest = data / "manifest.json"
    manifest.write_text('{"paths":[{"id":"held_test","split":"test"}]}', encoding="utf-8")
    protocol = dict(manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(), formal_variants=["selected_hold"], formal_seeds=[11])
    (study / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="completed formal freeze"):
        mechanism.collect("formal", "test", study=study, data=data, root=tmp_path)


def test_incomplete_freeze_is_rejected(tmp_path):
    study = tmp_path / "study"; study.mkdir()
    protocol = dict(source_sha256="source", manifest_sha256="manifest", formal_variants=["a", "b"], formal_seeds=[11, 22])
    (study / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    (study / "selection.json").write_text("{}", encoding="utf-8")
    freeze = dict(source_sha256="source", manifest_sha256="manifest",
                  protocol_sha256=hashlib.sha256((study/"protocol.json").read_bytes()).hexdigest(),
                  selection_sha256=hashlib.sha256((study/"selection.json").read_bytes()).hexdigest(),
                  models={"formal/a_seed11":"model"}, configs={"formal/a_seed11":"config"})
    (study / "freeze.json").write_text(json.dumps(freeze), encoding="utf-8")
    with pytest.raises(ValueError, match="every registered"):
        mechanism.check_test_freeze(study, protocol, mechanism.Inputs(tmp_path))


def test_jsonl_snapshot_retains_incomplete_tail(tmp_path):
    path = tmp_path / "updates.jsonl"
    path.write_bytes(b'{"step":1}\n{"step":')
    inputs = mechanism.Inputs(tmp_path)
    rows, partial = inputs.lines(path)
    assert rows == [{"step":1}] and partial
    assert next(iter(inputs.log_snapshots.values())) == path.read_bytes()


def test_formal_seed_is_fixed_and_vector_exports_keep_text(tmp_path):
    trace, result, cfg = example()
    stats = mechanism.trajectory_statistics(trace, result, cfg)
    bundle = dict(seed=22, variant="selected_hold", config=cfg,
                  paths=[dict(path_id=result["path_id"], trace=trace, statistics=stats)])
    with pytest.raises(ValueError, match="seed 11"):
        mechanism.mechanism_figure(bundle, "validation", tmp_path)
    bundle["seed"] = 11
    mechanism.style()
    files = mechanism.mechanism_figure(bundle, "validation", tmp_path)
    assert {path.suffix for path in files} == {".svg", ".pdf", ".png"}
    svg = next(path for path in files if path.suffix == ".svg").read_text(encoding="utf-8")
    assert "<text" in svg and "Seed 11" in svg


def test_import_does_not_import_runner_environment_or_torch():
    code = """import sys
sys.path.insert(0, 'PPO_project/tools')
import preview_event_mechanism
assert 'torch' not in sys.modules
assert 'preview_event_study' not in sys.modules
assert 'preview_event_report' not in sys.modules
assert not any(name.startswith('src.environment') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


@pytest.mark.parametrize("enabled", [False, True])
def test_figure_generation_requires_opt_in_and_preserves_old_manifest(tmp_path, monkeypatch, enabled):
    output = tmp_path / "data"; output.mkdir()
    figure_dir = tmp_path / "figures"; figure_dir.mkdir()
    image = figure_dir / "existing.png"; image.write_bytes(b"existing reference draft")
    old = json.dumps({"figures": [{"path": str(image), "sha256": hashlib.sha256(image.read_bytes()).hexdigest()}]}).encode()
    (output / "mechanism_manifest.json").write_bytes(old)
    called = []
    monkeypatch.setattr(mechanism, "learning_figure", lambda *a: called.append("learning") or [])
    monkeypatch.setattr(mechanism, "pilot_summary_figure", lambda *a: called.append("pilot") or [])
    collected = dict(stage="pilot", split="validation", methods=[], paths=[], learning=[], histories=[],
                     epochs=[], statistics=[], bundles=[], pending=[], inputs=mechanism.Inputs(tmp_path))
    options = {"generate_figures": True} if enabled else {}
    mechanism.build(collected, output, figure_dir, **options)
    assert called == (["learning", "pilot"] if enabled else [])
    assert image.read_bytes() == b"existing reference draft"
    saved = json.loads((output / "mechanism_manifest.json").read_text(encoding="utf-8"))
    assert saved["figure_generation_requested"] == enabled
    assert Path(saved["previous_reference_figure_manifest"]).read_bytes() == old
    mechanism.build(collected, output, figure_dir)
    repeated = json.loads((output / "mechanism_manifest.json").read_text(encoding="utf-8"))
    assert Path(repeated["previous_reference_figure_manifest"]).read_bytes() == old
