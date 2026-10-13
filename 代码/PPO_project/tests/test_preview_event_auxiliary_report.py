"""Read-only auxiliary reporting keeps failed and missing observations distinct."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import preview_event_auxiliary_report as report


def fixture(tmp_path, monkeypatch):
    for name, value in [("ROOT", tmp_path), ("PAPER", tmp_path / "paper"), ("STUDY", tmp_path / "study"),
                        ("DATA", tmp_path / "data"), ("REPLAY", tmp_path / "replay")]:
        monkeypatch.setattr(report, name, value)
    report.STUDY.mkdir()
    report.DATA.mkdir()
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    protocol = dict(source_sha256="fixture", formal_variants=["legacy_direct", "state_direct", "selected_fixed", "selected_hold"],
                    formal_seeds=[11, 22, 33, 44, 55], formal_steps=262144)
    write(report.STUDY / "protocol.json", protocol)
    write(report.DATA / "manifest.json", {"paths": [{"id": p, "split": "validation"} for p in ["a", "b", "c"]]})
    common = report.STUDY / "pilot/state_fixed_seed101"
    common.mkdir(parents=True)
    (common / "config.yaml").write_text(yaml.safe_dump({"environment": {"epsilon": 2.5, "interpolation_period": .001},
        "revision": {"episode_max_steps": 60000}, "kinematic_constraints": {"MAX_VEL": 100.}}), encoding="utf-8")
    write(common / "provenance.json", {"source_sha256": "fixture", "config_sha256": report.sha(common / "config.yaml")})
    return write


def test_import_is_independent_of_application_and_policy_runners():
    code = """import sys
sys.path.insert(0,'PPO_project/tools')
import preview_event_auxiliary_report
assert 'preview_event_study' not in sys.modules
assert 'preview_event_nnc_transfer' not in sys.modules
assert not any(name.startswith('src.') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2], check=True)


def test_all_failures_stay_in_descriptive_time_and_constraint_denominators():
    rows = [{"metrics": dict(success=True, valid_trace=True, elapsed_time_s=10., contour_error_max_mm=.4,
                failure_reasons=[], constraints={"feed_mm_s": dict(peak=20., satisfied=True, violating_samples=0)})},
            {"metrics": dict(success=False, valid_trace=True, elapsed_time_s=2., contour_error_max_mm=2.,
                failure_reasons=["constraint_feed_mm_s"], constraints={"feed_mm_s": dict(peak=101., satisfied=False, violating_samples=3)})}]
    result = report.summarize(rows)
    assert result["n"] == 2 and result["successes"] == 1 and result["failures"] == 1
    assert result["elapsed_time_all_s"]["mean"] == 6.
    assert result["constraints"]["feed_mm_s"]["failed_runs"] == 1
    assert result["constraints"]["feed_mm_s"]["peaks"]["maximum"] == 101.
    assert result["constraints"]["feed_mm_s"]["violating_samples"] == 3


def test_invalid_trace_is_not_zero_time_or_zero_constraint_violations():
    result = report.summarize([{"metrics": dict(success=False, valid_trace=False, failure_reasons=["invalid_trace"])}])
    assert result["n"] == result["failures"] == result["invalid_traces"] == 1
    assert result["elapsed_time_all_s"]["mean"] is None
    assert result["elapsed_time_all_s"]["missing_n"] == 1
    for value in result["constraints"].values():
        assert value["unknown_runs"] == 1 and value["peaks"]["mean"] is None
        assert value["violating_samples"] is None


def test_test_freeze_rejection_precedes_reference_and_result_reads(monkeypatch):
    seen = []
    def deny(inputs):
        seen.append("gate")
        raise ValueError("Incomplete freeze")
    monkeypatch.setattr(report, "require_complete_freeze", deny)
    monkeypatch.setattr(report.Inputs, "read", lambda *a, **k: pytest.fail("Test input read before gate"))
    with pytest.raises(ValueError, match="Incomplete freeze"):
        report.collect("test")
    assert seen == ["gate"]


def test_test_gate_rejects_incomplete_model_set(tmp_path, monkeypatch):
    write = fixture(tmp_path, monkeypatch)
    write(report.STUDY / "selection.json", {})
    frozen = dict(models={}, configs={}, source_sha256="fixture",
        protocol_sha256=report.sha(report.STUDY / "protocol.json"), selection_sha256=report.sha(report.STUDY / "selection.json"),
        manifest_sha256=report.sha(report.DATA / "manifest.json"))
    write(report.STUDY / "freeze.json", frozen)
    with pytest.raises(ValueError, match="roster"):
        report.require_complete_freeze(report.Inputs())


def test_missing_auxiliary_jobs_are_pending_not_failed_observations(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    result = report.collect("validation")
    assert result["expected_trajectories"] == 18 and len(result["pending"]) == 18
    assert result["rows"] == [] and not result["complete"]
    assert all(s["n"] == s["failures"] == 0 for s in result["summaries"])
    assert "Incomplete" in report.table(result)


def test_partial_report_requires_label_and_preserves_prior_outputs(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    out = report.STUDY / "reports/preview_event_auxiliary_validation"
    monkeypatch.setattr(sys, "argv", ["preview_event_auxiliary_report.py", "--split", "validation"])
    with pytest.raises(ValueError, match="incomplete"):
        report.main()
    assert not out.exists()
    monkeypatch.setattr(sys, "argv", ["preview_event_auxiliary_report.py", "--split", "validation", "--allow-partial"])
    report.main()
    before = (out / "report.json").read_bytes()
    saved = json.loads(before)
    assert not saved["complete"] and len(saved["pending"]) == 18
    assert "Incomplete" in (report.PAPER / "generated/preview_event_auxiliary_validation.tex").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        report.main()
    assert (out / "report.json").read_bytes() == before


def test_archive_member_must_match_recorded_source_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "ROOT", tmp_path)
    source = tmp_path / "source.py"
    source.write_bytes(b"preserved-source")
    mapping = {"source.py": report.sha(source)}
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("source.py", b"different-source")
    with pytest.raises(ValueError, match="ZIP member"):
        report.verify_sources(tmp_path, mapping, report.digest(mapping), report.Inputs(), archive)


def test_registered_file_hash_cannot_be_replaced_silently(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "ROOT", tmp_path)
    source = tmp_path / "trace.npz"
    source.write_bytes(b"changed-data")
    with pytest.raises(ValueError, match="hash mismatch"):
        report.Inputs().file(source, hashlib.sha256(b"original-data").hexdigest())
