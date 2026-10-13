"""Synthetic 60-outcome exporter tests; never read real holdout artifacts."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
import preview_formal_report_v2 as module
from test_preview_formal_holdout import frozen_study, put_json
from test_preview_formal_paper_tables import saved_report, reseal_report
from test_preview_formal_statistics import synthetic_rows, SEEDS, PATHS


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture
def export_fixture(saved_report, monkeypatch):
    root, freeze, existing_report, rows = saved_report
    validate = module.original.validate_freeze
    manifest = module.original.frozen_manifest
    read_json = module.original.read_json

    def guarded_read(path):
        assert Path(path).resolve().is_relative_to(root.resolve()), "Exporter attempted to read real data"
        return read_json(path)

    # Both original functions bind ROOT in their default arguments at definition
    # time, so updating the module variable alone would not isolate fixture data.
    monkeypatch.setattr(module.original, "ROOT", root)
    monkeypatch.setattr(module.original, "validate_freeze", lambda path: validate(path, root=root))
    monkeypatch.setattr(module.original, "frozen_manifest", lambda gate: manifest(gate, root=root))
    monkeypatch.setattr(module.original, "read_json", guarded_read)
    return root, freeze, existing_report, rows, freeze.parent/"reports/export_fixture_v2"


def test_group_scope_and_paired_scope_coexist_without_changing_statistics():
    stats = module.statistics.summarize(synthetic_rows(), SEEDS, PATHS)
    before = deepcopy(stats)
    methods, pairs = module.grouped_rows(stats)
    assert stats == before
    assert len(methods) == 30 and len(pairs) == 15
    assert {row["group_scope"] for row in pairs} == {"overall", "by_family", "by_seed", "by_path"}
    for row in pairs:
        source = stats["overall"] if row["group_scope"] == "overall" else stats[row["group_scope"]][row["label"]]
        assert {key: value for key, value in row.items() if key not in ("group_scope", "label")} == source["paired"]
        assert row["scope"] == source["paired"]["scope"]
        assert "Conditional on joint success" in row["scope"]


def test_complete_export_keeps_60_rows_original_stats_and_all_csv_groups(export_fixture):
    root, freeze, existing, rows, output = export_fixture
    before = {p: module.original.sha(p) for p in (freeze, existing, existing.parent/"manifest.json")}
    result = module.build_report(freeze, output)
    assert result["rows"] == 60
    report = module.original.read_json(output/"report.json")
    expected = module.statistics.summarize(rows, list(module.original.SEEDS), {r["path_id"]:r["family"] for r in rows})
    assert report["statistics"] == expected
    assert report["rows"] == rows
    assert len(csv_rows(output/"raw_metrics.csv")) == 60
    paired = csv_rows(output/"paired_statistics.csv")
    assert len(paired) == 15
    assert set(paired[0]) >= {"group_scope", "scope", "label"}
    assert all(row["scope"] == expected["overall"]["paired"]["scope"] for row in paired)
    assert len(csv_rows(output/"method_statistics.csv")) == 30
    manifest = module.original.read_json(output/"manifest.json")
    assert len(manifest["inputs"]) == 181  # freeze plus raw/trace/timing for 60 runs
    assert len(manifest["export_source_files"]) == 3
    for name, digest in manifest["outputs"].items():
        assert module.original.sha(output/name) == digest
    for path, digest in before.items():
        assert module.original.sha(path) == digest


def test_failed_fast_outcome_stays_in_60_rows_and_failure_denominator(export_fixture):
    root, freeze, _, rows, output = export_fixture
    failed = next(r for r in rows if r["method"] == "adaptive")
    failed.update(success=False, elapsed_time_s=.001, full_rest_satisfied=False,
                  failure_reasons=["terminal_not_at_rest"])
    reseal_report(root, freeze, rows)
    module.build_report(freeze, output)
    report = module.original.read_json(output/"report.json")
    assert len(report["rows"]) == 60
    adaptive = report["statistics"]["overall"]["adaptive"]
    paired = report["statistics"]["overall"]["paired"]
    assert adaptive["n_runs"] == 30 and adaptive["n_failure"] == 1
    assert adaptive["success_rate"] == pytest.approx(29/30)
    assert paired["n_common_success"] == 29
    assert paired["time_reduction_fraction"] == pytest.approx(.2)
    raw = csv_rows(output/"raw_metrics.csv")
    assert sum(row["success"] == "False" for row in raw) == 1
    assert any(json.loads(row["failure_reasons"]) == ["terminal_not_at_rest"] for row in raw)


@pytest.mark.parametrize("fault", ["missing_outcome", "changed_trace", "changed_timing", "wrong_source_binding", "changed_training_source", "changed_frozen_evaluator", "changed_training_log"])
def test_missing_or_changed_sources_refuse_before_creating_export(export_fixture, fault):
    root, freeze, _, rows, output = export_fixture
    row = rows[0]
    outcome = freeze.parent/"holdout"/f"{row['method']}_seed{row['seed']}"/f"{row['path_id']}.json"
    if fault == "missing_outcome":
        outcome.unlink()
    elif fault in ("changed_trace", "changed_timing"):
        field = "trace" if fault == "changed_trace" else "inference_samples"
        path = root/row[field]["file"]
        path.write_bytes(path.read_bytes()+b"changed")
    elif fault == "wrong_source_binding":
        value = module.original.read_json(outcome)
        value["source_sha256"] = "0"*64
        put_json(outcome, value)
    elif fault == "changed_training_source":
        (root/"PPO_project/src/continuation_fixture.py").write_text("# changed", encoding="utf-8")
    elif fault == "changed_frozen_evaluator":
        (root/"PPO_project/tools/preview_formal_statistics.py").write_text("# changed", encoding="utf-8")
    else:
        record = module.original.read_json(freeze)["records"][0]
        (root/record["checkpoint_dir"]/"updates.jsonl").write_bytes(b"changed")
    with pytest.raises((ValueError, FileNotFoundError)):
        module.build_report(freeze, output)
    assert not output.exists()


def test_duplicate_output_refuses_without_overwriting_prior_report(export_fixture):
    _, freeze, _, _, output = export_fixture
    module.build_report(freeze, output)
    before = {p.name:module.original.sha(p) for p in output.iterdir()}
    with pytest.raises(ValueError, match="Choose a new"):
        module.build_report(freeze, output)
    assert {p.name:module.original.sha(p) for p in output.iterdir()} == before


def test_export_outside_formal_reports_is_rejected(export_fixture):
    root, freeze, _, _, _ = export_fixture
    output = root/"unexpected_export"
    with pytest.raises(ValueError, match="Choose a new"):
        module.build_report(freeze, output)
    assert not output.exists()
