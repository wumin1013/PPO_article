"""Paper-table provenance and rendering tests with synthetic saved reports only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import preview_formal_paper_tables as tables
from test_preview_formal_holdout import frozen_study, put_json, artifact
from test_preview_formal_statistics import synthetic_rows


def reseal_report(root, freeze_path, rows):
    """Generate fixture artifacts, never read any real study outputs."""
    freeze = tables.read_json(freeze_path)
    freeze_digest = tables.sha(freeze_path)
    models = {(r["seed"], r["method"]): r for r in freeze["records"]}
    inputs = {freeze_path.relative_to(root).as_posix(): freeze_digest}
    for row in rows:
        model = models[row["seed"], row["method"]]
        row.update(freeze_sha256=freeze_digest,
                   checkpoint_sha256=model["artifacts"]["checkpoint.pt"]["sha256"],
                   config_sha256=model["artifacts"]["config.yaml"]["sha256"],
                   source_sha256=freeze["source"]["source_sha256"],
                   holdout_manifest_sha256=freeze["holdout_manifest"]["sha256"],
                   selected_step=model["selected_step"])
        directory = freeze_path.parent / "holdout" / f"{row['method']}_seed{row['seed']}"
        directory.mkdir(parents=True, exist_ok=True)
        trace = directory / f"{row['path_id']}.npz"
        trace.write_bytes(b"synthetic hash-only trace; this is deliberately not a loadable NPZ")
        timing = directory / f"{row['path_id']}_inference.json"
        put_json(timing, {"seconds": [0.01, 0.02]})
        row.update(trace=artifact(root, trace), inference_samples=artifact(root, timing))
        original = directory / f"{row['path_id']}.json"
        put_json(original, row)
        for path in (trace, timing, original):
            inputs[path.relative_to(root).as_posix()] = tables.sha(path)
    report_dir = freeze_path.parent / "reports" / "synthetic_report_v1"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / "report.json"
    expected_paths = {row["path_id"]: row["family"] for row in rows}
    put_json(report, {"freeze_sha256": freeze_digest, "rows": rows,
                      "statistics": tables.formal_stats.summarize(rows, freeze["seeds"], expected_paths)})
    for name in tables.REPORT_OUTPUTS - {"report.json"}:
        (report_dir / name).write_text("synthetic,csv\n", encoding="utf-8")
    put_json(report_dir / "manifest.json", {"inputs": inputs,
             "outputs": {name: tables.sha(report_dir / name) for name in tables.REPORT_OUTPUTS}})
    return report


@pytest.fixture
def saved_report(frozen_study):
    root, freeze_path, _ = frozen_study
    freeze = tables.read_json(freeze_path)
    protocol_path = root / freeze["protocol"]["file"]
    protocol = tables.read_json(protocol_path)
    source_name = "PPO_project/tools/preview_formal_statistics.py"
    (root / source_name).write_bytes(Path(tables.formal_stats.__file__).read_bytes())
    source_files = protocol["evaluation_sources"]["source_files"]
    source_files[source_name] = tables.sha(root / source_name)
    protocol["evaluation_sources"]["source_sha256"] = hashlib.sha256(
        json.dumps(source_files, sort_keys=True).encode()).hexdigest()
    put_json(protocol_path, protocol)
    freeze["protocol"] = artifact(root, protocol_path)
    put_json(freeze_path, freeze)
    rows = synthetic_rows()
    for row in rows:
        row["path_id"] = row["path_id"].removeprefix("synthetic_")
    report = reseal_report(root, freeze_path, rows)
    return root, freeze_path, report, rows


def rehash_report(report):
    path = report.parent / "manifest.json"
    manifest = tables.read_json(path)
    manifest["outputs"]["report.json"] = tables.sha(report)
    put_json(path, manifest)


def test_builds_twelve_main_rows_five_seed_rows_and_binds_all_outputs(saved_report):
    root, freeze, report, _ = saved_report
    original_hashes = {path: tables.sha(path) for path in (freeze, report, report.parent / "manifest.json")}
    result = tables.build_tables(report, freeze, root=root)
    output = Path(result["output"])
    assert output == report.parent / "paper_tables_v1"
    assert result["main_rows"] == 12 and result["seed_rows"] == 5
    data = tables.read_json(output / "paper_table_data.json")
    assert len(data["main_rows"]) == 12 and len(data["seed_rows"]) == 5
    assert data["main_rows"][0]["common_success_n"] == 5
    tex = (output / "formal_results.tex").read_text(encoding="utf-8")
    rendered_rows = [line for line in tex.splitlines() if line.startswith(tuple(f"P{i} &" for i in range(1, 7)))]
    assert len(rendered_rows) == 12
    assert all(line.endswith(r"\\") for line in rendered_rows)
    assert "10.000 \\pm 0.000" in tex and "8.000 \\pm 0.000" in tex
    seed_tex = (output / "formal_seeds.tex").read_text(encoding="utf-8")
    assert len([line for line in seed_tex.splitlines() if " & 256 & 256 & 4096+4096 & " in line]) == 5
    macros = (output / "formal_macros.tex").read_text(encoding="utf-8")
    assert r"\newcommand{\FormalOverallTimeReductionPercent}{20.00\%}" in macros
    manifest = tables.read_json(output / "manifest.json")
    assert manifest["statistics_recomputed_equal"] is True
    assert manifest["n_verified_outcomes"] == 60
    for name, digest in manifest["outputs"].items():
        assert tables.sha(output / name) == digest
        assert not (output / name).read_bytes().startswith(b"\xef\xbb\xbf")
    for path, digest in original_hashes.items():
        assert tables.sha(path) == digest
    with pytest.raises(ValueError, match="Preserve existing"):
        tables.build_tables(report, freeze, root=root)


def test_failed_fast_run_excluded_from_both_paired_time_cells_but_included_in_errors(saved_report):
    root, freeze, _, rows = saved_report
    bad = next(row for row in rows if row["seed"] == 11 and row["method"] == "adaptive")
    path_id = bad["path_id"]
    bad.update(success=False, elapsed_time_s=0.001, failure_reasons=["terminal_not_at_rest"],
               full_rest_satisfied=False, contour_error_max_mm=9.0, contour_error_rmse_mm=2.0)
    next(row for row in rows if row["seed"] == 11 and row["method"] == "fixed3"
         and row["path_id"] == path_id)["elapsed_time_s"] = 1000.0
    report = reseal_report(root, freeze, rows)
    data = tables.table_data(tables.verify_report(report, freeze, root=root))
    pair = {row["method"]: row for row in data["main_rows"] if row["path_id"] == path_id}
    assert pair["adaptive"]["successes"] == 4 and pair["fixed3"]["successes"] == 5
    assert pair["adaptive"]["common_success_n"] == pair["fixed3"]["common_success_n"] == 4
    assert pair["fixed3"]["common_time_mean_s"] == 10.0
    assert pair["adaptive"]["common_time_mean_s"] == 8.0
    assert pair["adaptive"]["all_runs_peak_error_worst_mm"] == 9.0
    assert pair["adaptive"]["all_runs_rmse_mean_mm"] == pytest.approx((2.0 + 4 * 0.01) / 5)
    assert pair["adaptive"]["common_success_seeds"] == [22, 33, 44, 55]


def test_no_joint_success_renders_na_not_zero(saved_report):
    root, freeze, _, rows = saved_report
    for row in rows:
        if (row["seed"] in (11, 22)) == (row["method"] == "fixed3"):
            row.update(success=False, elapsed_time_s=0.001, failure_reasons=["failed"])
    report = reseal_report(root, freeze, rows)
    verified = tables.verify_report(report, freeze, root=root)
    data = tables.table_data(verified)
    assert all(row["common_time_mean_s"] is None for row in data["main_rows"])
    assert all(row["common_success_n"] == 0 for row in data["main_rows"])
    assert r"\newcommand{\FormalOverallTimeReductionPercent}{NA}" in tables.render_macros(data)
    assert r"\newcommand{\FormalOverallTimeReductionCI}{NA}" in tables.render_macros(data)
    assert r"$\mathrm{NA}$" in tables.render_main(data)


@pytest.mark.parametrize("fault", ["raw_sha", "trace_sha", "report_sha", "statistics", "row_copy", "missing_row", "legacy_manifest", "training_log"])
def test_changed_or_legacy_sources_refuse_before_creating_tables(saved_report, fault):
    root, freeze, report, rows = saved_report
    if fault in ("raw_sha", "trace_sha"):
        if fault == "trace_sha":
            path = root / rows[0]["trace"]["file"]
        else:
            path = freeze.parent / "holdout" / f"{rows[0]['method']}_seed{rows[0]['seed']}" / f"{rows[0]['path_id']}.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif fault == "report_sha":
        report.write_text(report.read_text(encoding="utf-8") + " ", encoding="utf-8")
    elif fault in ("statistics", "row_copy", "missing_row"):
        value = tables.read_json(report)
        if fault == "statistics":
            value["statistics"]["overall"]["paired"]["time_reduction_fraction"] = 0.99
        elif fault == "row_copy":
            value["rows"][0]["elapsed_time_s"] += 100.0
        else:
            value["rows"].pop()
        put_json(report, value)
        rehash_report(report)
    elif fault == "legacy_manifest":
        put_json(report.parent / "manifest.json", {"files": {"report.json": tables.sha(report)}})
    else:
        model = tables.read_json(freeze)["records"][0]
        path = root / model["additional_budget_completion"]["file"]
        value = tables.read_json(path)
        value["decisions"] = 1
        put_json(path, value)
    with pytest.raises(ValueError):
        tables.build_tables(report, freeze, root=root)
    assert not (report.parent / "paper_tables_v1").exists()


def test_incomplete_budget_rejected_even_if_report_hashes_are_rebound(saved_report):
    root, freeze, _, rows = saved_report
    value = tables.read_json(freeze)
    value["records"][0]["available_additional_decisions"] = 256
    put_json(freeze, value)
    report = reseal_report(root, freeze, rows)
    with pytest.raises(ValueError, match="4096"):
        tables.build_tables(report, freeze, root=root)


def test_one_common_seed_has_na_sd_and_ci_and_nonfinite_error_is_not_hidden(saved_report):
    root, freeze, _, rows = saved_report
    for row in rows:
        if row["seed"] != 11:
            row.update(success=False, elapsed_time_s=0.001, failure_reasons=["failed"])
    rows[0]["contour_error_max_mm"] = float("nan")
    report = reseal_report(root, freeze, rows)
    data = tables.table_data(tables.verify_report(report, freeze, root=root))
    assert all(row["common_success_n"] == 1 and row["common_time_sd_s"] is None for row in data["main_rows"])
    affected = next(row for row in data["main_rows"] if row["path_id"] == rows[0]["path_id"] and row["method"] == rows[0]["method"])
    assert affected["peak_error_finite_n"] == 4
    assert affected["all_runs_peak_error_worst_mm"] is None
    assert r"\pm \mathrm{NA}" in tables.render_main(data)
    assert r"\newcommand{\FormalOverallTimeReductionCI}{NA}" in tables.render_macros(data)


def test_tex_escape_is_literal_and_safe():
    assert tables.tex_escape("path_a&b%") == r"path\_a\&b\%"
