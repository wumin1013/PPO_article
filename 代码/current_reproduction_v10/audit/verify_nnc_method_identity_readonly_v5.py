"""Read retained evidence and emit an identity/provenance audit; no rollout calls."""
from pathlib import Path
import ast
import hashlib
import json
import re
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT.parent / "JNN_revision_20261011_v3"
ARCHIVE = ROOT.parent / "retained" / "历史实验摘要与代码.zip"
OUT = Path(__file__).with_name("nnc_method_identity_readonly_v5.json")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def peaks(positions, dt):
    velocity = np.diff(positions, axis=0) / dt
    acceleration = np.diff(np.vstack([np.zeros(2), velocity]), axis=0) / dt
    jerk = np.diff(np.vstack([np.zeros(2), acceleration]), axis=0) / dt
    return [float(np.linalg.norm(a, axis=1).max()) for a in (velocity, acceleration, jerk)]


def main():
    suffixes = [
        "revision_20260908/historical/baseline_policy/baseline_policy.yaml",
        "revision_20260908/historical/source/PPO_project/src/environment/cnc_env.py",
        "revision_20260908/historical/source/PPO_project/tools/phase32_export_best_trajectories.py",
        "revision_20260913_experiments_v1/nnc_planar_environment.py",
        "revision_20260913_restart_30m_v1/nnc_geometry_index.py",
        "revision_20260913_restart_30m_v1/nnc_scan_adapter.py",
    ]
    recovered = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        names = archive.namelist()
        registration_entry = next(n for n in names if n.endswith("original_nnc_current_cases/registration.json"))
        registration_bytes = archive.read(registration_entry)
        registration = json.loads(registration_bytes)
        for suffix in suffixes:
            entry = next(n for n in names if n.endswith(suffix))
            data = archive.read(entry)
            old_path, registered_sha = next(
                (k, v) for k, v in registration["source_sha256"].items()
                if k.replace("\\", "/").endswith(suffix)
            )
            assert sha(data) == registered_sha, suffix
            record = dict(entry=entry, suffix=suffix, bytes=len(data), sha256=sha(data),
                          old_registered_path=old_path, matched_registration=True)
            if suffix.endswith("cnc_env.py"):
                source = data.decode("utf-8-sig")
                record["speed_cap_flag_occurrences"] = [
                    dict(line=i + 1, text=line)
                    for i, line in enumerate(source.splitlines()) if "_p4_speed_cap_enabled" in line
                ]
                record["methods"] = [dict(name=n.name, line=n.lineno, end_line=n.end_lineno)
                    for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef)
                    and n.name in ["_compute_p4_pre_step_status", "_scan_for_next_turn", "step"]]
            recovered.append(record)
        config_entries = [n for n in names if "original_nnc_current_cases/" in n and n.endswith("effective_config.json")]
        effective_configs = [dict(entry=n, sha256=sha(archive.read(n))) for n in config_entries]
        config = json.loads(archive.read(config_entries[0]))
        assert len({r["sha256"] for r in effective_configs}) == 1
        checkpoint_entries = [n for n in names if n.lower().endswith((".pt", ".pth"))]

    replay = ROOT / "baseline_records/nnc_fairness_v3/retained_replay_script.py"
    assert sha(replay.read_bytes()) == next(v for k, v in registration["source_sha256"].items()
                                           if k.endswith("check_original_nnc_current_cases.py"))
    binding = read(ROOT / "baseline_records/nnc_fairness_v3/retained_binding.json")
    prefix_checks = []
    for case, expected in binding["records"].items():
        folder = ROOT / "baseline_records/nnc_fairness_v3"
        result_file, trace_file = folder / f"{case}_result.json", folder / f"{case}_trace.npz"
        result = read(result_file)
        assert sha(result_file.read_bytes()) == expected["result_sha256"]
        assert sha(trace_file.read_bytes()) == expected["trace_sha256"] == result["trace_sha256"]
        with np.load(trace_file, allow_pickle=False) as trace:
            observed = peaks(trace["positions"], result["physical_interval_s"])
            shapes = {key: list(trace[key].shape) for key in ("positions", "observations", "actions")}
        target = list(expected["prefix_peaks_independently_reconstructed"].values())
        assert np.allclose(observed, target, rtol=1e-12, atol=1e-8)
        assert result["complete_time_s"] is None and not result["full_task_peaks"]
        prefix_checks.append(dict(case=case, result_sha256=sha(result_file.read_bytes()),
            trace_sha256=sha(trace_file.read_bytes()), shapes=shapes, vector_peaks=observed,
            peak_to_limit=[x / y for x, y in zip(observed, [100, 2000, 62500])],
            elapsed_prefix_s=result["observed_prefix_time_s"], complete_time_s=None,
            complete_time_comparison_eligible=False))

    analysis_file = ROOT / "manuscript/results_v3/analysis.json"
    analysis = read(analysis_file)
    assert sha(analysis_file.read_bytes()) == sha((V3 / "manuscript/results_v3/analysis.json").read_bytes())
    figure_registration_file = V3 / "prospective/final_v3/registration.json"
    figure_registration = read(figure_registration_file)
    methods = ["adaptive", "fixed", "S2R", "SCMO"]
    selected = {(path, method, 101 if method in ["adaptive", "fixed"] else None, .001)
                for path in figure_registration["representative_figure_paths"] for method in methods}
    trajectory_checks, figure_traces, failures = [], [], []
    for row in analysis["rows"]:
        result_file = V3 / row["result_file"]
        result_sha = sha(result_file.read_bytes())
        assert result_sha == analysis["inputs_sha256"][row["result_file"]]
        result = read(result_file)
        if not row["success"]:
            failures.append(dict(path=row["path"], method=row["method"], seed=row["seed"],
                                 dt_s=row["dt_s"], reason=row["reason"], result_sha256=result_sha))
            continue
        trace_file = V3 / row["trace_file"]
        trace_sha = sha(trace_file.read_bytes())
        assert trace_sha == result["trace_sha256"]
        with np.load(trace_file, allow_pickle=False) as trace:
            observed = peaks(trace["positions"], row["dt_s"])
            samples = len(trace["positions"])
        assert np.allclose(observed, [row[k] for k in ("v_peak", "a_peak", "j_peak")], rtol=1e-12, atol=1e-8)
        assert np.allclose(observed, result["vector_peaks"], rtol=1e-12, atol=1e-8)
        check = dict(path=row["path"], method=row["method"], seed=row["seed"], dt_s=row["dt_s"],
            result_file=str(result_file), result_sha256=result_sha, trace_file=str(trace_file),
            trace_sha256=trace_sha, samples=samples, complete_time_s=row["complete_time_s"], vector_peaks=observed)
        trajectory_checks.append(check)
        if (row["path"], row["method"], row["seed"], row["dt_s"]) in selected:
            figure_traces.append(check)
    assert len(figure_traces) == 8
    maximum = (np.max([r["vector_peaks"] for r in trajectory_checks], axis=0) / [100, 2000, 62500]).tolist()
    pending, active = ["manuscript.tex"], []
    while pending:
        name = pending.pop()
        if name in active:
            continue
        active.append(name)
        source = (ROOT / "shared_manuscript" / name).read_text("utf-8")
        pending.extend(re.findall(r"\\input\{([^}]+)\}", source))
    assert "generated_historical_main.tex" not in active
    metadata_path = ROOT.parent / "JMP_evidence_20261010_v2/literature/crossref/Li2020.json"
    metadata = read(metadata_path)["message"]
    report = dict(
        audit_scope="Read-only retained source, hash binding and position-difference verification; no imports of planner/agent code, model calls, training or new trajectories.",
        archive=str(ARCHIVE), archive_entries=len(names), registration_entry=registration_entry,
        registration_sha256=sha(registration_bytes), recovered_sources=recovered,
        effective_configs=effective_configs, model_files_in_this_zip=len(checkpoint_entries),
        checkpoint_registered_sha256="24d263a0e07fe471e7995ebae2d0a0ee7a2ee03e2feec80ea50aa5b94b0cfe6b",
        checkpoint_recovered=False, replay_sha256=sha(replay.read_bytes()),
        bound_config=dict(environment=config["environment"], experiment=config["experiment"],
            ppo=config["ppo"], training=config["training"], kinematic_constraints=config["kinematic_constraints"]),
        naming="Historical project NNC-inspired PPO adapter, with a planar cap-removal diagnostic; no independently verified complete reproduction of Li et al. established by this audit.",
        protected_mechanism_attribution="Recovered project source proves cap removal relative to the archived project Env. This audit has no primary-paper/full-original-code evidence assigning those caps or turn prebraking to Li et al.'s method.",
        old_prefix_checks=prefix_checks,
        current_constraints=dict(analysis_sha256=sha(analysis_file.read_bytes()), original_data_root=str(V3),
            figure_registration_sha256=sha(figure_registration_file.read_bytes()), active_inputs=active,
            methods=methods, representative_paths=figure_registration["representative_figure_paths"],
            figure_traces=figure_traces, verified_grid_rows=len(analysis["rows"]),
            verified_successful_traces=len(trajectory_checks), failures_retained=failures,
            successful_grid_maximum_normalized_peaks=maximum,
            contains_old_nnc_diagnostic=False, missing_source_data_in_v5_snapshot=True,
            plot_script_sha256=sha((ROOT / "manuscript/plot_final_v3.py").read_bytes()),
            prose_script_sha256=sha((ROOT / "manuscript/write_evidence_text_v3.py").read_bytes()),
            output_pdf_sha256=sha((ROOT / "shared_manuscript/constraints_v3.pdf").read_bytes()),
            numeric_scope="The generated maxima summarize successful grid rows; four planning failures have no traces and remain explicit in the table. This is historical v3 evidence, not a fresh v5 final test."),
        paper_metadata=dict(source=str(metadata_path), sha256=sha(metadata_path.read_bytes()),
            title=metadata["title"], author=metadata["author"], published=metadata["published"],
            doi=metadata["DOI"], volume=metadata["volume"], article_number=metadata["article-number"],
            primary_paper_read_in_this_audit=False),
    )
    OUT.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(output=str(OUT), recovered_sources=len(recovered),
        matching_old_prefixes=len(prefix_checks), checked_grid_rows=len(analysis["rows"]),
        checked_successful_traces=len(trajectory_checks), figure_traces=len(figure_traces),
        failures=len(failures), maximum_normalized_peaks=maximum)))


if __name__ == "__main__":
    main()
