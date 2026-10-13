"""Read-only report gates, snapshot identity, and cumulative cost accounting."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
import continued_study_report as report


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture(params=["full_batch", "target_tracking"])
def snapshot(tmp_path, monkeypatch, request):
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(report, "BASE", tmp_path/"continued")
    monkeypatch.setattr(report, "MANIFEST", tmp_path/"paths.json")
    monkeypatch.setattr(report, "INPUTS", {})
    study = request.param
    root = report.BASE/report.SPECS[study][0]
    live = root/"development/preview_time_fixed3_seed101"
    folder = live/"checkpoints/step0"
    folder.mkdir(parents=True)
    write_json(report.MANIFEST, {"paths": []})
    source = tmp_path/"source.py"
    source.write_text("# fixture only\n", encoding="utf-8")
    files = {"source.py": digest(source)}
    source_hash = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    protocol = {"source": {"source_files": files, "source_sha256": source_hash}, "batch_macro_decisions": 64}
    write_json(root/"protocol.json", protocol)
    report.read(root/"protocol.json")
    prior = tmp_path/"prior/checkpoint.pt"
    prior.parent.mkdir(); prior.write_bytes(b"opaque fixture prior; must not deserialize")
    write_json(prior.parent/"training_summary.json", dict(checkpoint_sha256=digest(prior), decisions=4096, physics_steps=80000, command_steps=81000, training_wall_s=10.))
    cfg = {"revision": {"init_checkpoint": "prior/checkpoint.pt", "init_checkpoint_sha256": digest(prior), "pretraining_macro_decisions": 4096}}
    (folder/"config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (folder/"checkpoint.pt").write_bytes(b"opaque fixture checkpoint; must not deserialize")
    provenance = {"source_files": files, "source_sha256": source_hash, "config_sha256": digest(folder/"config.yaml"), "protocol_sha256": digest(root/"protocol.json")}
    write_json(folder/"provenance.json", provenance)
    with zipfile.ZipFile(folder/"source.zip", "w") as z:
        z.write(source, "source.py")
    training = dict(status="registered_development_milestone_complete", decisions=0, method="fixed3", seed=101,
        optimized_decisions=0, pending_buffer_decisions=0, zero_step_initial_diagnostic=True, optimizer_updates=0,
        episodes=0, physics_steps=0, command_steps=0, training_wall_s=0., checkpoint_sha256=digest(folder/"checkpoint.pt"))
    write_json(folder/"training_summary.json", training)
    (folder/"updates.jsonl").write_bytes(b""); (folder/"episodes.jsonl").write_bytes(b"")
    np.savez(folder/"training_final_partial.npz", data=np.empty((0, 0)))
    members = {p.name: digest(p) for p in folder.iterdir() if p.is_file()}
    write_json(folder/"snapshot_manifest.json", {"decisions": 0, "files": members})
    write_json(folder/"validation/binding.json", dict(split="validation", source_sha256=source_hash,
        checkpoint_sha256=digest(folder/"checkpoint.pt"), config_sha256=digest(folder/"config.yaml"), manifest_sha256=digest(report.MANIFEST)))
    write_json(folder/"validation/summary.json", {"rows": []})
    return root, live, folder, protocol, study


def test_registered_steps_preserve_old_and_new_protocols():
    assert report.registered_steps("positive", {"additional_macro_decisions": 4096}) == [4096]
    assert report.registered_steps("local", {"additional_validation_milestones": [4096, 8192]}) == [4096, 8192]
    assert report.registered_steps("full_batch", {"validation_milestones": [0, 256, 512]}) == [0, 256, 512]
    assert report.registered_steps("target_tracking", {"validation_milestones": [0, 256, 512, 1024]}) == [0, 256, 512, 1024]


def test_step_zero_is_bound_but_not_selection_eligible(snapshot):
    _, live, folder, protocol, study = snapshot
    result, rows = report.process_checkpoint(study, folder, live, 0, protocol, set())
    assert result["status"] == "complete" and not rows
    assert result["initial_diagnostic"] and not result["selection_eligible"]
    assert result["attributed_total_decisions"] == 4096
    assert result["training_diagnostic"]["log_path"].startswith(result["checkpoint_directory"])


def test_pending_snapshot_never_reads_checkpoint(snapshot, monkeypatch):
    _, live, folder, protocol, study = snapshot
    (folder/"validation/summary.json").unlink()
    def forbidden(*args, **kwargs):
        raise AssertionError("Pending checkpoint must not be read")
    monkeypatch.setattr(report, "bind", forbidden)
    result, rows = report.process_checkpoint(study, folder, live, 0, protocol, set())
    assert result["status"] == "pending" and not rows


def test_checkpoint_tampering_is_rejected(snapshot):
    _, live, folder, protocol, study = snapshot
    (folder/"checkpoint.pt").write_bytes(b"changed")
    with pytest.raises(AssertionError, match="SHA mismatch"):
        report.process_checkpoint(study, folder, live, 0, protocol, set())


def test_manifest_without_committed_logs_is_rejected(snapshot):
    _, live, folder, protocol, study = snapshot
    path = folder/"snapshot_manifest.json"
    content = json.loads(path.read_text(encoding="utf-8"))
    content["files"].pop("episodes.jsonl")
    write_json(path, content)
    with pytest.raises(AssertionError, match="Incomplete snapshot"):
        report.process_checkpoint(study, folder, live, 0, protocol, set())


def test_step_zero_cannot_be_smuggled_into_saved_selection(snapshot):
    root, live, folder, protocol, study = snapshot
    checkpoint, _ = report.process_checkpoint(study, folder, live, 0, protocol, set())
    selected = dict(method="fixed3", step=0, checkpoint=checkpoint["checkpoint_directory"], checkpoint_sha256=checkpoint["checkpoint_sha256"],
                    validation_sha256=digest(folder/"validation/summary.json"), successes=0, count=0)
    write_json(root/"selection_budget256.json", dict(budget=256, zero_step_excluded=True,
        records=[dict(selected=selected, all_checkpoints=[selected])], uniform_fixed=selected))
    with pytest.raises(AssertionError, match="Step0"):
        report.recorded_selections([study], [checkpoint], 256)


def test_cumulative_budget_deduplicates_milestones_and_prior():
    prior = dict(checkpoint_sha256="same-prior", decisions=4096, optimizer_updates=71, adam_steps_per_network=284)
    def record(run, steps, status="complete"):
        return dict(live_run=run, additional_decisions=steps, status=status, prior=prior,
                    training=dict(decisions=steps, physics_steps=steps*20, command_steps=steps*20+5,
                                  training_wall_s=steps/100, environment_replay_wall_s=steps/1000,
                                  optimizer_updates=steps//64, adam_steps_per_network=steps//16))
    result = report.cumulative_cost([record("a", 0), record("a", 256), record("a", 512), record("b", 256), record("b", 1024, "pending")])
    assert result["cumulative_new_decisions"] == 768
    assert result["cumulative_recorded_optimizer_updates"] == 12
    assert result["unique_prior_decisions"] == 4096
    assert result["completed_run_count"] == 2
    assert result["cumulative_recorded_recovery_wall_s"] == pytest.approx(.768)
    assert result["cumulative_recorded_adam_steps_per_network"] == 48
    assert result["unique_prior_recorded_optimizer_updates"] == 71
    assert result["unique_prior_recorded_adam_steps_per_network"] == 284


def test_retained_initializer_includes_better_baseline_and_failed_paths():
    paths = {"mixed", "dense"}
    def row(path, success, elapsed, checkpoint="adaptive256"):
        return dict(path_id=path, success=success, elapsed_time_s=elapsed,
                    checkpoint_directory=checkpoint, reference_length_mm=10., constraints={"speed": {"limit": 100.}})
    full = [dict(method="adaptive", seed=101, additional_decisions=256, checkpoint_directory="adaptive256")]
    rows = [row("mixed", True, 4.), row("dense", False, .3)]
    baselines = [dict(baseline_id="full_batch_step0_fixed3", successes=2,
                     rows=[row("mixed", True, 5.), row("dense", True, 5.2)]),
                 dict(baseline_id="sensor_original_fixed3", successes=2,
                     rows=[row("mixed", True, 5.), row("dense", True, 5.1)])]
    result = report.compare_retained_initializers(full, rows, paths, baselines)
    assert len(result) == 2
    assert {r["baseline_id"] for r in result} == {b["baseline_id"] for b in baselines}
    for r in result:
        assert r["adaptive_successes"] == 1 and r["fixed_successes"] == 2
        assert not r["full_suite_success"] and r["common_success_path_ids"] == ["mixed"]
        assert r["improvement_pct"] == 20.
        assert next(p for p in r["per_path"] if p["path_id"] == "dense")["improvement_pct"] is None


def test_retained_initializer_comparison_does_not_mix_studies():
    def row(checkpoint, time):
        return dict(path_id="mixed", success=True, elapsed_time_s=time, checkpoint_directory=checkpoint,
                    reference_length_mm=10., constraints={"speed": {"limit": 100.}})
    checkpoints = [dict(study=study, method="adaptive", seed=101, additional_decisions=256,
                        checkpoint_directory=study) for study in ("full_batch", "target_tracking")]
    rows = [row("full_batch", 4.), row("target_tracking", 3.)]
    baselines = [dict(study=study, baseline_id="sensor_original_fixed3", successes=1, rows=[row("prior", 5.)])
                 for study in ("full_batch", "target_tracking")]
    result = report.compare_retained_initializers(checkpoints, rows, {"mixed"}, baselines)
    assert len(result) == 2
    assert {r["study"]: r["improvement_pct"] for r in result} == {"full_batch": 20., "target_tracking": 40.}


def test_target_adaptive_only_still_loads_known_baselines(monkeypatch):
    def row(checkpoint, time):
        return dict(path_id="mixed", success=True, elapsed_time_s=time, checkpoint_directory=checkpoint,
                    reference_length_mm=10., constraints={"speed": {"limit": 100.}})
    checkpoint = dict(study="target_tracking", method="adaptive", status="complete", seed=101,
                      additional_decisions=256, checkpoint_directory="target256")
    known = [dict(study="target_tracking", method=method, baseline_id=name, successes=1, rows=[row("prior", time)])
             for method, name, time in (("fixed3", "sensor_original_fixed3", 5.), ("adaptive", "full_batch_selected_adaptive", 4.))]
    monkeypatch.setattr(report, "retained_target_development_baselines", lambda paths: known)
    baselines, comparisons = report.retained_initializer_results([checkpoint], [row("target256", 3.)], {"mixed"})
    assert baselines == known and len(comparisons) == 2
    assert {r["baseline_method"] for r in comparisons} == {"fixed3", "adaptive"}
    assert all("fixed_successes" not in r for r in comparisons)
    assert all(r["baseline_successes"] == 1 and r["full_suite_success"] for r in comparisons)


def test_feed_diagnostics_exclude_terminal_and_ignore_old_latent(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(report, "INPUTS", {})
    path = tmp_path/"trace.npz"
    columns = ["timestamp_s", "feed_logit", "feed_executed_action", "macro_decision", "terminal_braking", "v_mm_s", "latent_1"]
    data = np.array([[0, np.nan, np.nan, 0, np.nan, 0, -999],
                     [.001, -3, 1/(1+np.exp(3)), 1, np.nan, 1, -999],
                     [.002, -3, 1/(1+np.exp(3)), 0, np.nan, 1, -999],
                     [.003, 20, 0, np.nan, 1, 0, -999],
                     [.004, 20, 0, np.nan, 1, 0, -999]])
    np.savez(path, columns=columns, data=data)
    result = report.trace_diagnostics(path, digest(path))
    assert result["policy_command_samples"] == 2 and result["macro_decisions_observed"] == 1
    assert result["executed_zero_actions"] == 0 and result["negative_feed_logits"] == 2
    assert result["terminal_tail_s"] == pytest.approx(.002)


def test_existing_output_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "REPORTS", tmp_path)
    output = tmp_path/"existing"
    output.mkdir(); (output/"keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        report.build(["full_batch"], output, 256)
    assert (output/"keep.txt").read_text(encoding="utf-8") == "keep"
