"""Synthetic filesystem audit; no real models, training or holdout are read."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
import formal_validation_summary as module


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_delivery(folder):
    binding = load(folder/"binding.json")
    put(folder/"delivery_manifest.json", dict(files={p.name: module.sha(p) for p in folder.iterdir()
        if p.is_file() and p.name != "delivery_manifest.json"},
        **{k: binding[k] for k in ("checkpoint_sha256", "config_sha256", "source_sha256")}))


@pytest.fixture
def study(tmp_path):
    root, formal = tmp_path, tmp_path/"study"
    source_file = root/"PPO_project/src/synthetic_fixture.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("# synthetic numerical source\n", encoding="utf-8")
    files = {source_file.relative_to(root).as_posix(): module.sha(source_file)}
    source = dict(source_files=files, source_sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest())
    development = root/"PPO_project/data/preview_event_v1/manifest.json"
    put(development, dict(paths=[]))
    paths = ["mixed_validation", "dense_validation", "curve_validation"]
    protocol = dict(version="formal-preview-continuation-v1", seeds=list(module.SEEDS), methods=list(module.METHODS),
        validation_milestones=list(module.STEPS), selection_milestones=list(module.STEPS),
        validation_path_ids=paths, pretraining_decisions=4096, additional_decisions=4096,
        selection_rule="success, all-three time, earliest", sources=dict(pretraining=source, continuation=source),
        training_validation_manifest=dict(file=development.relative_to(root).as_posix(), sha256=module.sha(development)))
    put(formal/"protocol.json", protocol)

    def core(folder, cfg, training, payload):
        folder.mkdir(parents=True, exist_ok=True)
        (folder/"checkpoint.pt").write_bytes(payload.encode())
        (folder/"config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        put(folder/"provenance.json", dict(**source, config_sha256=module.sha(folder/"config.yaml")))
        with zipfile.ZipFile(folder/"source.zip", "w") as archive:
            archive.write(source_file, source_file.relative_to(root).as_posix())
        put(folder/"training_summary.json", dict(**training, checkpoint_sha256=module.sha(folder/"checkpoint.pt")))

    for seed in module.SEEDS:
        for method in module.METHODS:
            prior = formal/f"pretraining/formal/preview_time_{method}_seed{seed}"
            core(prior, dict(seed=seed, revision={}), dict(seed=seed, method=method, decisions=4096, pretraining_steps=0), f"prior{method}{seed}")
            costs = dict(decisions=4096, training_wall_s=1.)
            put(prior/"costs.json", costs)
            cfg = dict(seed=seed, environment=dict(epsilon=2.5, interpolation_period=.001),
                kinematic_constraints=dict(MAX_VEL=100.), revision=dict(adaptive_preview=method == "adaptive",
                macro_period_steps=20, fixed_distance_mm=3., initial_preview_mm=3.,
                active_action_dims=[1, 2] if method == "adaptive" else [1], terminal_definition="same rest",
                init_checkpoint=(prior/"checkpoint.pt").relative_to(root).as_posix(),
                init_checkpoint_sha256=module.sha(prior/"checkpoint.pt")))
            candidates = []
            for step in module.STEPS:
                folder = module.checkpoint_dir(formal, method, seed, step)
                core(folder, cfg, dict(seed=seed, method=method, decisions=step, optimized_decisions=step,
                    pending_buffer_decisions=0), f"actor{seed}{method}{step}")
                for name in ("episodes.jsonl", "updates.jsonl", "training_final_partial.npz"):
                    (folder/name).write_bytes(b"synthetic log")
                put(folder/"snapshot_manifest.json", dict(decisions=step, files={p.name:module.sha(p) for p in folder.iterdir() if p.is_file()}))
                validation = folder/"validation"
                binding = dict(checkpoint_sha256=module.sha(folder/"checkpoint.pt"),
                    config_sha256=module.sha(folder/"config.yaml"), source_sha256=source["source_sha256"],
                    split="validation", manifest_sha256=module.sha(development))
                put(validation/"binding.json", binding)
                rows = []
                for path in paths:
                    trace = validation/f"{path}.npz"
                    trace.write_bytes(f"synthetic trace {seed} {method} {step} {path}".encode())
                    rows.append(dict(seed=seed, method=method, training_decisions=step, path_id=path,
                        success=True, failure_reasons=[], elapsed_time_s=8. if step == 256 else 9., evaluation_wall_s=.5,
                        checkpoint_sha256=binding["checkpoint_sha256"], config_sha256=binding["config_sha256"],
                        source_sha256=binding["source_sha256"], trace_sha256=module.sha(trace)))
                    put(validation/f"{path}.json", rows[-1])
                put(validation/"summary.json", dict(rows=rows))
                save_delivery(validation)
                candidates.append(dict(step=step, successes=3, count=3,
                    complete_validation_time_s=sum(r["elapsed_time_s"] for r in rows),
                    checkpoint_dir=folder.relative_to(root).as_posix(), checkpoint_sha256=binding["checkpoint_sha256"],
                    validation=dict(file=(validation/"summary.json").relative_to(root).as_posix(), sha256=module.sha(validation/"summary.json")),
                    validation_sha256=module.sha(validation/"summary.json"),
                    training_summary_sha256=module.sha(folder/"training_summary.json"),
                    snapshot_manifest_sha256=module.sha(folder/"snapshot_manifest.json"), validation_evaluation_wall_s=1.5))
            put(formal/f"selections/{method}_seed{seed}.json", dict(seed=seed, method=method, all_checkpoints=candidates,
                selected=candidates[1], available_pretraining_decisions=4096, available_additional_decisions=4096,
                step0_eligible=True, rule=protocol["selection_rule"], costs=dict(pretraining=costs,
                continuation=load(module.checkpoint_dir(formal, method, seed, 4096)/"training_summary.json"),
                validation_evaluation_wall_s=9.)))
    return root, formal


def test_complete_audit_writes_exact_180_and_30(study):
    root, formal = study
    result = module.build_report(formal, root/"report", root)
    assert result["methods"]["adaptive"]["successes"] == 15
    verification = load(root/"report/verification.json")
    assert verification["candidate_rows"] == 180 and verification["selected_rows"] == 30
    assert not verification["holdout_access"] and not verification["actor_execution"]
    assert not (root/"report/progress.json").exists()


def test_missing_selections_only_writes_progress(study):
    root, formal = study
    for method in module.METHODS:
        (formal/f"selections/{method}_seed55.json").unlink()
    result = module.build_report(formal, root/"progress", root)
    assert result["status"] == "incomplete" and result["selection_count"] == 8
    assert [p.name for p in (root/"progress").iterdir()] == ["progress.json"]
    assert not {"all_rows", "selected_rows", "methods", "common_success"}.intersection(result)


@pytest.mark.parametrize("name", ["checkpoint.pt", "config.yaml", "episodes.jsonl", "validation/mixed_validation.npz", "validation/mixed_validation.json", "validation/summary.json"])
def test_any_archived_artifact_change_refuses_report(study, name):
    root, formal = study
    folder = module.checkpoint_dir(formal, "fixed3", 11, 0)
    (folder/name).write_bytes(b"changed")
    with pytest.raises(ValueError):
        module.build_report(formal, root/"report", root)
    assert not (root/"report").exists()


def test_omitted_delivery_member_cannot_hide_missing_trace(study):
    root, formal = study
    folder = module.checkpoint_dir(formal, "fixed3", 11, 0)/"validation"
    trace = folder/"mixed_validation.npz"
    trace.unlink()
    delivery = load(folder/"delivery_manifest.json")
    del delivery["files"][trace.name]
    put(folder/"delivery_manifest.json", delivery)
    with pytest.raises(ValueError, match="Missing delivery_manifest"):
        module.collect(formal, root)


def test_raw_json_summary_semantic_mismatch_despite_fresh_delivery(study):
    root, formal = study
    folder = module.checkpoint_dir(formal, "fixed3", 11, 0)/"validation"
    row = load(folder/"mixed_validation.json")
    row["elapsed_time_s"] = 999.
    put(folder/"mixed_validation.json", row)
    save_delivery(folder)
    with pytest.raises(ValueError, match="Raw JSON differs"):
        module.collect(formal, root)


@pytest.mark.parametrize("field,value", [("seed", 999), ("method", "adaptive"), ("step0_eligible", False), ("available_additional_decisions", 0)])
def test_selection_identity_rule_or_budget_mismatch(study, field, value):
    root, formal = study
    path = formal/"selections/fixed3_seed11.json"
    selection = load(path)
    selection[field] = value
    put(path, selection)
    with pytest.raises(ValueError):
        module.collect(formal, root)


def test_faster_checkpoint_selection_is_recomputed(study):
    root, formal = study
    path = formal/"selections/fixed3_seed11.json"
    selection = load(path)
    selection["selected"] = selection["all_checkpoints"][0]
    put(path, selection)
    with pytest.raises(ValueError, match="Checkpoint selection"):
        module.collect(formal, root)


def test_existing_output_preserved_without_reading_study(tmp_path):
    output = tmp_path/"existing"
    output.mkdir()
    (output/"keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="Preserve previous report"):
        module.build_report(tmp_path/"missing", output, tmp_path)
    assert (output/"keep.txt").read_text() == "keep"


def test_holdout_is_never_permitted_by_audit(tmp_path):
    path = tmp_path/"PPO_project/data/preview_continuation_v1/manifest.json"
    put(path, dict(secret="never read"))
    with pytest.raises(ValueError, match="Holdout access"):
        module.Audit(tmp_path).file(path)


def test_changed_input_after_collection_is_detected(tmp_path):
    path = tmp_path/"input.json"
    put(path, dict(a=1))
    audit = module.Audit(tmp_path)
    audit.json(path)
    put(path, dict(a=2))
    with pytest.raises(ValueError, match="Checksum mismatch"):
        audit.unchanged()
