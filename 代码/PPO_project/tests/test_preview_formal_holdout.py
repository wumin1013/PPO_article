"""Synthetic ten-model gate: no real policy or held-out geometry is accessed."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
import preview_formal_holdout as module


def put_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def artifact(root, path):
    return dict(file=path.relative_to(root).as_posix(), sha256=module.sha(path))


def source(root, names):
    files = {}
    for name in names:
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# Synthetic source: {name}\n", encoding="utf-8")
        files[name] = module.sha(path)
    return dict(source_files=files, source_sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest())


@pytest.fixture
def frozen_study(tmp_path):
    root = tmp_path
    formal = root/"论文项目/revision_20260910_continued/formal_v1"
    formal.mkdir(parents=True)
    pre_source = source(root, ["PPO_project/src/pre_fixture.py"])
    continuation_source = source(root, ["PPO_project/src/continuation_fixture.py"])
    evaluator_source = source(root, ["PPO_project/tools/preview_formal_holdout.py", "PPO_project/tools/preview_formal_statistics.py"])
    holdout = root/"PPO_project/data/preview_continuation_v1/manifest.json"
    paths = [dict(id=f"{family}_{i}", family=family, split="test") for family in ("mixed_lengths", "dense_corners", "line_curve") for i in (1, 2)]
    put_json(holdout, dict(paths=paths))
    protocol_path = formal/"protocol.json"
    protocol = dict(seeds=list(module.SEEDS), methods=list(module.METHODS), sources=dict(pretraining=pre_source, continuation=continuation_source),
                    evaluation_sources=evaluator_source, holdout_manifest=artifact(root, holdout), validation_path_ids=["val_mixed", "val_dense", "val_curve"])
    put_json(protocol_path, protocol)

    def make_files(directory, cfg, metadata, state, payload):
        directory.mkdir(parents=True, exist_ok=True)
        (directory/"checkpoint.pt").write_bytes(payload)
        (directory/"config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        put_json(directory/"provenance.json", dict(**state, config_sha256=module.sha(directory/"config.yaml")))
        with zipfile.ZipFile(directory/"source.zip", "w") as archive:
            for name in state["source_files"]:
                archive.write(root/name, name)
        put_json(directory/"training_summary.json", dict(**metadata, checkpoint_sha256=module.sha(directory/"checkpoint.pt")))

    records = []
    for seed in module.SEEDS:
        for method in module.METHODS:
            prior = formal/"pretraining"/f"{method}_seed{seed}"
            make_files(prior, dict(seed=seed, revision={}), dict(seed=seed, method=method, decisions=4096, pretraining_steps=0), pre_source, f"prior-{seed}-{method}".encode())
            cfg = dict(seed=seed, environment=dict(epsilon=2.5, interpolation_period=.001, max_steps=60000),
                kinematic_constraints=dict(MAX_VEL=100.), revision=dict(adaptive_preview=method == "adaptive", macro_period_steps=20,
                fixed_distance_mm=3., initial_preview_mm=3., active_action_dims=[1, 2] if method == "adaptive" else [1],
                terminal_definition="Original v2; complete rest", init_checkpoint=(prior/"checkpoint.pt").relative_to(root).as_posix(),
                init_checkpoint_sha256=module.sha(prior/"checkpoint.pt")))
            candidates = []
            for step in module.STEPS:
                directory = formal/"continuation"/f"{method}_seed{seed}"/"checkpoints"/f"step{step}"
                make_files(directory, cfg, dict(seed=seed, method=method, decisions=step, optimized_decisions=step, pending_buffer_decisions=0), continuation_source, f"actor-{seed}-{method}-{step}".encode())
                for name in ("episodes.jsonl", "updates.jsonl", "training_final_partial.npz"):
                    (directory/name).write_bytes(b"synthetic immutable log")
                put_json(directory/"snapshot_manifest.json", dict(decisions=step, files={p.name: module.sha(p) for p in directory.iterdir() if p.is_file()}))
                rows = [dict(seed=seed, method=method, path_id=path, checkpoint_sha256=module.sha(directory/"checkpoint.pt"), success=True,
                             elapsed_time_s=9. if step == 256 else 10.) for path in protocol["validation_path_ids"]]
                put_json(directory/"validation/summary.json", dict(rows=rows))
                candidates.append(dict(step=step, successes=3, count=3, complete_validation_time_s=sum(r["elapsed_time_s"] for r in rows),
                    checkpoint_dir=directory.relative_to(root).as_posix(), checkpoint_sha256=module.sha(directory/"checkpoint.pt"),
                    validation_sha256=module.sha(directory/"validation/summary.json"), training_summary_sha256=module.sha(directory/"training_summary.json"),
                    snapshot_manifest_sha256=module.sha(directory/"snapshot_manifest.json")))
            selected = next(c for c in candidates if c["step"] == 256)
            chosen = root/selected["checkpoint_dir"]
            selection_path = formal/"selections"/f"{method}_seed{seed}.json"
            put_json(selection_path, dict(seed=seed, method=method, selected=selected, all_checkpoints=candidates, step0_eligible=True))
            final = root/candidates[-1]["checkpoint_dir"]
            names = module.CORE_ARTIFACTS | {"snapshot_manifest.json", "validation/summary.json"}
            records.append(dict(seed=seed, method=method, selected_step=256, available_pretraining_decisions=4096, available_additional_decisions=4096,
                checkpoint_dir=chosen.relative_to(root).as_posix(), artifacts={n: artifact(root, chosen/n) for n in names},
                pretraining=dict(checkpoint_dir=prior.relative_to(root).as_posix(), decisions=4096, artifacts={n: artifact(root, prior/n) for n in module.CORE_ARTIFACTS}),
                selection=artifact(root, selection_path), additional_budget_completion=artifact(root, final/"training_summary.json"),
                additional_budget_snapshot_manifest=artifact(root, final/"snapshot_manifest.json"), source_sha256=continuation_source["source_sha256"]))
    freeze = formal/"freeze.json"
    put_json(freeze, dict(version="formal-preview-continuation-v1", protocol=artifact(root, protocol_path), holdout_manifest=artifact(root, holdout),
        seeds=list(module.SEEDS), methods=list(module.METHODS), records=records, source=continuation_source, all_training_complete=True))
    return root, freeze, holdout


def deny_holdout(monkeypatch, holdout):
    original = module.read_json
    def guarded(path):
        if Path(path) == holdout:
            raise AssertionError("Holdout manifest read before failed gate")
        return original(path)
    monkeypatch.setattr(module, "read_json", guarded)


def test_valid_gate_uses_separate_protocol_sources_and_does_not_read_holdout(frozen_study, monkeypatch):
    root, freeze, holdout = frozen_study
    deny_holdout(monkeypatch, holdout)
    gate = module.validate_freeze(freeze, root)
    assert len(gate["models"]) == 10


def test_reference_loader_only_runs_after_all_ten_models_pass(frozen_study):
    root, freeze, _ = frozen_study
    seen = []
    gate, manifest, references = module.load_frozen_references(freeze, root, lambda path, data: seen.append(path) or [[0., 0.], [1., 0.]])
    assert len(gate["models"]) == 10 and len(seen) == len(references) == len(manifest["paths"]) == 6


@pytest.mark.parametrize("fault", ["missing_freeze", "one_missing_model", "duplicate_model", "last_model_tampered", "evaluator_changed", "missing_last_budget", "snapshot_log_tampered", "missing_validation_opportunity", "selected_later_model", "wrong_selection_successes"])
def test_any_freeze_mismatch_refuses_holdout_before_loader(frozen_study, monkeypatch, fault):
    root, freeze, holdout = frozen_study
    data = module.read_json(freeze)
    record = data["records"][-1]
    if fault == "missing_freeze":
        freeze.unlink()
    elif fault == "one_missing_model":
        data["records"].pop(); put_json(freeze, data)
    elif fault == "duplicate_model":
        data["records"][-1] = data["records"][0]; put_json(freeze, data)
    elif fault == "last_model_tampered":
        (root/record["artifacts"]["checkpoint.pt"]["file"]).write_bytes(b"changed")
    elif fault == "evaluator_changed":
        (root/"PPO_project/tools/preview_formal_statistics.py").write_text("# changed", encoding="utf-8")
    elif fault == "missing_last_budget":
        (root/record["additional_budget_completion"]["file"]).unlink()
    elif fault == "snapshot_log_tampered":
        (root/record["checkpoint_dir"]/"episodes.jsonl").write_bytes(b"changed")
    else:
        path = root/record["selection"]["file"]
        selection = module.read_json(path)
        if fault == "missing_validation_opportunity": selection["all_checkpoints"].pop()
        elif fault == "selected_later_model": selection["selected"] = selection["all_checkpoints"][-1]
        else: selection["all_checkpoints"][0]["successes"] = 2
        put_json(path, selection); record["selection"] = artifact(root, path); put_json(freeze, data)
    deny_holdout(monkeypatch, holdout)
    def loader(*args):
        raise AssertionError("Test references must not be loaded")
    with pytest.raises((module.FreezeError, FileNotFoundError)):
        module.load_frozen_references(freeze, root, loader)


def test_terminal_tail_requires_contiguous_saved_suffix():
    trace = [dict(timestamp_s=0.), dict(timestamp_s=.001), dict(timestamp_s=.002, terminal_braking=1.), dict(timestamp_s=.003, terminal_braking=1.)]
    assert module.terminal_tail(trace) == pytest.approx(.002)
    trace[-1]["terminal_braking"] = 0.
    with pytest.raises(module.FreezeError): module.terminal_tail(trace)


def test_new_output_never_overwrites_previous_evidence(tmp_path):
    path = tmp_path/"result.json"
    module.write_json_new(path, dict(success=False))
    with pytest.raises(FileExistsError): module.write_json_new(path, dict(success=True))
    assert module.read_json(path) == dict(success=False)
