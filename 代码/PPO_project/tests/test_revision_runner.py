"""Runner provenance/selection tests, isolated from all real experimental data.

Uses synthetic geometries and temporary models only. Production controller
source and all real experimental artifacts are deliberately left unchanged.
"""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import revision_experiment as runner


class RevisionRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ppo_runner_test_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.data = self.directory / "data"
        self.data.mkdir()
        self.manifest = {"paths": [{"id": f"train_{i}", "split": "train"} for i in range(3)]
                         + [{"id": "validation_probe", "split": "validation"},
                            {"id": "test_guard_only", "split": "test"}]}
        runner.write_json(self.data / "manifest.json", self.manifest)
        self.config = runner.config_for("decoupled", 7)
        self.production_config = copy.deepcopy(self.config)
        self.production_manifest = runner.load_manifest()
        self.config["environment"]["max_steps"] = 4
        self.config["revision"]["training_path_ids"] = ["train_0", "train_1", "train_2"]
        self.config["ppo"].update(hidden_dim=32, epochs=1)
        self.source = {"source_sha256": "isolated-test-source", "source_files": {}}
        replacements = {
            "OUT": self.directory / "experiments",
            "DEFAULT_DATA_DIR": self.data,
            "source_state": lambda: copy.deepcopy(self.source),
            "config_for": lambda _variant, _seed: copy.deepcopy(self.config),
            "load_manifest": lambda: copy.deepcopy(self.manifest),
            "load_path": lambda _path_id: np.array([[0., 0.], [100., 0.]]),
        }
        for attribute, value in replacements.items():
            patcher = mock.patch.object(runner, attribute, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.stdout = contextlib.redirect_stdout(io.StringIO())
        self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)

    def train_probe(self, steps=9):
        return runner.train("decoupled", 7, "isolated", steps)

    def provenance_run(self):
        run = runner.OUT / "isolated" / "decoupled_seed7"
        run.mkdir(parents=True)
        (run / "config.yaml").write_text(yaml.safe_dump(self.config), encoding="utf-8")
        (run / "checkpoint.pt").write_bytes(b"test gate fixture; never loaded as a real model")
        runner.write_json(run / "provenance.json", {
            "source_sha256": self.source["source_sha256"],
            "source_files": {},
            "config_sha256": runner.sha(run / "config.yaml"),
            "path_manifest_sha256": runner.sha(self.data / "manifest.json"),
        })
        return run

    def frozen_gate(self, run):
        key = str(run.relative_to(runner.OUT)).replace("\\", "/")
        freeze = dict(models={key: runner.sha(run / "checkpoint.pt")},
                      configs={key: runner.sha(run / "config.yaml")},
                      source_sha256=self.source["source_sha256"],
                      path_manifest_sha256=runner.sha(self.data / "manifest.json"))
        runner.write_json(runner.OUT / "freeze.json", freeze)

    def assert_rejected_before_test_geometry(self, run, expression):
        with mock.patch.object(runner, "load_path", side_effect=AssertionError("Test geometry must not be read")):
            with self.assertRaisesRegex(ValueError, expression):
                runner.evaluate(run, "test")

    def test_train_rotates_across_all_three_paths(self):
        run = self.train_probe(9)
        rows = [json.loads(line) for line in (run / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["path_id"] for row in rows], ["train_0", "train_1", "train_2"])
        self.assertEqual([row["episode_steps"] for row in rows], [4, 4, 1])
        self.assertEqual([row["done_reason"] for row in rows[:2]], ["max_steps", "max_steps"])
        self.assertTrue(rows[-1]["budget_truncation"])
        checkpoint = torch.load(run / "checkpoint.pt", weights_only=False)
        self.assertEqual(checkpoint["steps"], 9)

    def test_production_training_budget_covers_only_all_three_training_paths(self):
        ids = self.production_config["revision"]["training_path_ids"]
        designated = [row["id"] for row in self.production_manifest["paths"] if row["split"] == "train"]
        self.assertEqual(ids, designated)
        self.assertEqual(len(ids), 3)
        self.assertTrue(self.production_config["reward_weights"]["boundary"]["enabled"])
        self.assertLessEqual(len(ids) * self.production_config["environment"]["max_steps"], 4096)

    def test_resume_same_budget_is_file_idempotent(self):
        run = self.train_probe(9)
        before = {str(path.relative_to(run)): path.read_bytes() for path in run.rglob("*") if path.is_file()}
        result = self.train_probe(9)
        after = {str(path.relative_to(run)): path.read_bytes() for path in run.rglob("*") if path.is_file()}
        self.assertEqual(result, run)
        self.assertEqual(before, after)

    def test_resume_rejects_changed_configuration(self):
        self.train_probe(4)
        self.config["environment"]["epsilon"] = 1.0
        with self.assertRaisesRegex(ValueError, "changed configuration"):
            self.train_probe(8)

    def test_resume_rejects_changed_source(self):
        self.train_probe(4)
        self.source["source_sha256"] = "different-source"
        with self.assertRaisesRegex(ValueError, "Source changed"):
            self.train_probe(8)

    def test_evaluation_rejects_training_source_mismatch(self):
        run = self.provenance_run()
        self.source["source_sha256"] = "different-source"
        with self.assertRaisesRegex(ValueError, "Source changed"):
            runner.evaluate(run, "validation")

    def test_evaluation_rejects_results_from_old_checkpoint(self):
        run = self.provenance_run()
        old = run / "evaluation_validation_nominal" / "validation_probe.json"
        payload = dict(checkpoint_sha256="before-resume-checkpoint", config_sha256=runner.sha(run / "config.yaml"),
                       code_sha256=self.source["source_sha256"], success=False)
        runner.write_json(old, payload)
        before = old.read_bytes()
        with self.assertRaisesRegex(ValueError, "Existing result belongs to another"):
            runner.evaluate(run, "validation")
        self.assertEqual(before, old.read_bytes())

    def test_test_gate_rejects_missing_freeze(self):
        run = self.provenance_run()
        with mock.patch.object(runner, "load_path", side_effect=AssertionError("Test geometry must not be read")):
            with self.assertRaises(FileNotFoundError):
                runner.evaluate(run, "test")

    def test_test_gate_rejects_model_mutation(self):
        run = self.provenance_run()
        self.frozen_gate(run)
        (run / "checkpoint.pt").write_bytes(b"changed")
        self.assert_rejected_before_test_geometry(run, "not the frozen test checkpoint")

    def test_test_gate_rejects_configuration_mutation(self):
        run = self.provenance_run()
        self.frozen_gate(run)
        with (run / "config.yaml").open("a", encoding="utf-8") as handle:
            handle.write("\n# A changed configuration artifact\n")
        self.assert_rejected_before_test_geometry(run, "configuration.*modified|configuration/source changed")

    def test_test_gate_rejects_manifest_mutation(self):
        run = self.provenance_run()
        self.frozen_gate(run)
        (self.data / "manifest.json").write_text('{"changed": true}', encoding="utf-8")
        self.assert_rejected_before_test_geometry(run, "path manifest changed")

    def freeze_fixture(self, budget=8192):
        scores = {"coupled": .3, "decoupled": .5, "fixed_short": .1, "fixed_mid": .8, "fixed_long": .4}
        for variant, progress in scores.items():
            summary = runner.OUT / "screen" / f"{variant}_seed101" / "evaluation_validation_nominal" / "summary.json"
            runner.write_json(summary, {"rows": [dict(success=False, path_completed=False, ordered_progress=progress, elapsed_time_s=1.)]})
        for variant in ["coupled", "decoupled", "fixed_mid"]:
            for seed in [11, 22, 33, 44, 55]:
                run = runner.OUT / "main" / f"{variant}_seed{seed}"
                run.mkdir(parents=True)
                torch.save({"steps": budget}, run / "checkpoint.pt")
                (run / "config.yaml").write_text("test_fixture: true\n", encoding="utf-8")
        runner.write_json(runner.OUT / "protocol.json", {"fixture": True})

    def test_freeze_selects_actual_ordered_progress_and_blocks_overwrite(self):
        self.freeze_fixture()
        runner.freeze()
        frozen = json.loads((runner.OUT / "freeze.json").read_text(encoding="utf-8"))
        self.assertEqual(frozen["selected_fixed"], "fixed_mid")
        self.assertEqual(len(frozen["models"]), 15)
        self.assertEqual(len(frozen["configs"]), 15)
        with self.assertRaises(FileExistsError):
            runner.freeze()

    def test_freeze_rejects_incomplete_main_budget(self):
        self.freeze_fixture(budget=8191)
        with self.assertRaisesRegex(ValueError, "8192-step budget"):
            runner.freeze()
        self.assertFalse((runner.OUT / "freeze.json").exists())

    def short_evaluation(self, run, distance):
        original = runner.make_env
        traces = []

        def short_env(*args, **kwargs):
            env, obs = original(*args, **kwargs)
            env.max_steps = 3  # Only the evaluator's wiring is under test.
            return env, obs

        metrics = dict(success=False, path_completed=False, ordered_progress=0., elapsed_time_s=.003,
                       failure_reasons=["isolated_test_probe"])
        with mock.patch.object(runner, "make_env", side_effect=short_env), \
             mock.patch.object(runner, "save_trace", side_effect=lambda _path, rows: traces.append(copy.deepcopy(rows))), \
             mock.patch.object(runner, "evaluate_trajectory", return_value=metrics):
            result = runner.evaluate(run, "validation", diagnostic_fixed=distance)
        return result, traces

    def test_fixed_output_diagnostic_really_executes_requested_distance(self):
        run = self.train_probe(4)
        rows, traces = self.short_evaluation(run, 2.3)
        self.assertEqual(len(traces), 1)
        self.assertGreater(len(traces[0]), 1)
        np.testing.assert_allclose([row["lookahead_mm"] for row in traces[0]], 2.3)
        self.assertEqual(rows[0]["comparison_role"], "output replacement only; not independent fixed training")
        effective = run / "evaluation_validation_nominal_fixed_2.3_diagnostic" / "effective_config.yaml"
        cfg = yaml.safe_load(effective.read_text(encoding="utf-8"))
        self.assertFalse(cfg["reward_weights"]["lookahead_control"]["enabled"])
        self.assertEqual(cfg["reward_weights"]["lookahead"]["distance"], 2.3)
        self.assertEqual(rows[0]["effective_config_sha256"], runner.sha(effective))
        self.assertEqual(rows[0]["evaluation_source_sha256"], self.source["source_sha256"])

    def test_different_diagnostic_distance_must_not_reuse_cached_result(self):
        run = self.train_probe(4)
        self.short_evaluation(run, 1.2)
        old_path = run / "evaluation_validation_nominal_fixed_1.2_diagnostic" / "validation_probe.json"
        old_bytes = old_path.read_bytes()
        _, traces = self.short_evaluation(run, 4.0)
        self.assertEqual(len(traces), 1)
        np.testing.assert_allclose([row["lookahead_mm"] for row in traces[0]], 4.0)
        self.assertEqual(old_bytes, old_path.read_bytes())
        self.assertTrue((run / "evaluation_validation_nominal_fixed_4_diagnostic" / "validation_probe.json").exists())

    def test_validation_must_reject_modified_training_configuration(self):
        run = self.train_probe(4)
        config_path = run / "config.yaml"
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        cfg["environment"]["epsilon"] = 1.0
        config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.short_evaluation(run, 2.3)

    def test_resume_must_reject_changed_path_manifest(self):
        self.train_probe(4)
        (self.data / "manifest.json").write_text('{"changed": true}', encoding="utf-8")
        with self.assertRaises(ValueError):
            self.train_probe(8)

    def test_validation_rejects_changed_training_path_manifest(self):
        run = self.provenance_run()
        (self.data / "manifest.json").write_text('{"changed": true}', encoding="utf-8")
        with mock.patch.object(runner, "load_path", side_effect=AssertionError("No geometry may be read after a provenance mismatch")):
            with self.assertRaisesRegex(ValueError, "path manifest changed"):
                runner.evaluate(run, "validation")

    def test_existing_effective_configuration_cannot_be_overwritten(self):
        run = self.train_probe(4)
        self.short_evaluation(run, 2.3)
        effective = run / "evaluation_validation_nominal_fixed_2.3_diagnostic" / "effective_config.yaml"
        cfg = yaml.safe_load(effective.read_text(encoding="utf-8"))
        cfg["reward_weights"]["lookahead"]["distance"] = 4.
        effective.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        before = effective.read_bytes()
        with self.assertRaisesRegex(ValueError, "different effective configuration"):
            self.short_evaluation(run, 2.3)
        self.assertEqual(before, effective.read_bytes())

    def test_precise_guard_upgrade_accepts_only_unchanged_runtime_files(self):
        script = "PPO_project/tools/revision_experiment.py"
        runtime = "PPO_project/src/environment/cnc_env.py"
        old = dict(source_sha256="old-aggregate", source_files={script: runner.GUARD_ONLY_PREDECESSOR, runtime: "runtime-hash"})
        new = dict(source_sha256="new-aggregate", source_files={script: "new-guard-runner-hash", runtime: "runtime-hash"})
        runner.check_source_compatibility(old, new)
        for changed in [
            {script: "new-guard-runner-hash", runtime: "modified-runtime"},
            {script: "new-guard-runner-hash", runtime: "runtime-hash", "new_runtime.py": "added"},
        ]:
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "Source changed"):
                runner.check_source_compatibility(old, dict(source_sha256="changed", source_files=changed))
        old["source_files"][script] = "unrecognized-older-runner"
        with self.assertRaisesRegex(ValueError, "Source changed"):
            runner.check_source_compatibility(old, new)


if __name__ == "__main__":
    unittest.main()
