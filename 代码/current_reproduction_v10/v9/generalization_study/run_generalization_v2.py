"""Registered serial replay of the five previously observed complete programs."""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OLD = Path("D:/PPO_article_work/JNN_dominance_20261011_v5")
sys.path[:0] = [str(ROOT / "src"), str(OLD / "src")]
from common import sha, write_json, utc

REG = HERE / "registration_v2.json"
PARENT = OLD / "evidence/frozen_main_generalization_v6.json"
LAUNCH = ROOT / "launch_study/registration.json"
SELECTION = ROOT / "launch_study/selection.json"
OMEGA = 60.0
WALL = 1800.0


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def register():
    if REG.exists():
        raise FileExistsError("Preserve existing registration")
    parent, launch, selection = read(PARENT), read(LAUNCH), read(SELECTION)
    assert selection["registration_sha256"] == sha(LAUNCH)
    assert selection["adoptable"] and selection["selected_omega"] == OMEGA
    assert parent["model_sha256"] == launch["model_sha256"]["JNN"]
    dependencies = {str(OLD / "src" / relative): digest for relative, digest in parent["source_sha256"].items()}
    dependencies.update(launch["source_files"])
    dependencies[str(Path(__file__).resolve())] = sha(__file__)
    cases = parent["cases"]
    grid = []
    for case in cases:
        for dt in (.001, .0005):
            name = f"{case['id']}_JNN_launch60_dt{round(dt * 1e6)}us"
            grid.append(dict(run_id=name, case=case["id"], method="JNN", dt_s=dt,
                trace=str(HERE / "results" / (name + ".npz")), record=str(HERE / "results" / (name + ".json")),
                physical_tick_budget=int(round(case["physical_time_budget_s"] / dt))))
    write_json(REG, dict(created_utc=utc(), study_id="v9_launch60_observed_five_program_replay",
        scope="All five previously observed synthetic complete programs at both command periods; this is a new-version re-evaluation, not a newly independent holdout.",
        parent_registration=str(PARENT), parent_registration_sha256=sha(PARENT),
        launch_registration=str(LAUNCH), launch_registration_sha256=sha(LAUNCH),
        selection=str(SELECTION), selection_sha256=sha(SELECTION), omega=OMEGA,
        selection_uses_this_study=False, model=str(OLD / parent["model"]), model_sha256=parent["model_sha256"],
        training_seed=923202, new_training_updates=0, model_updated=False,
        executor="src/launch_tracking.py::install", source_sha256=dependencies,
        runtime_manifest_sha256=sha(OLD / "runtime/source_manifest.json"),
        planar_vector_limits=parent["planar_vector_limits"], nominal_tolerance_mm=parent["nominal_tolerance_mm"],
        completion=parent["completion"], arc_handling=parent["arc_handling"],
        wall_limit_per_run_s=WALL, initial_preflight="Finite14-element observation, finite3-element actor output; exact initial position, zero velocity and acceleration before any command.",
        execution="One worker process at a time, all ten registered cells; no truncation, reordering, retuning or outcome-based case selection.",
        statistical_unit="One complete fixed-policy execution per synthetic program/resolution; resolutions are not training seeds.",
        cases=cases, grid=grid))
    verify()
    print(json.dumps(dict(registration=str(REG), sha256=sha(REG), cells=len(grid))), flush=True)


def verify():
    reg = read(REG)
    for path_key in ("parent_registration", "launch_registration", "selection"):
        assert sha(reg[path_key]) == reg[path_key + "_sha256"], path_key
    assert sha(reg["model"]) == reg["model_sha256"]
    for path, digest in reg["source_sha256"].items():
        assert sha(path) == digest, path
    assert sha(OLD / "runtime/source_manifest.json") == reg["runtime_manifest_sha256"]
    for item in read(OLD / "runtime/source_manifest.json"):
        assert sha(OLD / "runtime" / item["relative"]) == item["sha256"]
    for case in reg["cases"]:
        for key in ("reference", "geometry"):
            assert sha(OLD / case[key]) == case[key + "_sha256"]
    return reg


def worker(run_id):
    reg = verify()
    cell = next(row for row in reg["grid"] if row["run_id"] == run_id)
    case = next(row for row in reg["cases"] if row["id"] == cell["case"])
    output = Path(cell["trace"])
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("Never overwrite a run or partial trace")
    output.parent.mkdir(parents=True, exist_ok=True)
    cpu_started, wall_started = time.process_time(), time.perf_counter()
    reference = np.loadtxt(OLD / case["reference"], delimiter=",", skiprows=1)
    import torch
    torch.set_num_threads(1)
    from frozen_policy import FrozenPolicy
    from launch_tracking import install
    import evaluate
    from metrics_exact_index_v5 import metrics
    evaluate.metrics = metrics
    settings = install(omega=OMEGA, dt=cell["dt_s"], tolerance=case["evaluation_tolerance_mm"])
    policy = FrozenPolicy(reg["model"])
    class Actor:
        def act(self, observation, deterministic=True):
            return policy.mean(observation).numpy(), None, None
    from compact_farvector_runtime import FarVectorRuntime
    try:
        initial = FarVectorRuntime(reference, potential_speed=100.)
        observation = initial.observation()
        action = policy.mean(observation).numpy()
        assert observation.shape == (14,) and np.isfinite(observation).all()
        assert action.shape == (3,) and np.isfinite(action).all()
        assert np.array_equal(initial.env.position, reference[0])
        assert np.array_equal(initial.env.velocity, np.zeros(2))
        assert np.array_equal(initial.env.acceleration, np.zeros(2))
        preflight = dict(passed=True, observation_shape=list(observation.shape), action_shape=list(action.shape),
            observation=observation.tolist(), action=action.tolist(), initial_velocity_mm_s=initial.env.velocity.tolist(),
            initial_acceleration_mm_s2=initial.env.acceleration.tolist(), policy_commands_executed=0)
        del initial
    except Exception as exc:
        preflight = dict(passed=False, error=type(exc).__name__ + ": " + str(exc), traceback=traceback.format_exc(), policy_commands_executed=0)
    preflight_path = output.with_name(output.stem + "_preflight.json")
    write_json(preflight_path, preflight)
    if preflight["passed"]:
        result = evaluate.rollout(Actor(), reference, output, dt=cell["dt_s"], wall_s=WALL,
            tolerance=case["evaluation_tolerance_mm"], runtime_version="v3")
    else:
        np.savez_compressed(output, reference=reference, positions=reference[:1],
            observations=np.empty((0, 14)), actions=np.empty((0, 3)), rows=np.empty((0, 13)))
        result = dict(success=False, complete_time_s=None, elapsed_prefix_s=0., samples=1, dt_s=cell["dt_s"],
            reason="preflight_failure", error=preflight["error"], trace_sha256=sha(output))
    result.update(study_id=reg["study_id"], run_id=run_id, case=case["id"], method="JNN", omega=OMEGA,
        phase="observed_generalization", model_sha256=reg["model_sha256"], model_updated=False,
        new_training_updates=0, runtime_version=settings["version"], executor_setup=settings,
        registration_sha256=sha(REG), selection_sha256=reg["selection_sha256"],
        reference_sha256=case["reference_sha256"], geometry_sha256=case["geometry_sha256"],
        nominal_tolerance_mm=case["nominal_tolerance_mm"], arc_chord_error_bound_mm=case["arc_chord_error_bound_mm"],
        evaluation_tolerance_mm=case["evaluation_tolerance_mm"], physical_time_budget_s=case["physical_time_budget_s"],
        physical_tick_budget=cell["physical_tick_budget"], preflight=str(preflight_path), preflight_sha256=sha(preflight_path),
        process_cpu_s=time.process_time() - cpu_started, worker_wall_s=time.perf_counter() - wall_started,
        cost_scope="Worker setup, preflight, complete policy replay, trace serialization and independent audit; no real-time inference.",
        trace=str(output), trace_sha256=sha(output), origin="synthetic", actual_CAM=False,
        action_source="Same frozen c16 actor; selected finite launch60 execution law; no training.")
    write_json(output.with_suffix(".json"), result)
    print(json.dumps({key: result[key] for key in ("run_id", "success", "complete_time_s", "reason")}), flush=True)


def serial():
    reg = verify()
    for cell in reg["grid"]:
        record = Path(cell["record"])
        if record.exists():
            assert read(record)["registration_sha256"] == sha(REG)
            continue
        record.parent.mkdir(parents=True, exist_ok=True)
        log = record.with_suffix(".log")
        with log.open("x", encoding="utf-8") as stream:
            process = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "worker", "--run-id", cell["run_id"]],
                stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT, check=False)
        if process.returncode != 0:
            print(log.read_text(encoding="utf-8")[-6000:], flush=True)
            raise RuntimeError("Worker process failed; preserve log and any partial trace")
        result = read(record)
        print(json.dumps({key: result[key] for key in ("run_id", "success", "complete_time_s", "reason")}), flush=True)


def summarise():
    reg = verify()
    from metrics_exact_index_v5 import metrics
    rows, index = [], []
    for cell in reg["grid"]:
        result = read(cell["record"])
        case = next(row for row in reg["cases"] if row["id"] == cell["case"])
        assert result["registration_sha256"] == sha(REG) and sha(cell["trace"]) == result["trace_sha256"]
        with np.load(cell["trace"], allow_pickle=False) as archive:
            reference, positions = archive["reference"], archive["positions"]
        assert np.array_equal(reference, np.loadtxt(OLD / case["reference"], delimiter=",", skiprows=1))
        independent = metrics(reference, positions, dt=cell["dt_s"], tolerance=case["evaluation_tolerance_mm"],
            limits=reg["planar_vector_limits"], declared_success=result["success"], reason=result["reason"])
        assert independent["success"] == result["success"] and independent["complete_time_s"] == result["complete_time_s"]
        if result["success"]:
            assert len(positions) - 1 <= cell["physical_tick_budget"]
        old_cell = next(row for row in read(PARENT)["grid"] if row["case"] == case["id"] and row["dt_s"] == cell["dt_s"])
        old_result = read(OLD / old_cell["record"])
        rows.append(dict(**cell, record_sha256=sha(cell["record"]), trace_sha256=sha(cell["trace"]),
            metrics=independent, preflight=result["preflight"], preflight_sha256=result["preflight_sha256"],
            old_v6_success=old_result["success"], old_v6_time_s=old_result["complete_time_s"],
            old_record=str(OLD / old_cell["record"]), old_record_sha256=sha(OLD / old_cell["record"]),
            wall_s=result.get("worker_wall_s"), process_cpu_s=result.get("process_cpu_s")))
        index.append(dict(**cell, split="previously_observed_generalization", variant="v9_finite_launch_omega60", omega=OMEGA,
            formal_new_experiment=True, reference=str(OLD / case["reference"]), reference_sha256=case["reference_sha256"],
            nominal_tolerance_mm=case["nominal_tolerance_mm"], arc_chord_error_bound_mm=case["arc_chord_error_bound_mm"],
            physical_time_budget_s=case["physical_time_budget_s"]))
    write_json(HERE / "run_index.json", dict(schema_version=1, registration_sha256=sha(REG), runs=index))
    write_json(HERE / "summary.json", dict(registration_sha256=sha(REG), selection_sha256=reg["selection_sha256"],
        expected_cells=10, recorded_cells=len(rows), successful_cells=sum(row["metrics"]["success"] for row in rows),
        independent_full_interval_audit=True, new_training_updates=0, independent_holdout=False,
        scope=reg["scope"], rows=rows))
    print(json.dumps(dict(summary=str(HERE / "summary.json"), successful_cells=sum(row["metrics"]["success"] for row in rows), total_cells=len(rows))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "serial", "worker", "summary"))
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.command == "register": register()
    elif args.command == "serial": serial()
    elif args.command == "summary": summarise()
    else: worker(args.run_id)
