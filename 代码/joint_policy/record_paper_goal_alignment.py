"""Record the user's goal correction; preserve all historical results and hashes."""
import hashlib
import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
STAGE = BASE / "paper_evidence_review/goal_alignment"
GUIDE = "PPO_article_研究目标_图表与写作指南.md"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    if STAGE.exists():
        raise RuntimeError("Alignment receipt already exists; do not overwrite or rerun.")
    now = datetime.now(timezone.utc).isoformat()
    expected = {
        "canonical_gate_band_worst_guard_joint_learning_20260920/shadow_training_policy.pt":
            "25abeb3b88e45efac1d8311669f45a3fcaa26dc785d9bfbd216734b7f5a9c173",
        "canonical_gate_band_shape_cruise_completion_feedback_20260918/candidate_0.5.pt":
            "86cdfe3e1bf550a34332bab2678fed9a12e3a33611173ba57a75a1818daf5a86",
        "论文研究边界与续接指令_20260920.md":
            "2c10f01c27fa7a9f3531b3b3a2bea0162fdba0d3325f5a844fa5bb788f0c8a20",
        "sealed_test_preparation_v1/paths.json":
            "8ef7d772056373a68e9f3d74ff7d539ebef34a295ad165fc1fb0d92e23ad929a",
    }
    checked = []

    def check(rel, digest):
        actual = sha(BASE / rel)
        checked.append({"path": rel, "expected_sha256": digest,
                        "actual_sha256": actual, "pass": actual == digest})
        if actual != digest:
            raise RuntimeError(f"Hash mismatch: {rel}")

    for rel, digest in expected.items():
        check(rel, digest)
    for folder in ["remaining_public_long", "nonlocal_state_diagnostic", "support_repair", "bend_residual_learning"]:
        prefix = f"paper_evidence_review/{folder}"
        item = read(BASE / prefix / "receipt.json")
        check(prefix + "/results.json", item["results_sha256"])
        check(prefix + "/registration.json", item["registration_sha256"])
    prefix = "paper_evidence_review/public_long"
    item = read(BASE / prefix / "receipt.json")
    for filename, key in [("evaluation.json", "evaluation_sha256"),
                          ("trace.npz", "trace_sha256"),
                          ("registration.json", "registration_sha256")]:
        check(prefix + "/" + filename, item[key])
    manifest = read(BASE / "PPO_article_manuscript/figure_sources.json")
    for case in manifest["cases"]:
        check(case["trace"], case["trace_sha256"])

    # Process identity is checked freshly. Merely inspecting all Python metadata
    # does not operate on any other project. Exclude this bookkeeping process.
    command = (
        "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); "
        "@(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\\.exe$' } | "
        "Select-Object ProcessId,ParentProcessId,Name,CreationDate,CommandLine) | ConvertTo-Json -Depth 4 -Compress"
    )
    raw = subprocess.run(["powershell", "-NoProfile", "-Command", command],
                         check=True, capture_output=True).stdout.decode("utf-8-sig").strip()
    processes = json.loads(raw) if raw else []
    if isinstance(processes, dict):
        processes = [processes]
    names = ("evaluate_paper_confirmatory.py", "train_paper_bend_residual.py",
             "evaluate_paper_support_repair.py", "complete_public_long_review.py",
             "draw_paper_complete_cases.py")
    active = [p for p in processes if p["ProcessId"] != os.getpid() and
              any(n in (p.get("CommandLine") or "") for n in names)]
    if active:
        raise RuntimeError("A relevant process remains active; inspect before recording a stop.")

    confirm = BASE / "paper_evidence_review/confirmatory"
    if (confirm / "receipt.json").exists():
        raise RuntimeError("A final confirmatory receipt appeared; reconcile before writing interruption.")
    partial = read(confirm / "results.json")["results"]
    counts = dict(Counter(r["split"] for r in partial))
    exposed_ids = sorted({r["case"] for r in partial if r["split"] == "sealed"})
    case_metadata = read(confirm / "evaluation_cases.json")
    opened_ids = [c["id"] for c in case_metadata if c.get("split") == "sealed"]
    if len(opened_ids) != 30:
        raise RuntimeError("Unexpected exposed-case metadata; inspect its schema.")
    referenced = {str((BASE / r["trace"]).resolve()).casefold()
                  for r in partial if r.get("trace")}
    extra_traces = [str(p.relative_to(BASE)) for p in confirm.rglob("*.npz")
                    if str(p.resolve()).casefold() not in referenced]

    STAGE.mkdir()
    state_path = BASE / "当前执行状态.json"
    (STAGE / "execution_state_before.json").write_bytes(state_path.read_bytes())
    write(STAGE / "hash_verification.json", {"verified_utc": now, "checks": checked,
                                            "all_pass": True})
    write(STAGE / "process_snapshot.json", {
        "checked_utc": now, "metadata": processes, "bookkeeping_pid": os.getpid(),
        "relevant_experiment_processes": active,
        "scope": "Fresh Python-process metadata; no process started or stopped by this script."})
    interruption = {
        "recorded_utc": now,
        "status": "interrupted_on_user_goal_correction_not_confirmatory_completion",
        "reason": "User requires goal-demonstrating complete experiments before freezing; public long success remains 0/3.",
        "termination_history": {"pid": 29844, "script": "evaluate_paper_confirmatory.py",
                                "source": "Earlier identity-checked stop in this task; this script verifies absence only."},
        "saved_evaluation_records": len(partial), "saved_records_by_split": counts,
        "sealed_case_ids_with_saved_records": exposed_ids,
        "sealed_coordinates_opened_count": len(opened_ids),
        "sealed_coordinates_opened_case_ids": opened_ids,
        "sealed_set_remains_unseen": False,
        "sealed_results_allowed_for_tuning": False,
        "partial_results_are_full_set_evidence": False,
        "unlisted_trace_files_preserved_without_outcome_analysis": extra_traces,
        "results_sha256": sha(confirm / "results.json"),
        "evaluation_cases_sha256": sha(confirm / "evaluation_cases.json"),
        "method_freeze_sha256": sha(confirm / "method_freeze.json"),
        "sealed_open_receipt_sha256": sha(confirm / "sealed_open_receipt.json"),
        "method_freeze_disposition": "Historical code/model snapshot; not accepted as goal attainment or current instruction to continue evaluation.",
        "automatic_resume_allowed": False, "goal_complete": False,
        "formal_model_replaced": False,
    }
    write(confirm / "interruption_receipt.json", interruption)
    next_action = (
        "Deliver the original-goal, figure and manuscript guide and complete-case evidence. "
        "Goal remains unmet: no current-candidate successful public long path. "
        "No automatic confirmatory resume, sealed-result tuning or third algorithm round. "
        "Prepare any further repair hypothesis using only training/public-development evidence; "
        "an extension beyond the used 2/2 algorithm rounds requires a concrete bounded proposal and user confirmation."
    )
    continuation = {
        "created_utc": now, "status": "goal_unmet_confirmatory_interrupted_guide_delivered",
        "guide": GUIDE, "guide_sha256": sha(BASE / GUIDE),
        "algorithm_rounds_used": 2, "algorithm_rounds_cap": 2,
        "round_wall_s": {"support_repair": 15.249676600000384, "bend_residual_learning": 526.9340749000003},
        "public_long_successes": 0, "public_long_count": 3,
        "short_complete_illustrative_cases": 7, "failed_long_illustrative_cases": 1,
        "independent_from_scratch_ppo_seeds": 0,
        "sealed_test_opened": True, "confirmatory_evaluation_complete": False,
        "interruption_receipt": "paper_evidence_review/confirmatory/interruption_receipt.json",
        "interruption_receipt_sha256": sha(confirm / "interruption_receipt.json"),
        "formal_adoption": False, "goal_complete": False,
        "user_priority": "Demonstrate the original experimental goal before method freeze and manuscript-ready claims.",
        "current_work": "Record correction, existing-evidence review, and manuscript planning only.",
        "next_action": next_action,
        "this_recording_step_new_actor_calls": 0, "this_recording_step_new_physics_ticks": 0,
        "this_recording_step_training_updates": 0,
    }
    write(STAGE / "continuation.json", continuation)
    sources = ["paper_evidence_review/remaining_public_long/receipt.json",
               "paper_evidence_review/nonlocal_state_diagnostic/receipt.json",
               "paper_evidence_review/support_repair/receipt.json",
               "paper_evidence_review/bend_residual_learning/receipt.json",
               "paper_evidence_review/followthrough/receipt.json",
               "paper_evidence_review/论文完成执行约定.md",
               "逐图保留与修改报告_进行中.md", "method_figure_contract_v1.md",
               "s2_figure_layer_contract.md", "PPO_article_manuscript/figure_sources.json",
               "record_paper_goal_alignment.py"]
    outputs = [GUIDE, "paper_evidence_review/confirmatory/interruption_receipt.json"] + [
        str(p.relative_to(BASE)) for p in STAGE.iterdir() if p.is_file()]
    receipt = {
        "created_utc": now, "status": "completed_record_correction_and_writing_guide_not_research_goal",
        "sources_sha256": {r: sha(BASE / r) for r in sources},
        "outputs_sha256": {r: sha(BASE / r) for r in outputs},
        "hash_checks": len(checked), "goal_complete": False,
        "new_training_or_simulation_in_this_step": False,
        "no_relevant_experiment_process_running": True,
        "prior_hash_bound_records_preserved": True,
        "usage_snapshot": {"weekly_remaining_percent": 48, "reset_credits": 0,
                           "source": "Live Codex usage tool during this response", "reset_used": False},
    }
    write(STAGE / "receipt.json", receipt)
    state = read(state_path)
    state.update(updated_utc=now, goal_complete=False, original_objective_unchanged=True,
                 running=False, running_commands=[], observed_research_processes=[],
                 active_experiment=None, active_parallel_stages=[],
                 current_action="goal_unmet_existing_evidence_and_writing_guide",
                 current_experiment=None, next_action=next_action,
                 latest_execution_findings=continuation,
                 current="Goal unmet; confirmed 0/3 public long success; premature confirmatory evaluation interrupted.",
                 next=next_action)
    state["paper_goal_alignment"] = continuation
    resume = state["latest_resume_20260919"]
    resume.update(learning_status="no_learning_running_goal_unmet_confirmatory_interrupted",
                  next_action=next_action, goal_complete=False, latest_execution_utc=now,
                  no_research_session_left_running_by_this_turn=True)
    review = resume["paper_evidence_review"]
    review.update(algorithm_rounds_used=2, algorithm_rounds_cap=2,
                  no_new_training_or_simulation=False,
                  no_new_training_or_simulation_scope="Initial review only; two later rounds completed, current guide step adds none.",
                  next_action=next_action, formal_adoption=False,
                  public_long_successes=0, public_long_count=3,
                  sealed_test_opened=True, sealed_set_remains_unseen=False,
                  confirmatory_complete=False, automatic_confirmatory_resume_allowed=False,
                  latest_continuation={
                      "path": "paper_evidence_review/goal_alignment/continuation.json",
                      "sha256": sha(STAGE / "continuation.json"),
                      "receipt": "paper_evidence_review/goal_alignment/receipt.json",
                      "receipt_sha256": sha(STAGE / "receipt.json"),
                      "status": continuation["status"]})
    for boundary in [state["paper_research_boundary"], resume["paper_research_boundary"]]:
        boundary["latest_user_clarification"] = "Original experimental goal must be demonstrated before freeze; manuscript planning is not goal completion."
        boundary["goal_and_writing_guide"] = GUIDE
        boundary["guide_sha256"] = sha(BASE / GUIDE)
    write(state_path, state)
    write(STAGE / "state_update_receipt.json", {
        "created_utc": now, "state_before_sha256": sha(STAGE / "execution_state_before.json"),
        "state_after_sha256": sha(state_path), "continuation_receipt_sha256": sha(STAGE / "receipt.json"),
        "goal_complete": False, "sealed_test_opened": True, "algorithm_rounds_used": 2})
    print(json.dumps({"status": receipt["status"], "hash_checks": len(checked),
                      "saved_records": len(partial), "splits": counts,
                      "evaluated_sealed_cases": exposed_ids,
                      "extra_traces_preserved": len(extra_traces), "guide": GUIDE,
                      "experiment_processes": active}, ensure_ascii=False))


if __name__ == "__main__":
    main()
