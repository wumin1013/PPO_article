"""Apply the user's single-composite-case scope to the living guide and state."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
STAGE = BASE / "paper_evidence_review/single_case_scope"
GUIDE = BASE / "PPO_article_研究目标_图表与写作指南.md"
STATE = BASE / "当前执行状态.json"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def write(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    if (STAGE / "receipt.json").exists():
        raise RuntimeError("Scope revision already recorded; do not repeat it.")
    old_receipt = read(BASE / "paper_evidence_review/goal_alignment/receipt.json")
    old_guide_hash = old_receipt["outputs_sha256"][GUIDE.name]
    assert sha(STAGE / "guide_before.md") == old_guide_hash == sha(GUIDE)
    assert sha(STAGE / "execution_state_before.json") == sha(STATE)
    content = (STAGE / "guide.md").read_bytes()
    assert not content.startswith(b"\xef\xbb\xbf")
    assert "\ufffd" not in content.decode("utf-8")
    now = datetime.now(timezone.utc).isoformat()
    next_action = (
        "Follow the user's single-composite-case scope. First inspect already saved complete paths for long/short "
        "straights, obtuse/right/acute corners and a circular arc; reuse qualified evidence. If needed, audit and "
        "register one composite geometry and fixed comparison windows offline before any new evaluation. "
        "Obtain complete-case corner-speed, high-speed-cruise and completion-time evidence with credible baselines; "
        "three historical long paths need not all succeed. No automatic new optimization round or sealed-test resume."
    )
    scope = {
        "updated_utc": now,
        "status": "single_composite_case_research_and_writing_plan_revised",
        "user_instruction": "One successful complete composite case with long/short straights, obtuse/right/acute corners and an arc is sufficient for the central case-study demonstration; compare all three angle categories in the mechanism figure.",
        "required_primary_geometry_cases": 1,
        "required_geometry_features": ["long_straights", "short_straights", "obtuse_corner", "right_angle_corner", "acute_corner", "circular_arc"],
        "all_three_historical_long_paths_must_succeed": False,
        "claim_scope": "Measured advantage over named baselines on the specified complete composite case; universal superiority is not inferred.",
        "goals": ["less_corner_speed_reduction", "more_near_100mm_s_cruise", "lower_complete_path_time"],
        "hard_constraints_unchanged": True,
        "mechanism_figure": {
            "columns": ["obtuse", "right_angle", "acute"],
            "angle_convention": "interior angle beta = 180 degrees - abs(heading change delta)",
            "proposed_interior_angles_deg": [135, 90, 45],
            "rows": ["steering_action", "requested_feed", "preview_distance", "actual_heading", "actual_curvature", "actual_speed"],
            "source": "same final policy and same complete composite-path trace",
            "existing_two_column_figure_satisfies_new_requirement": False,
        },
        "manuscript_main_figures": 9, "manuscript_target_pages": [20, 25],
        "geometry_proposal": {
            "status": "design_proposal_only_not_registered_or_evaluated",
            "straight_lengths_mm": [220, 160, 20, 200, 180],
            "corner_heading_changes_deg": [45, -90, 135],
            "arc_radius_mm": 30, "arc_sweep_deg": -90,
            "closed": False, "approximate_reference_length_mm": 827.1238898038469,
        },
        "independent_seeds_default": 3,
        "single_model_case_study_may_be_reported_with_repeatability_limit": True,
        "algorithm_rounds_used": 2, "algorithm_rounds_cap": 2,
        "additional_algorithm_round_authorized_by_this_scope_revision": False,
        "sealed_test_opened_historically": True, "sealed_test_resume_allowed": False,
        "new_policy_evaluations": 0, "new_physics_ticks": 0, "new_training_updates": 0,
        "formal_model_adopted": False, "goal_complete": False,
        "guide": GUIDE.name, "guide_sha256": hashlib.sha256(content).hexdigest(),
        "supersedes_planning_scope": "paper_evidence_review/goal_alignment/continuation.json",
        "historical_guide": "paper_evidence_review/single_case_scope/guide_before.md",
        "next_action": next_action,
    }
    GUIDE.write_bytes(content)
    write(STAGE / "continuation.json", scope)
    write(STAGE / "historical_artifact_mapping.json", {
        "previous_receipt": "paper_evidence_review/goal_alignment/receipt.json",
        "previous_receipt_sha256": sha(BASE / "paper_evidence_review/goal_alignment/receipt.json"),
        "living_artifact": GUIDE.name, "previous_sha256": old_guide_hash,
        "exact_previous_bytes_preserved_at": "paper_evidence_review/single_case_scope/guide_before.md",
        "current_sha256": sha(GUIDE),
        "reason": "Explicit user-requested revision from multi-case scope to one complete composite case; prior receipts unchanged.",
    })
    sources = ["record_single_case_scope.py", "paper_evidence_review/goal_alignment/receipt.json",
               "paper_evidence_review/confirmatory/interruption_receipt.json",
               "论文研究边界与续接指令_20260920.md", "s2_figure_layer_contract.md"]
    output_paths = [GUIDE] + sorted(p for p in STAGE.iterdir() if p.is_file())
    receipt = {
        "created_utc": now, "status": "completed_single_case_goal_figure_writing_guide_revision",
        "sources_sha256": {r: sha(BASE / r) for r in sources},
        "outputs_sha256": {str(p.relative_to(BASE)): sha(p) for p in output_paths},
        "new_training_or_simulation": False, "goal_complete": False,
        "historical_records_and_models_preserved": True,
        "live_usage_snapshot": {"weekly_remaining_percent": 48, "reset_credits": 0, "reset_used": False},
    }
    write(STAGE / "receipt.json", receipt)
    latest = {
        "path": "paper_evidence_review/single_case_scope/continuation.json",
        "sha256": sha(STAGE / "continuation.json"),
        "receipt": "paper_evidence_review/single_case_scope/receipt.json",
        "receipt_sha256": sha(STAGE / "receipt.json"), "status": scope["status"],
    }
    state = read(STATE)
    state.update(updated_utc=now, goal_complete=False, original_objective_unchanged=True,
                 current_action="single_composite_case_goal_and_writing_scope_revised",
                 current="One complete composite case is the target; no requirement for all three historical long cases to succeed.",
                 next=next_action, next_action=next_action,
                 latest_execution_findings=scope, paper_single_case_scope=scope)
    if "paper_goal_alignment" in state:
        state["paper_goal_alignment"]["superseded_planning_scope_by"] = latest["path"]
        state["paper_goal_alignment"]["historical_guide_preserved_at"] = scope["historical_guide"]
    resume = state["latest_resume_20260919"]
    resume.update(next_action=next_action, latest_execution_utc=now, goal_complete=False,
                  learning_status="no_new_experiments_single_composite_case_scope_revised")
    review = resume["paper_evidence_review"]
    review.update(next_action=next_action, latest_continuation=latest,
                  required_primary_geometry_cases=1,
                  all_three_historical_long_paths_must_succeed=False,
                  historical_public_long_results_scope="Background limitation; not a mandatory three-path acceptance gate.",
                  current_goal_success_witness_verified=False,
                  current_guide=GUIDE.name, current_guide_sha256=sha(GUIDE))
    for boundary in [state["paper_research_boundary"], resume["paper_research_boundary"]]:
        boundary.update(latest_user_clarification=scope["user_instruction"],
                        required_primary_geometry_cases=1,
                        all_three_historical_long_paths_must_succeed=False,
                        claim_scope=scope["claim_scope"],
                        source_priority="Latest explicit user scope and current guide override earlier multi-case planning defaults; hard constraints unchanged.",
                        goal_and_writing_guide=GUIDE.name, guide_sha256=sha(GUIDE),
                        latest_scope_receipt=latest["receipt"],
                        latest_scope_receipt_sha256=latest["receipt_sha256"])
    write(STATE, state)
    write(STAGE / "state_update_receipt.json", {
        "created_utc": now, "state_before_sha256": sha(STAGE / "execution_state_before.json"),
        "state_after_sha256": sha(STATE), "scope_receipt_sha256": sha(STAGE / "receipt.json"),
        "guide_sha256": sha(GUIDE), "goal_complete": False})
    # Validate the actual delivered artifacts, not implementation-mirroring tests.
    for group in ["sources_sha256", "outputs_sha256"]:
        for path, digest in receipt[group].items():
            assert sha(BASE / path) == digest, path
    actual = read(STATE)
    assert actual["paper_single_case_scope"]["required_primary_geometry_cases"] == 1
    assert not actual["paper_single_case_scope"]["all_three_historical_long_paths_must_succeed"]
    print(json.dumps({"guide": GUIDE.name, "primary_cases": 1, "main_figures": 9,
                      "mechanism_columns": scope["mechanism_figure"]["columns"],
                      "utf8_no_bom": True, "receipt_bindings_verified": sum(len(receipt[k]) for k in ["sources_sha256", "outputs_sha256"]),
                      "new_experiments": 0, "historical_guide_hash_preserved": old_guide_hash}, ensure_ascii=False))


if __name__ == "__main__":
    main()
