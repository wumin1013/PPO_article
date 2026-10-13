"""Bind the reviewed manuscript to frozen evidence; run only in conda PPO.

This performs file-integrity and existing-result checks, not new simulation.
Historical records are copied before any status or audit record is replaced.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess

ROOT = Path(r"D:\PPO_article_work")
BASE = ROOT / "论文项目/joint_policy_20260915_v1"
OUT = Path(r"D:\PPO_article_joint_runs\natural_closed_resume")
EVIDENCE = BASE / "paper_evidence_review/verified_results"
SRC = ROOT / "论文项目/submission_package_EAAI/06_latex_source_files/latex_source_anonymous"
BUILD = OUT / "manuscript_build"
FIGS = EVIDENCE / "figures_final"
BASE_PDF = Path(r"D:\Nutstore\1\基于强化学习的运动规划方法\论文项目\submission_package_EAAI\01_manuscript_without_author_details\manuscript_without_author_details.pdf")
NOW = datetime.now(timezone.utc)
STAMP = NOW.strftime("%Y%m%dT%H%M%S%fZ")
HISTORY = OUT / "delivery_state_history" / STAMP


def read_json(path):
    return Path(path).read_text(encoding="utf-8")


def load(path):
    return json.loads(read_json(path))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def record(path):
    path = Path(path)
    return {"path": str(path), "sha256": digest(path)}


def backup(path):
    path = Path(path)
    if not path.exists():
        return None
    HISTORY.mkdir(parents=True, exist_ok=True)
    target = HISTORY / path.name
    if target.exists():
        raise RuntimeError(f"Duplicate backup name: {target}")
    shutil.copy2(path, target)
    return record(target)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def check(success, message):
    if not success:
        raise RuntimeError(message)


def verify_result(obj):
    check(obj["success"] is True, "Incomplete primary result")
    check(obj["geometry"]["ordered_complete"] and obj["geometry"]["corridor_inside"], "Geometry failure")
    check(obj["geometry"]["gates_passed"] == obj["geometry"]["gate_count"] == 2362, "Gate mismatch")
    check(obj["audit"]["kinematic_pass"] and obj["audit"]["rest_pass"], "Dynamics/rest failure")
    check(obj["analytic"]["analytic_corridor_pass"] and obj["analytic"]["unresolved_segments"] == 0, "Unresolved analytic audit")


check(os.environ.get("CONDA_DEFAULT_ENV") == "PPO", "Activate conda PPO first")
branch = subprocess.check_output(["git", "-C", str(ROOT), "branch", "--show-current"], text=True).strip()
check(branch == "codex/complex-path-adaptive-preview", "Unexpected branch")
baseline_sha = digest(BASE_PDF)
check(baseline_sha == "074724d783318f227720ce09fe07410eb67221c572d661962c4ff2d7db64caa7", "Approved baseline changed")
binding_path = EVIDENCE / "paper_result_binding.json"
binding = load(binding_path)
check(binding["method_frozen"] and binding["research_acceptance"]["passed"], "Research not accepted")
checked_bindings = []


def verify_bindings(value):
    if isinstance(value, dict):
        for key, expected in value.items():
            if key.endswith("_sha256") and key[:-7] in value:
                path = Path(value[key[:-7]])
                check(digest(path) == expected, f"Binding mismatch: {path}")
                checked_bindings.append(str(path))
        for child in value.values():
            verify_bindings(child)
    elif isinstance(value, list):
        for child in value:
            verify_bindings(child)


verify_bindings(binding)
check(binding["reference_sha256"] == "1a36c33792dac716fc9e1fb83f92861837bfbcaf93e8611eee332ba245feee5a", "Reference changed")
runtime_manifest = load(binding["runtime_sources_manifest"])
for item in runtime_manifest["records"]:
    check(digest(item["snapshot"]) == item["sha256"], f"Frozen source mismatch: {item['snapshot']}")
check(digest(OUT / "frozen_policy_sources/inherited_policy.pt") == binding["model_sha256"], "Frozen model mismatch")
check(digest(OUT / "frozen_policy_sources/selected_policy.json") == binding["selected_policy_sha256"], "Frozen parameters mismatch")

results = {name: load(paths["result"]) for name, paths in binding["primary_methods"].items()}
refined = {name: load(path) for name, path in binding["resolution_results"].items()}
for result in [*results.values(), *refined.values()]:
    verify_result(result)
numerical = load(binding["research_acceptance"]["numerical_gate"])
check(numerical["numerical_pass"], "Numerical screen failed")
for method, gate in numerical["comparisons"].items():
    nominal_gain = results[method]["complete_time_s"] - results["Joint"]["complete_time_s"]
    refined_gain = refined[method]["complete_time_s"] - refined["Joint"]["complete_time_s"]
    check(abs(nominal_gain - gate["nominal_gain_s"]) < 1e-9, "Nominal gain mismatch")
    check(abs(refined_gain - gate["half_ms_gain_s"]) < 1e-9, "Refined gain mismatch")
    check(gate["passed"] and min(nominal_gain, refined_gain) > gate["required_gain_s"], "Gain threshold failed")

fig_receipt = load(FIGS / "receipt.json")
check(fig_receipt["source_binding_sha256"] == digest(binding_path), "Figure evidence binding mismatch")
for name, expected in fig_receipt["files_sha256"].items():
    check(digest(FIGS / name) == expected, f"Figure artifact changed: {name}")
for number in range(1, 10):
    name = f"Figure_{number}.pdf"
    check(digest(SRC / "figures_verified" / name) == digest(FIGS / name), f"Manuscript figure mismatch: {name}")

tex = (SRC / "manuscript.tex").read_text(encoding="utf-8")
bbl = (SRC / "manuscript.bbl").read_text(encoding="utf-8")
aux = (BUILD / "manuscript.aux").read_text(encoding="utf-8")
log = (BUILD / "manuscript.log").read_text(encoding="utf-8")
fls = (BUILD / "manuscript.fls").read_text(encoding="utf-8")
bad_log = [line for line in log.splitlines() if re.search(r"Overfull|Underfull|undefined|(?:LaTeX|Package \S+) Warning|^!", line)]
check(not bad_log, f"Unresolved compilation diagnostic: {bad_log}")
check(digest(SRC / "manuscript.bbl") == digest(BUILD / "manuscript.bbl"), "Compiled bibliography mismatch")
check(all(f"./figures_verified/Figure_{n}.pdf" in fls for n in range(1, 10)), "Wrong figure inputs")
check(len(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{Figure_[1-9]\.pdf\}", tex)) == 9, "Nine figures required")
keys = {key.strip() for group in re.findall(r"\\cite(?:\[[^]]*\])?\{([^}]+)\}", tex) for key in group.split(",")}
bibkeys = set(re.findall(r"\\bibitem\{([^}]+)\}", bbl))
auxkeys = set(re.findall(r"\\bibcite\{([^}]+)\}", aux))
check(keys == bibkeys == auxkeys and len(keys) == 20, "Citation resolution mismatch")
for path in [SRC / "manuscript.tex", SRC / "references.bib", SRC / "manuscript.bbl"]:
    raw = path.read_bytes()
    check(not raw.startswith(b"\xef\xbb\xbf") and "\ufffd" not in raw.decode("utf-8"), f"Encoding problem: {path}")
pdfinfo = subprocess.check_output(["pdfinfo", str(BUILD / "manuscript.pdf")]).decode("utf-8", errors="replace")
page_count = int(re.search(r"^Pages:\s+(\d+)", pdfinfo, re.MULTILINE).group(1))
check(page_count == 21, "Unexpected reviewed candidate page count")
check(all((BUILD / f"reviewed-page-{n:02d}.png").exists() for n in range(1, 22)), "Missing reviewed page")
with (EVIDENCE / "all34_corner_statistics.csv").open(encoding="utf-8", newline="") as stream:
    corner_rows = list(csv.DictReader(stream))
check(len(corner_rows) == 102, "Expected three methods times all 34 corners")

# All integrity checks above must pass before the delivery records are written.
audit_path = EVIDENCE / "citation_claim_audit.json"
audit_backup = backup(audit_path)
audit = load(audit_path)
for citation in audit["citations"]:
    key = citation["key"]
    citation["locations"] = [i for i, line in enumerate(tex.splitlines(), 1) if any(key in [k.strip() for k in group.split(",")] for group in re.findall(r"\\cite(?:\[[^]]*\])?\{([^}]+)\}", line))]
    check(citation["locations"], f"Missing citation: {key}")
audit.update({"updated_utc": NOW.isoformat(), "bibliography_sha256": digest(SRC / "references.bib"), "bbl_sha256": digest(SRC / "manuscript.bbl"), "tex_sha256": digest(SRC / "manuscript.tex"), "cited_count": len(keys), "resolved_bibitems": len(bibkeys), "no_unresolved_citations": True, "previous_audit": audit_backup, "corrections": ["Sencer2015 second author corrected from Kazunori to Kosuke Ishizaki against verified metadata.", "Sencer2014 is the historical key for the distinct 2015 IJAMT paper, DOI 10.1007/s00170-014-6386-2.", "Schulman2017PPO explicitly prints arXiv:1707.06347 in the unsrt bibliography."]})
write_json(audit_path, audit)

page_notes = {
    1: "Anonymous title and abstract; numeric claims and contribution scope readable.",
    2: "Introduction and contribution framing; no text collision or missing reference.",
    3: "Figure 1 planning framework; labels, caption and surrounding text readable.",
    4: "Figure 2 case, tolerance and angle coverage; equations and limits intact.",
    5: "Figure 3 execution chain; learning provenance separated from numerical timing.",
    6: "72-dimensional state table, causal memory and action decoder all fit.",
    7: "Vector KCM equations and complete algorithm fit without clipping.",
    8: "Reward, historical PPO versus current parameter optimization, and residual equations fit.",
    9: "Fair baseline protocol and local speed-deficit definition readable.",
    10: "Shape definitions, audit-layer table and original efficiency diagnosis readable.",
    11: "Complete result table, path-length decomposition and numerical-screen text fit.",
    12: "Figure 4 full XY/V/A/J and complete stopping; panel labels separated.",
    13: "Figure 5 three equal-aspect local shapes and mechanism explanation readable.",
    14: "Figure 6 has 135/90/45 degree columns and all six required rows; legends and units legible.",
    15: "All-34 statistics and cruise limits explicitly retain unfavorable findings.",
    16: "Figure 7 all corners and 0.25 m cruise bins; colorbar, axes and caption intact.",
    17: "Figure 8 fixed-preview opportunities and two interventions; failures distinct from full times.",
    18: "Figure 9 resolution and conditional-seed evidence; failed seeds visible; no invented error bars.",
    19: "Conclusion, data/code availability and beginning of bibliography fit.",
    20: "References 7-18 readable; distinct Sencer papers and corrected author names.",
    21: "Final two references readable, including the PPO arXiv identifier; expected end-page whitespace."
}
page_qa = {"reviewed_utc": NOW.isoformat(), "pdf": record(BUILD / "manuscript.pdf"), "page_count": 21, "rendering": "Poppler at 110 dpi", "reviewer": "Codex visual inspection in the continuing task; pages 1-4 reviewed before context handoff, pages 5-21 after handoff", "all_pages_visually_reviewed": True, "unresolved_layout_issues": [], "pages": [{"page": n, "render": record(BUILD / f"reviewed-page-{n:02d}.png"), "passed": True, "notes": page_notes[n]} for n in range(1, 22)]}
backup(EVIDENCE / "pdf_page_qa.json")
write_json(EVIDENCE / "pdf_page_qa.json", page_qa)
figure_qa = {"reviewed_utc": NOW.isoformat(), "source_binding": record(binding_path), "visual_qa_complete": True, "scope": "All nine figures inspected in the final rendered manuscript; initial generation receipt is retained unchanged.", "raw_data_smoothed": False, "unresolved_layout_issues": [], "figures": [{"number": n, "file": record(FIGS / f"Figure_{n}.pdf"), "page": {1: 3, 2: 4, 3: 5, 4: 12, 5: 13, 6: 14, 7: 16, 8: 17, 9: 18}[n], "passed": True} for n in range(1, 10)]}
backup(FIGS / "visual_qa.json")
write_json(FIGS / "visual_qa.json", figure_qa)

pdf_target = EVIDENCE / "manuscript_without_author_details.pdf"
backup(pdf_target)
shutil.copy2(BUILD / "manuscript.pdf", pdf_target)
check(digest(pdf_target) == page_qa["pdf"]["sha256"], "Delivery PDF copy mismatch")
index_path = EVIDENCE / "最终交付与证据索引.md"
check(index_path.exists(), "Write the delivery/evidence index before finalization")
artifacts = [pdf_target, SRC / "manuscript.tex", SRC / "references.bib", SRC / "manuscript.bbl", SRC / "elsarticle.cls", SRC / "unsrt.bst", binding_path, audit_path, EVIDENCE / "method_summary.csv", EVIDENCE / "all34_corner_statistics.csv", EVIDENCE / "summary_tables.json", EVIDENCE / "pdf_page_qa.json", FIGS / "visual_qa.json", FIGS / "receipt.json", FIGS / "figure_contract.json", index_path, BUILD / "manuscript.log", OUT / "draw_verified_figures.py", Path(__file__)]
artifacts.extend(SRC / "figures_verified" / f"Figure_{n}.pdf" for n in range(1, 10))
acceptance = {
    "accepted_utc": NOW.isoformat(), "project": "PPO_article", "working_directory": str(BASE), "branch": branch,
    "research_acceptance_passed": True, "method_frozen": True, "final_manuscript_accepted": True, "goal_complete": True,
    "scope": "Disclosed natural development lap and selected inherited composite controller; no unseen-path or independent-PPO claim.",
    "baseline_pdf": record(BASE_PDF), "baseline_pages": 18, "baseline_unchanged": True,
    "baseline_source_binding": record(OUT / "baseline_source_binding.json"),
    "pdf_pages": 21, "main_figures": 9, "resolved_citations": 20, "all34_rows": len(corner_rows),
    "task_contract": {"reference_length_m": 44.420514, "corners": 34, "ordered_gates": 2362, "one_sided_euclidean_tolerance_mm": 1.25, "vector_VAJ_limits": [100, 2000, 62500], "physical_rest_min_s": 0.020, "cruise_threshold_mm_s": 99},
    "complete_times_s": {name: {"1_ms": obj["complete_time_s"], "0.5_ms": refined[name]["complete_time_s"]} for name, obj in results.items()},
    "numerical_comparisons": numerical["comparisons"],
    "result_integrity": {"checksummed_binding_paths": len(checked_bindings), "frozen_runtime_source_files": len(runtime_manifest["records"]), "primary_and_refined_full_audits_passed": 6},
    "resolution_result_files": {name: record(path) for name, path in binding["resolution_results"].items()},
    "verification": {"all_pages_visually_reviewed": True, "all_nine_figures_match_included_files": True, "no_unresolved_citations_or_layout_warnings": True, "utf8_without_bom": True},
    "evidence_limits": ["Two-resolution observed sensitivity screen, not a continuous-time error bound.", "Three conditional search seeds share one inherited backbone; zero independent full-pipeline PPO replications and zero additional PPO updates in this continuation.", "Historical optimization budgets are unequal and disclosed; the strongest successful fixed-preview checkpoint is retained.", "No new sealed test was used; previously exposed sealed data is not called unseen.", "Joint cruise >=99 mm/s remains below S2, and local reversal/overshoot and unfavorable corners remain.", "Numerical command generation only: no hardware timing, servo or cutting-quality validation."],
    "immutable_stage_snapshot_note": "paper_result_binding.json and numerical_acceptance.json retain their original stage-local completion fields. This final acceptance is authoritative for delivery status; their contents and hashes are not rewritten.",
    "artifacts": [record(path) for path in artifacts]
}
backup(EVIDENCE / "final_acceptance.json")
write_json(EVIDENCE / "final_acceptance.json", acceptance)

state_path = BASE / "当前执行状态.json"
live_path = OUT / "live_research_summary.json"
previous_state = backup(state_path)
previous_live = backup(live_path)
state = {"updated_utc": NOW.isoformat(), "project": "PPO_article", "working_directory": str(BASE), "branch": branch, "original_objective_unchanged": True, "current": "当前自然复杂闭合长路径的研究验收及21页英文论文、九幅主图验收完成；证据范围和重复性限制见最终验收。", "goal_complete": True, "method_frozen": True, "research_acceptance_passed": True, "final_manuscript_accepted": True, "authoritative_acceptance": record(EVIDENCE / "final_acceptance.json"), "delivery_index": record(index_path), "final_pdf": record(pdf_target), "complete_times_s": acceptance["complete_times_s"], "new_ppo_updates": 0, "independent_full_pipeline_ppo_seeds": 0, "conditional_search_seeds": 3, "heldout_test_result": None, "historical_records": {"previous_execution_state": previous_state, "previous_live_summary": previous_live, "prior_history_directory": str(OUT / "live_state_history"), "binding_snapshot": str(binding_path)}, "operational_constraints": {"no_subagents": True, "no_new_project_number": True, "no_branch_switch_or_reset": True, "no_delete_clean_commit_push": True, "training_and_intermediate_root": str(OUT)}, "next_research_if_requested": ["从独立初始化开展完整PPO训练重复，单独登记预算。", "在当前冻结后建立新的封存测试，并只用于最终评估。", "测量硬件推理时延与真实伺服/加工表现。"]}
write_json(state_path, state)
write_json(live_path, {**state, "status_source": str(state_path)})

historical_note = "> 历史阶段记录（保留原文）：本文中的“未完成”“待冻结”等表述对应当时阶段；当前有效状态以 [最终交付与证据索引](" + index_path.as_posix() + ") 和 [最终验收](" + (EVIDENCE / "final_acceptance.json").as_posix() + ") 为准。\n\n"
for name in ["基稿与当前实现核对.md", "研究证据与论文边界.md", "论文证据架构.md"]:
    path = BASE / name
    backup(path)
    old_text = path.read_text(encoding="utf-8-sig")
    path.write_text(historical_note + old_text, encoding="utf-8")

print(json.dumps({"passed": True, "final_pdf": str(pdf_target), "pdf_sha256": digest(pdf_target), "pages": page_count, "main_figures": 9, "citations": len(keys), "binding_paths_checked": len(checked_bindings), "runtime_sources_checked": len(runtime_manifest["records"]), "state_backups": str(HISTORY), "final_acceptance": str(EVIDENCE / "final_acceptance.json")}, ensure_ascii=False, indent=2))
