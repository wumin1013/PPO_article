"""Audit a completed, uniformly selected reserve replay against its parent runs."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def final_runs(record):
    final = next(row for row in record["iterations"] if row["iteration"] == 3)
    return {run["dt_s"]: run for run in final["runs"]}


def main():
    parent_dir = ROOT / "results/copp_v5b_complete_development"
    replay_dir = ROOT / "results/copp_v5c_complete_development"
    parents = {row["path_id"]: row for row in read(parent_dir / "summary.json")}
    replays = read(replay_dir / "summary.json")
    assert len(parents) == len(replays) == 13
    assert set(parents) == {row["path_id"] for row in replays}
    checks = []
    total = 0
    successful = 0
    for replay in replays:
        parent = parents[replay["path_id"]]
        assert replay["status"] == "completed"
        assert [row["iteration"] for row in replay["iterations"]] == [1, 2, 3]
        for iteration in replay["iterations"]:
            assert iteration["solver_rerun"] is False
            parent_profile = Path(iteration["parent_profile"])
            assert parent_profile.parent == parent_dir
            assert sha(parent_profile) == iteration["parent_profile_sha256"]
            assert {run["dt_s"] for run in iteration["runs"]} == {0.001, 0.0005}
            for run in iteration["runs"]:
                assert run["execution"]["extra_time_reserve_fraction"] == 0.001
                assert sha(replay_dir / run["trace"]) == run["trace_sha256"]
                total += 1
                successful += bool(run["metrics"]["success"])
        old_runs = final_runs(parent)
        new_runs = final_runs(replay)
        clocks = parent["planner"]["clocks"]
        checks.append({
            "path_id": replay["path_id"],
            "candidate_iteration": 3,
            "parent_solve_wall_s": sum(row["solver_wall_s"] for row in parent["iterations"]),
            "parent_planning_clocks": clocks,
            "replay_total_wall_s": replay["total_wall_s"],
            "new_optimization_performed": False,
            "parent_profile_hashes_match": True,
            "trace_hashes_match": True,
            "resolution_time_difference_s": abs(new_runs[0.001]["metrics"]["complete_time_s"] - new_runs[0.0005]["metrics"]["complete_time_s"]),
            "comparisons": [{
                "dt_s": dt,
                "parent_complete_time_s": old_runs[dt]["metrics"]["complete_time_s"],
                "replay_complete_time_s": new_runs[dt]["metrics"]["complete_time_s"],
                "replay_success": new_runs[dt]["metrics"]["success"],
                "vector_peaks": new_runs[dt]["metrics"]["vector_peaks"],
                "vector_limit_ratios": new_runs[dt]["metrics"]["vector_limit_ratios"],
                "deviation_mm": new_runs[dt]["metrics"]["max_polyline_deviation_mm"],
                "geometry": new_runs[dt]["metrics"]["geometry"],
                "trace": str(replay_dir / new_runs[dt]["trace"]),
                "trace_sha256": new_runs[dt]["trace_sha256"],
            } for dt in (0.001, 0.0005)],
        })
    selection = ROOT / "audit/copp_v5c_reserve_selection.json"
    review = ROOT / "audit/distance_index_independent_review_v5.json"
    result = {
        "scope": "Observed development paths only; no independent final holdout claim",
        "method_identity": "Official COPP 0.2.3 TOPP3-SOCP with project geometry, vector constraints and common command evaluator",
        "reserve_fraction": 0.001,
        "selection_record": str(selection),
        "selection_record_sha256": sha(selection),
        "evaluator_independent_review": str(review),
        "evaluator_independent_review_sha256": sha(review),
        "programs": len(checks),
        "command_runs": total,
        "successful_command_runs": successful,
        "all_command_runs_pass": successful == total == 78,
        "maximum_fixed_candidate_resolution_time_difference_s": max(row["resolution_time_difference_s"] for row in checks),
        "cases": checks,
    }
    output = ROOT / "audit/copp_v5c_final_development_audit"
    output.with_suffix(".json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# COPP v5c统一时间余量开发审计", "",
        "方法身份：官方COPP 0.2.3 TOPP3-SOCP，加项目几何、向量约束与共同完整命令评价适配。所有程序均已观察，仅属于开发集。", "",
        f"验证集选择0.1%余量后统一重放13条完整路径；三轮、双分辨率共{successful}/{total}轨迹通过。固定第三轮，不逐路径挑选版本或轮次。父剖面、全部新轨迹校验均匹配。", "",
        "| 路径 | v5b 1 ms (s) | v5c 1 ms (s) | v5c 0.5 ms (s) | 双分辨率最大jerk (mm/s³) |",
        "|---|---:|---:|---:|---:|"]
    for row in checks:
        a, b = row["comparisons"]
        lines.append(f"| {row['path_id']} | {a['parent_complete_time_s']:.4f} | {a['replay_complete_time_s']:.4f} | {b['replay_complete_time_s']:.4f} | {max(a['vector_peaks'][2], b['vector_peaks'][2]):.3f} |")
    lines += ["", "v5c没有重新优化。原几何、约束组装、初值和三轮SOCP成本继承自v5b；重放墙钟仅包括重新生成命令及评价保存。两类成本在JSON中分开，不能将重放时间误称完整规划时间。", "",
        "完整执行包括原有序门、全命令区间几何、向量后向差分约束以及起终静止。距离索引已由独立逐点审查核实与原穷举结果完全相同；GateTracker未变。", "",
        "此记录不支持硬实时、实际轴跟踪、切削质量或独立最终测试结论。IJMTM原文仍缺失，当前公开实现不称为该文全流程复现。", ""]
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
