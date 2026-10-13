"""Regenerate COPP development summary and provenance from complete run records."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    folder=ROOT/"results/copp_development_v4e"
    registration=json.loads((folder/"registration_all.json").read_text("utf-8"))
    rows=[];details=[]
    for name in registration["input_sha256"]:
        record=json.loads((folder/(name+".json")).read_text("utf-8"))
        details.append(record)
        if record["status"]!="completed":
            rows.append(dict(path=name,status=record["status"],error=record.get("error")))
            continue
        for iteration in record["iterations"]:
            for run in iteration["runs"]:
                metric=run["metrics"]
                rows.append(dict(path=name,iteration=iteration["iteration"],dt_s=run["dt_s"],
                    success=metric["success"],completion_time_s=metric["complete_time_s"],
                    continuous_motion_time_s=run["execution"]["continuous_motion_duration_s"],
                    peaks=metric["vector_peaks"],geometry_complete=metric["geometry"]["geometry_complete"],
                    rest_s=metric["terminal_rest_s"],time_dilation=run["execution"]["time_dilation"],
                    solver_wall_s=iteration["solver_wall_s"],sampling_wall_s=run["execution"]["sampling_wall_s"],
                    trace_sha256=run["trace_sha256"]))
    comparisons=[]
    for name in ("dense","validation_trapezoid","validation_zigzag"):
        for iteration in (1,2,3):
            old=np.load(ROOT/"results/copp_development_v4"/f"{name}_iteration{iteration}_profile.npz")
            new=np.load(folder/f"{name}_iteration{iteration}_profile.npz")
            comparisons.append(dict(path=name,iteration=iteration,
                max_abs_difference={k:float(np.max(np.abs(old[k]-new[k]))) for k in ("s","a","b")}))
    files=[ROOT/"src"/n for n in ("copp_path_adapter_v4.py","copp_geometry_only_v4.py","copp_time_projection_v4.py","copp_time_audit_v4.py","copp_candidate_probe_v4.py","li2023_core_v4.py")]
    files += [ROOT/"literature/copp_source/LICENSE",ROOT/"literature/copp_wheels/copp_py-0.2.3-cp310-cp310-win_amd64.whl",ROOT/"literature/Wang2026_RDDP_arxiv_2605.19089.pdf"]
    final=[x for x in rows if x.get("iteration")==3]
    report=dict(scope="Four development programs, all iterations retained; no independent final-test evidence.",
        sha256={str(p.relative_to(ROOT)):sha(p) for p in files},
        registered_sources_match=registration["source_sha256"]==sha(ROOT/"src/copp_geometry_only_v4.py") and registration["core_sha256"]==sha(ROOT/"src/copp_path_adapter_v4.py"),
        completed_runs=sum(x.get("success",False) for x in rows),attempted_trajectory_runs=sum("success" in x for x in rows),
        fixed_third_iteration=final,all_runs=rows,
        old_vs_geometry_only_profile_comparison=comparisons,
        first_default_attempt="results/copp_development_v4: 2/18 complete commands passed; train_sine failed before COPP geometry construction. Original failures remain archived.")
    (ROOT/"audit/copp_development_summary_v4.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    text=["# COPP 全程时间参数化开发验证","","结果由完整运行记录生成；以下均为开发路径，不是独立最终测试。",
        "","| 完整路径 | 周期 / ms | 完成时间 / s | V峰值 | A峰值 | J峰值 | 通过 |","|---|---:|---:|---:|---:|---:|---|"]
    for row in final:
        p=row["peaks"]
        text.append(f'| {row["path"]} | {1000*row["dt_s"]:.1f} | {row["completion_time_s"]:.4f} | {p[0]:.5f} | {p[1]:.5f} | {p[2]:.5f} | {row["success"]} |')
    text.extend(["",f'三次迭代的完整命令运行：{report["completed_runs"]}/{report["attempted_trajectory_runs"]} 通过。固定第三轮作为候选输出。',
        "", "共同限值：100 mm/s、2000 mm/s²、62500 mm/s³；名义单侧公差1.25 mm；有解析圆弧时先扣弦误差预算。完整有序遍历、全命令区间几何和最终20 ms静止由共同metrics验收。",
        "", "## 实际计算成本（秒）", "", "| 路径 | 几何 | 网格与约束 | 二阶初值 | 三轮SOCP合计 |", "|---|---:|---:|---:|---:|"])
    for record in details:
        if "planner" not in record:continue
        c=record["planner"]["clocks"]
        text.append(f'| {record["path_id"]} | {c["geometry_wall_s"]:.6f} | {c["grid_constraints_wall_s"]:.6f} | {c["seed_wall_s"]:.6f} | {sum(x["solver_wall_s"] for x in record["iterations"]):.6f} |')
    text.extend(["","这是本机开发运行的墙钟时间，包含独立求解阶段记录；没有硬实时验证。采样和共同审计时间另存在逐运行JSON中。",
        "", "前后S2原始几何在三个能构造默认cap的程序上生成完全相同的站点和未伸缩COPP剖面（逐数组误差见JSON）；小正弦保留一个2.22e-16 mm原始直线piece，仅不为其制造重复的浮点站点。",
        "", "原始默认结果与所有修订版本保留在copp_development_v4、v4b、v4c、v4d；本表统一来自v4e冻结源码运行。"])
    (ROOT/"audit/copp_development_summary_v4.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    print(json.dumps({"passed":report["completed_runs"],"attempted":report["attempted_trajectory_runs"],"source_match":report["registered_sources_match"]}))


if __name__=="__main__":main()
