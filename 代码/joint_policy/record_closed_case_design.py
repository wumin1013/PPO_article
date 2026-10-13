"""Update the living plan to the user-requested >300 s closed contour."""
from pathlib import Path
from datetime import datetime, timezone
import json
from design_composite_long_case import sha, write

BASE = Path(__file__).resolve().parent
OUT = BASE / "paper_evidence_review/composite_closed_case"
GUIDE = BASE / "PPO_article_研究目标_图表与写作指南.md"
STATE = BASE / "当前执行状态.json"


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    if (OUT / "scope_update_receipt.json").exists():
        raise RuntimeError("Closed-case plan already recorded; do not overwrite its receipt.")
    assert sha(OUT / "guide_before.md") == sha(GUIDE)
    assert sha(OUT / "execution_state_before.json") == sha(STATE)
    now = datetime.now(timezone.utc).isoformat()
    audit = read(OUT / "geometry_audit.json")
    geometry = read(OUT / "geometry.json")
    receipt = read(OUT / "receipt.json")
    for rel, digest in receipt["outputs_sha256"].items():
        assert sha(BASE / rel) == digest
    text = GUIDE.read_text(encoding="utf-8")
    text = text.replace(
        "**本研究以一个完整综合案例为核心：一条路径同时包含长直线、短直线、钝角、直角、锐角和圆弧段。",
        "**本研究以一个完整综合案例为核心：一条单圈闭合长路径同时包含长直线、短直线、钝角、直角、锐角和圆弧段，完整执行时间超过300 s。")
    text = text.replace(
        "在一条同时含长短直线、三类拐角和圆弧的完整路径上，让学得策略协调转向、进给与前瞻，",
        "在一条完整一圈必然超过300 s、同时含长短直线、三类拐角和圆弧的闭合路径上，让学得策略协调转向、进给与前瞻，")
    text = text.replace(
        "采用**一个几何案例**，三类拐角从这个案例中提取局部窗口，不再分别建立三种长路径任务。初始默认用开放路径，避免把闭合接缝处理引入中心目标；这不改变圆弧段要求。",
        "采用**一个单圈闭合几何案例**，参考用户提供的凹凸轮廓形式重新设计，避免往返扫描。三类拐角从同一案例中提取局部窗口。起终点位于同一条长直线内部，几何切向连续；实际从静止起动、完成一圈后制动停稳，不能虚构跨停稳接缝的连续运动。")
    start = text.index("### 可供登记的一条具体几何提案")
    end = text.index("### 案例选择与数据角色", start)
    replacement = """### 已完成静态设计的单圈闭合长路径

当前唯一主案例：`paper_evidence_review/composite_closed_case/reference.csv`。解析构成、全部角点、三列机制窗口和哈希见同目录`geometry.json`，静态审核与时间下界见`geometry_audit.json`，预览图为`path_preview.png`。

这条路径借鉴用户图片的单圈凹凸轮廓形式，采用不同的凹槽、凸角、尺寸和圆弧连接，不复制原图的运动误差或性能数字。它已经完成几何设计，**尚未运行策略，不代表目标实验成功**。

| 项目 | 当前设计 |
|---|---|
| 拓扑 | 无自交的单圈闭合轮廓，顺序走完一圈；不是往返扫描 |
| 解析总长 | 40.468237 m |
| 沿参考以100 mm/s匀速的时间 | 404.682367 s，仅为参考值 |
| 允许切角后的完整时间下界 | **402.371026 s > 300 s**，由必须按序通过的截面计算；未额外计入加速、制动和停稳 |
| 长短直线 | 最长3535.534 mm；最短20 mm，保留局部毫米尺度，不把短特征整体放大 |
| 三类角 | 10个钝角、3个直角、1个锐角；全部14个角进入同案例角点统计 |
| 三列机制角点 | B：135°内角(3800,5200)；C：90°内角(7100,5400)；D：45°内角(9282.842712,6200) |
| 圆弧 | 4段，半径3000、1300、600、1700 mm；含凸、凹连接 |
| 起终点 | (1500,6000) mm；完成覆盖后回到起点附近并完整制动停稳 |
| 静态核验 | 无自交、无180°方向歧义、无相邻门冲突；非相邻段最小保守间距20 mm |
| 共同参考 | 4071个点，直线采样步长≤25 mm，弧最大弦高误差约0.001998 mm |

时间下界使用门截面的包围圆半径ε/cos(δ/2)，其中δ是相邻离散线段的航向转角。顺序门中心间距离减去两端包围圆半径的和，给出完整有序执行的路程下界；再除以100 mm/s。因此超过300 s由几何保证，不是故意降速凑出来。

解析圆弧与直线相切，但曲率可跳变；真实V/A/J仍须完整执行后审核。所有方法共用同一CSV，并保留解析圆弧定义，后续同时核查真实命令相对解析线/弧的距离。约0.002 mm的离散误差不能加到1.25 mm允差上。

此前827 mm开放路径仅是已被新要求替代的短提案；34.122 m四行往返方案保留为用户已否定的草案，不继续用作主案例。新闭合几何在策略运行前固定，方法尚未冻结。

"""
    text = text[:start] + replacement + text[end:]
    text = text.replace("| 完整可行 | 主方法在这一个综合案例上通过全部几何、运动学、覆盖及停稳检查 |",
                        "| 完整可行 | 主方法在这一个超过300 s的单圈闭合案例上通过全部几何、运动学、覆盖及停稳检查 |")
    text = text.replace("全参考路径；标记长/短段、三类内角、圆弧半径和方向；", "完整单圈闭合参考；标记长/短段、三类内角、圆弧半径、方向与超过300 s的时间下界；")
    text = text.replace(
        "目前已有完整短路径和局部机制素材，**尚未核验出一条满足本次全部几何构成并实现比较收益的完整综合案例**。",
        "目前已有完整短路径和局部机制素材，并已完成上述40.468 m闭合案例的静态几何审核；**这条案例尚未运行策略，完整成功与比较收益仍待实现**。")
    text = text.replace(
        "1. 离线核查已有完整路径是否满足最新综合案例构成；优先复用，避免重跑。\n2. 如需新案例，先核验上述几何提案并固定参考、三角窗口和比较指标；此步没有策略训练或物理执行。",
        "1. 已完成闭合长路径设计与静态审核，主案例和三列角点已固定；旧往返草案停止采用。\n2. 核对同案例执行、基线接口、解析圆弧距离与预算方案；保持共同参考和窗口，不重跑已完成实验。")
    text = text.replace("- [ ] 一条综合案例确实含长短直线、钝角、直角、锐角和圆弧，几何与数据角色清楚。",
                        "- [x] 一条单圈闭合综合长路径含长短直线、钝角、直角、锐角和圆弧，静态几何审核通过，保守时间下界402.371 s。")
    text = text.replace("本修订明确替代前一指南中多案例作为中心论证、两例机制图、三类长路径分别定稿和大规模总体评价图的安排。",
                        "最新修订采用用户要求的单圈闭合轮廓及超过300 s条件，替代短开放提案和往返草案；本研究继续替代更早的多案例中心论证、两例机制图、三类长路径分别定稿和大规模总体评价图安排。")
    text = text.replace("最新执行状态以本指南及本次范围修订收据为准。",
                        "短开放提案时期的指南还保存在`paper_evidence_review/composite_closed_case/guide_before.md`。最新执行状态以本指南及闭合案例的范围更新收据为准。")
    GUIDE.write_text(text, encoding="utf-8")
    write(OUT / "visual_qa.json", {
        "reviewed_utc": now, "status": "passed_geometry_preview_visual_review",
        "viewed": "path_preview.png", "sha256": sha(OUT / "path_preview.png"),
        "observations": ["Single irregular closed contour with separate concave and convex portions; no raster rows.",
                         "All three fixed corner windows legible and equal-aspect, with interior-angle and heading-change labels.",
                         "Start/end, direction, arc radii, units and non-performance status readable.",
                         "Caption identifies length and speed-only duration lower bound; no learned motion is drawn."],
        "pdf_page_visual_qa": "not claimed; preview only, not a completed manuscript figure",
    })
    next_action = (
        "Use only the designed single closed composite contour (40.468236747 m; ordered-gate duration lower bound 402.371025792 s). "
        "Prepare a bounded implementation/evaluation plan on this exact reference for complete success, reduced corner slowdown, "
        "more high-speed cruise and lower full time relative to credible S2/fixed-preview baselines. "
        "Do not use the rejected serpentine draft, reopen sealed tests or automatically launch a third optimization round."
    )
    continuation = {
        "created_utc": now, "status": "closed_long_case_designed_static_audit_passed_policy_goal_pending",
        "user_scope": "One closed irregular contour inspired by the supplied image, different geometry, all requested features, >300 s complete lap.",
        "required_primary_cases": 1, "closed": True, "lap_count": 1, "minimum_complete_duration_s": 300,
        "duration_requirement_strict": True, "duration_lower_bound_s": audit["ordered_gate_time_lower_bound_s"],
        "analytic_length_mm": audit["analytic_length_mm"],
        "reference": "paper_evidence_review/composite_closed_case/reference.csv",
        "reference_sha256": sha(OUT / "reference.csv"),
        "geometry": "paper_evidence_review/composite_closed_case/geometry.json",
        "geometry_sha256": sha(OUT / "geometry.json"),
        "geometry_receipt": "paper_evidence_review/composite_closed_case/receipt.json",
        "geometry_receipt_sha256": sha(OUT / "receipt.json"),
        "figure_preview": "paper_evidence_review/composite_closed_case/path_preview.png",
        "rejected_serpentine_design": "paper_evidence_review/composite_long_case",
        "corner_counts": audit["sharp_corner_counts"], "mechanism_columns": geometry["mechanism_columns"],
        "guide": GUIDE.name, "guide_sha256": sha(GUIDE),
        "goal_complete": False, "new_policy_evaluations": 0, "new_physics_ticks": 0, "new_training_updates": 0,
        "algorithm_rounds_used": 2, "algorithm_rounds_cap": 2,
        "method_frozen": False, "formal_model_adopted": False, "next_action": next_action,
    }
    write(OUT / "continuation.json", continuation)
    sources = ["record_closed_case_design.py", "paper_evidence_review/single_case_scope/receipt.json",
               "paper_evidence_review/composite_closed_case/receipt.json"]
    outputs = [GUIDE, OUT / "visual_qa.json", OUT / "continuation.json"]
    update_receipt = {
        "created_utc": now, "status": "completed_closed_long_case_guide_and_state_alignment",
        "sources_sha256": {p: sha(BASE / p) for p in sources},
        "outputs_sha256": {str(p.relative_to(BASE)): sha(p) for p in outputs},
        "previous_guide_preserved_at": "paper_evidence_review/composite_closed_case/guide_before.md",
        "previous_guide_sha256": sha(OUT / "guide_before.md"),
        "goal_complete": False, "new_training_or_simulation": False,
    }
    write(OUT / "scope_update_receipt.json", update_receipt)
    state = read(STATE)
    latest = {"path": "paper_evidence_review/composite_closed_case/continuation.json",
              "sha256": sha(OUT / "continuation.json"),
              "receipt": "paper_evidence_review/composite_closed_case/scope_update_receipt.json",
              "receipt_sha256": sha(OUT / "scope_update_receipt.json"), "status": continuation["status"]}
    state.update(updated_utc=now, goal_complete=False, next_action=next_action, next=next_action,
                 current_action="single_closed_long_contour_geometry_completed_policy_goal_pending",
                 current="One closed contour statically verified: 40.468 m, complete-lap speed-only lower bound 402.371 s; no policy rollout yet.",
                 paper_closed_case_design=continuation, latest_execution_findings=continuation)
    single = state["paper_single_case_scope"]
    single.update(updated_utc=now, status=continuation["status"], closed=True, minimum_complete_duration_s=300,
                  geometry_proposal={"status": "static_design_completed_no_policy_execution", "reference": continuation["reference"],
                                     "reference_sha256": continuation["reference_sha256"], "closed": True,
                                     "analytic_length_mm": audit["analytic_length_mm"], "time_lower_bound_s": audit["ordered_gate_time_lower_bound_s"]},
                  guide_sha256=sha(GUIDE), next_action=next_action)
    resume=state["latest_resume_20260919"]
    resume.update(latest_execution_utc=now, next_action=next_action,
                  learning_status="no_learning_running_closed_case_geometry_designed_goal_pending")
    resume["paper_evidence_review"].update(latest_continuation=latest, next_action=next_action,
        current_goal_success_witness_verified=False, current_guide_sha256=sha(GUIDE), primary_case=continuation["reference"],
        primary_case_closed=True, primary_case_minimum_duration_s=300,
        primary_case_static_time_lower_bound_s=audit["ordered_gate_time_lower_bound_s"])
    for boundary in [state["paper_research_boundary"], resume["paper_research_boundary"]]:
        boundary.update(latest_user_clarification=continuation["user_scope"], required_primary_geometry_cases=1,
                        required_case_closed=True, required_complete_duration_greater_than_s=300,
                        all_three_historical_long_paths_must_succeed=False,
                        goal_and_writing_guide=GUIDE.name, guide_sha256=sha(GUIDE),
                        latest_scope_receipt=latest["receipt"], latest_scope_receipt_sha256=latest["receipt_sha256"])
    write(STATE,state)
    write(OUT / "state_update_receipt.json", {
        "created_utc": now, "state_before_sha256": sha(OUT / "execution_state_before.json"),
        "state_after_sha256": sha(STATE), "scope_receipt_sha256": sha(OUT / "scope_update_receipt.json")})
    for mapping in [update_receipt["sources_sha256"],update_receipt["outputs_sha256"]]:
        for p,h in mapping.items(): assert sha(BASE/p)==h
    assert not GUIDE.read_bytes().startswith(b"\xef\xbb\xbf")
    assert not STATE.read_bytes().startswith(b"\xef\xbb\xbf")
    print(json.dumps({"guide_updated":True,"case_closed":True,"length_m":audit["analytic_length_mm"]/1000,
                      "duration_lower_bound_s":audit["ordered_gate_time_lower_bound_s"],"new_simulation":False}))


if __name__ == "__main__":
    main()
