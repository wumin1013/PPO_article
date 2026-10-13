"""Record the line-dominant closed case and update the living research guide."""
import json
from pathlib import Path
from datetime import datetime, timezone
from design_composite_long_case import sha, write

BASE=Path(__file__).resolve().parent
OUT=BASE/"paper_evidence_review/composite_closed_alternating_case"
GUIDE=BASE/"PPO_article_研究目标_图表与写作指南.md"
STATE=BASE/"当前执行状态.json"


def read(p): return json.loads(p.read_text(encoding="utf-8"))


def main():
    if (OUT/"scope_update_receipt.json").exists():
        raise RuntimeError("Scope already recorded; do not overwrite.")
    assert sha(GUIDE)==sha(OUT/"guide_before.md")
    assert sha(STATE)==sha(OUT/"execution_state_before.json")
    receipt=read(OUT/"receipt.json")
    for p,h in receipt["outputs_sha256"].items(): assert sha(BASE/p)==h
    a=read(OUT/"geometry_audit.json"); g=read(OUT/"geometry.json")
    now=datetime.now(timezone.utc).isoformat()
    text=GUIDE.read_text(encoding="utf-8")
    text=text.replace("一条单圈闭合长路径同时包含长直线、短直线、钝角、直角、锐角和圆弧段，",
                      "一条单圈闭合长路径以长短直线频繁交替为主体，同时包含钝角、直角、锐角和少量圆弧段，")
    start=text.index("### 已完成静态设计的单圈闭合长路径")
    end=text.index("### 案例选择与数据角色",start)
    section=f"""### 已完成静态设计的单圈闭合长路径

当前唯一主案例：`paper_evidence_review/composite_closed_alternating_case/reference.csv`。解析构成、全部角点、三列机制窗口和哈希见同目录`geometry.json`，静态审核与时间下界见`geometry_audit.json`，预览图为`path_preview.png`。

按用户最新要求，圆弧减少为两段，主体采用长直线—短直线的频繁切换，同时保持单圈凹凸轮廓。**当前为几何设计结果，尚未运行策略，不表示目标实验已经成功。**

| 项目 | 当前设计 |
|---|---|
| 拓扑 | 无自交的单圈闭合轮廓，完整走一圈；不是往返扫描 |
| 解析总长 | {a['analytic_length_mm']/1000:.6f} m |
| 沿参考100 mm/s匀速时间 | {a['reference_constant_100_time_s']:.6f} s，仅为参考值 |
| 允许切角后的完整时间下界 | **{a['ordered_gate_time_lower_bound_s']:.6f} s > 300 s**；还未计加减速和停稳 |
| 长短直线数量 | 43段直线：23段长直线(≥700 mm)，20段短直线(≤200 mm)，36处相邻长/短切换 |
| 直线实际长度范围 | 最短20 mm、最长{a['maximum_intrinsic_straight_mm']:.3f} mm；短段约20–114 mm |
| 直线总长度占比 | **{100*a['straight_length_fraction']:.3f}%**；总长度由直线主导 |
| 三类角 | 20个钝角、18个直角、2个锐角；40个角全部进入本案例统计 |
| 三列机制角点 | B：135°内角(2480,5920)；C：90°内角(3700,5920)；D：45°内角(5041.421356,6100) |
| 圆弧 | 仅两段90°圆弧，半径600、700 mm，合计约2.042 m |
| 起终点 | (1000,6000) mm；静止起动，完整一圈后制动停稳 |
| 静态核验 | 无自交、无180°方向歧义、无相邻门冲突；非相邻段最小保守间距20 mm |
| 共同参考 | {a['reference_points']}个点，直线采样步长≤25 mm，弧最大弦高误差≤0.002 mm |

长短段计数按登记的直线原语，起终点所在直线因起动/停稳位置分为首、尾段。分类在运行前固定，不因实验结果改阈值。长段用于持续巡航，频繁短段与拐角用于检验协调；统计必须同时报告全部角点与短段表现，不能只以大量长直线拉高巡航总量。

时间下界使用门截面的包围圆半径ε/cos(δ/2)，其中δ是相邻离散线段的航向转角。顺序门中心距离减去两端半径，给出完整有序执行的路程下界，再除以100 mm/s。因此超过300 s由几何保证，不是故意降速凑时长。

所有方法共用同一CSV与解析线/弧定义。弧离散误差不加到1.25 mm允差上，后续真实运动仍需解析几何、V/A/J、有序覆盖及制动停稳审核。圆弧连接处几何相切不等于实际运动已经合规。

827 mm短开放提案、四行往返草案和此前四圆弧闭合方案均保留为已被用户修正的设计，不继续用作主案例。当前几何在策略运行前固定；最终策略尚未冻结。

"""
    text=text[:start]+section+text[end:]
    text=text.replace("并已完成上述40.468 m闭合案例的静态几何审核", "并已完成上述36.015 m长短直线交替闭合案例的静态几何审核")
    text=text.replace("保守时间下界402.371 s", "保守时间下界358.428 s")
    text=text.replace("旧往返草案停止采用", "旧往返草案与四圆弧方案停止采用")
    text=text.replace("最新执行状态以本指南及闭合案例的范围更新收据为准。",
                      "四圆弧闭合方案时期的指南保存在`paper_evidence_review/composite_closed_alternating_case/guide_before.md`。最新执行状态以本指南及长短直线交替案例的范围更新收据为准。")
    text=text.replace("| **Fig.7 拐角降速与高速巡航 / Corner slowdown and high-speed cruising** | 完整速度曲线与几何区段对齐；",
                      "| **Fig.7 拐角降速与高速巡航 / Corner slowdown and high-speed cruising** | 完整速度曲线与长/短直线及圆弧区段对齐；")
    GUIDE.write_text(text,encoding="utf-8")
    write(OUT/"visual_qa.json",dict(reviewed_utc=now,status="geometry_preview_visually_checked",
        viewed_png_sha256=sha(OUT/"path_preview.png"),
        observations=["One closed contour, frequent small line changes visible; only two arcs highlighted.",
                      "Long/short counts and straight-length fraction are legible.",
                      "All three corner windows show the correct interior and heading-change angles with true XY aspect.",
                      "Length and strict >300 s lower bound are labelled as geometry, not a policy result."],
        manuscript_pdf_review=False))
    next_action=("Use the single closed alternating-line reference: 36.015378825 m, 23 long/20 short straight primitives, "
        "two arcs, complete-lap speed-only lower bound 358.427550701 s. Prepare a finite same-case implementation and "
        "comparison plan for reduced corner slowdown, increased high-speed cruise and full-time gains. "
        "No automatic third optimization round or sealed-test resume; superseded geometric drafts are preserved.")
    c=dict(created_utc=now,status="line_dominant_closed_geometry_designed_policy_goal_pending",
        user_scope="Frequent long/short straight alternation, fewer arcs, one closed contour and complete duration >300 s.",
        required_primary_cases=1,closed=True,lap_count=1,minimum_complete_duration_s=300,
        duration_requirement_strict=True,duration_lower_bound_s=a["ordered_gate_time_lower_bound_s"],
        analytic_length_mm=a["analytic_length_mm"],reference=str((OUT/"reference.csv").relative_to(BASE)),
        reference_sha256=sha(OUT/"reference.csv"),geometry=str((OUT/"geometry.json").relative_to(BASE)),
        geometry_sha256=sha(OUT/"geometry.json"),geometry_receipt=str((OUT/"receipt.json").relative_to(BASE)),
        geometry_receipt_sha256=sha(OUT/"receipt.json"),figure_preview=str((OUT/"path_preview.png").relative_to(BASE)),
        straight_count=43,long_straight_count=23,short_straight_count=20,arc_count=2,
        straight_length_fraction=a["straight_length_fraction"],short_long_neighbor_switches=a["short_long_neighbor_switches"],
        corner_counts=a["sharp_corner_counts"],mechanism_columns=g["mechanism_columns"],
        guide=GUIDE.name,guide_sha256=sha(GUIDE),goal_complete=False,new_policy_evaluations=0,
        new_physics_ticks=0,new_training_updates=0,algorithm_rounds_used=2,algorithm_rounds_cap=2,
        method_frozen=False,formal_model_adopted=False,next_action=next_action,
        superseded_geometry_preserved=["paper_evidence_review/composite_long_case","paper_evidence_review/composite_closed_case"])
    write(OUT/"continuation.json",c)
    sources=["record_alternating_closed_design.py","prepare_alternating_closed_design.py",
             "paper_evidence_review/composite_closed_case/scope_update_receipt.json",str((OUT/"receipt.json").relative_to(BASE))]
    update=dict(created_utc=now,status="completed_alternating_line_geometry_guide_alignment",
        sources_sha256={p:sha(BASE/p) for p in sources},
        outputs_sha256={str(p.relative_to(BASE)):sha(p) for p in [GUIDE,OUT/"visual_qa.json",OUT/"continuation.json"]},
        previous_guide_preserved_at=str((OUT/"guide_before.md").relative_to(BASE)),
        previous_guide_sha256=sha(OUT/"guide_before.md"),goal_complete=False,new_training_or_simulation=False)
    write(OUT/"scope_update_receipt.json",update)
    latest=dict(path=str((OUT/"continuation.json").relative_to(BASE)),sha256=sha(OUT/"continuation.json"),
                receipt=str((OUT/"scope_update_receipt.json").relative_to(BASE)),
                receipt_sha256=sha(OUT/"scope_update_receipt.json"),status=c["status"])
    state=read(STATE)
    state.update(updated_utc=now,goal_complete=False,next_action=next_action,next=next_action,
        current_action="alternating_long_short_closed_geometry_completed_policy_goal_pending",
        current="One closed contour, 36.015 m, 43 straight primitives and two arcs; no policy rollout yet.",
        paper_closed_case_design=c,latest_execution_findings=c)
    single=state["paper_single_case_scope"]
    single.update(updated_utc=now,status=c["status"],closed=True,minimum_complete_duration_s=300,
        geometry_proposal=dict(status="static_design_completed_no_policy_execution",reference=c["reference"],
            reference_sha256=c["reference_sha256"],closed=True,analytic_length_mm=a["analytic_length_mm"],
            time_lower_bound_s=a["ordered_gate_time_lower_bound_s"],long_straight_count=23,short_straight_count=20,arc_count=2),
        guide_sha256=sha(GUIDE),next_action=next_action)
    resume=state["latest_resume_20260919"]
    resume.update(latest_execution_utc=now,next_action=next_action,
        learning_status="no_learning_running_alternating_closed_geometry_designed_goal_pending")
    resume["paper_evidence_review"].update(latest_continuation=latest,next_action=next_action,
        current_goal_success_witness_verified=False,current_guide_sha256=sha(GUIDE),primary_case=c["reference"],
        primary_case_closed=True,primary_case_minimum_duration_s=300,
        primary_case_static_time_lower_bound_s=a["ordered_gate_time_lower_bound_s"])
    for boundary in [state["paper_research_boundary"],resume["paper_research_boundary"]]:
        boundary.update(latest_user_clarification=c["user_scope"],required_primary_geometry_cases=1,
            required_case_closed=True,required_complete_duration_greater_than_s=300,
            frequent_long_short_straight_alternation_required=True,
            all_three_historical_long_paths_must_succeed=False,goal_and_writing_guide=GUIDE.name,
            guide_sha256=sha(GUIDE),latest_scope_receipt=latest["receipt"],latest_scope_receipt_sha256=latest["receipt_sha256"])
    write(STATE,state)
    write(OUT/"state_update_receipt.json",dict(created_utc=now,state_before_sha256=sha(OUT/"execution_state_before.json"),
        state_after_sha256=sha(STATE),scope_receipt_sha256=sha(OUT/"scope_update_receipt.json")))
    for mapping in [update["sources_sha256"],update["outputs_sha256"]]:
        for p,h in mapping.items(): assert sha(BASE/p)==h
    for p in [GUIDE,STATE]:
        assert not p.read_bytes().startswith(b"\xef\xbb\xbf")
        assert "\ufffd" not in p.read_text(encoding="utf-8")
    print(json.dumps(dict(guide_updated=True,case_closed=True,long_lines=23,short_lines=20,arcs=2,
        duration_lower_bound_s=a["ordered_gate_time_lower_bound_s"],new_simulation=False)))


if __name__=="__main__": main()
