"""Bind reviewed geometry, flexible timing scope and the living writing guide."""
import json
from datetime import datetime, timezone
from pathlib import Path

from design_composite_long_case import sha, write

BASE = Path(__file__).resolve().parent
OUT = BASE / "paper_evidence_review/natural_closed_case"
GUIDE = BASE / "PPO_article_研究目标_图表与写作指南.md"
STATE = BASE / "当前执行状态.json"


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def rel(p):
    return p.relative_to(BASE).as_posix()


def main():
    if (OUT / "scope_update_receipt.json").exists():
        raise RuntimeError("Scope already recorded; preserve and inspect.")
    assert sha(GUIDE) == sha(OUT / "guide_before.md")
    assert sha(STATE) == sha(OUT / "execution_state_before.json")
    original = read(OUT / "receipt.json")
    for path, digest in original["outputs_sha256"].items():
        assert sha(BASE / path) == digest
    a, g = read(OUT / "geometry_audit.json"), read(OUT / "geometry.json")
    r = read(OUT / "geometry_reporting.json")
    assert r["reference_sha256"] == sha(OUT / "reference.csv")
    now = datetime.now(timezone.utc).isoformat()
    verified_sources = {
        "论文研究边界与续接指令_20260920.md": "2c10f01c27fa7a9f3531b3b3a2bea0162fdba0d3325f5a844fa5bb788f0c8a20",
        "canonical_gate_band_worst_guard_joint_learning_20260920/shadow_training_policy.pt": "25abeb3b88e45efac1d8311669f45a3fcaa26dc785d9bfbd216734b7f5a9c173",
        "canonical_gate_band_shape_cruise_completion_feedback_20260918/candidate_0.5.pt": "86cdfe3e1bf550a34332bab2678fed9a12e3a33611173ba57a75a1818daf5a86",
    }
    for path, digest in verified_sources.items():
        assert sha(BASE / path) == digest
    angles = "、".join(f"{v:.1f}°" for v in r["distinct_included_angles_deg_rounded_0_1"])
    text = GUIDE.read_text(encoding="utf-8")

    def replace(old, new):
        nonlocal text
        assert old in text, old
        text = text.replace(old, new)

    replace("更新日期：2026-09-21。根据用户最新要求修订，沿用PPO_article，不新增项目编号。",
            "更新日期：2026-09-21。按最新要求采用自然凹凸、多角度、长短直线频繁交替的单圈长路径；控制周期由算法设计决定。沿用PPO_article，不新增项目编号。")
    replace("同时包含钝角、直角、锐角和少量圆弧段，", "同时包含多种不同数值的钝角、直角、锐角和少量圆弧段，")
    replace("### 不变的硬约束", "### 保留的物理硬约束")
    replace("- 20 ms策略决策、1 ms执行。\n", "")
    replace("- 完整制动，至少20 ms停稳；完整时间必须包含二者。",
            "- 完整制动，至少20 ms的物理停稳时长；完整时间必须包含二者。20 ms在这里是持续时间，不是策略周期，也不等于20个采样点。")
    marker = "## 二、唯一主案例怎样定义"
    timing = """### 控制周期按算法需要设计

**20 ms策略决策、1 ms执行是旧实现的设置，不再是不可修改的研究约束。** 可以选择其他固定周期、多速率或事件触发执行，以能否形成目标运动和可信比较为依据。当前只改变研究设计权限，尚未选择新周期，也未修改现有运行器或模型。

| 时间层 | 可设计内容 | 论文必须披露的内容 |
|---|---|---|
| 策略决策 | 固定周期、条件触发、可变间隔 | 真实决策时间戳、动作更新次数、事件规则或周期 |
| 执行更新 | 动作保持、插值、多速率约束执行 | 执行步长或连续命令定义、插值阶次、滤波和制动规则 |
| 物理与数值审核 | 解析极值/区间界，或足够细的时间审核并做必要的步长收敛检查 | 审核分辨率、离散误差及峰值检查；不能靠降采样漏掉允差和V/A/J超限 |

不同时间层分别记录。若使用连续插值命令，应检查区间内部而不只检查控制节点；若以积分仿真近似实际运动，应核查关键峰值和完成判定随步长细化是否稳定。1 ms可作为已有实现的检查起点，但不是自动的充分保证。几何参考点的25 mm间距也不是执行或审核时间步长。

时间设置改变时，奖励积分、折扣与GAE的有效时间尺度、训练/前瞻时域、滤波常数、终端制动和停稳判据都应按物理时间核对，不能只改一个dt后沿用不等价的步数参数。若重新训练或搜索，仍计入明确的有限实验预算。

公平比较共用任务、几何和物理限值。主方法与固定前瞻、动作消融尽量共用时序和执行器，以识别动作学习的作用；S2按其实际规划方式运行。若比较方法使用不同控制时序，明确这是完整系统比较，并报告决策次数、推理/规划成本和实际更新规则，不把全部收益归因于三动作。无需另设工业实时认证门槛。

"""
    replace(marker, timing + marker)
    replace("角度类别按两条相邻直线的路径内角β命名，实际航向转角为δ，满足β=180°−|δ|。左/右转用δ的正负号另行表示。",
            "角度类别按入边反向射线与出边射线的较小夹角β命名，实际航向转角为δ，满足β=180°−|δ|。旧字段interior_angle_deg沿用这个定义；它不是凹多边形可能大于180°的区域内角。左/右转用δ的正负号另行表示。宏观凹槽深度与局部转向强度是两个量。")
    replace("| 机制图列 | 内角β的提案 | 航向转角绝对值 | 解释 |", "| 机制图代表列 | 夹角β | 航向转角绝对值 | 解释 |")
    start = text.index("### 已完成静态设计的单圈闭合长路径")
    end = text.index("### 案例选择与数据角色", start)
    geometry_section = f"""### 已完成静态设计的自然复杂闭合长路径

当前唯一主案例：`paper_evidence_review/natural_closed_case/reference.csv`。解析线/弧、全部角点及三列机制窗口见同目录`geometry.json`；几何审核和时间下界见`geometry_audit.json`。正式设计预览为`trajectory_case.png`，完整角度分布为`angle_coverage.png`，配套说明为`轨迹案例.md`。

主体为不对称深凹槽、明显凸起和斜向长边，保留两段圆弧；上侧约1800 mm深的锐角凹槽、约1300 mm高的凸起，下侧不等深折线与80 mm短平台提供自然的整体复杂度。**当前为参考几何设计，尚未运行策略，完整目标实验仍待实现。**

| 项目 | 当前设计与静态审核 |
|---|---|
| 拓扑 | 一条无自交的闭合轮廓，完整走一圈 |
| 解析总长 | **{a['analytic_length_mm']/1000:.6f} m** |
| 沿参考100 mm/s匀速时间 | {a['reference_constant_100_time_s']:.6f} s，仅为参考值 |
| 允许切角后的完整时间下界 | **{a['ordered_gate_time_lower_bound_s']:.6f} s > 300 s**；未计加减速与停稳 |
| 真实直线段 | 合并共线分段后**36段：20长＋16短**；32处相邻长短切换 |
| 长短定义与长度范围 | 本案例短段≤700 mm、长段>700 mm；真实直线段80.000–2262.742 mm |
| 直线长度占比 | **{100*a['straight_length_fraction']:.3f}%** |
| 全部拐角 | 34个：29钝角、3直角、2锐角；按0.1°显示共有26种不同夹角 |
| 角度范围 | 夹角45.000°–175.679°；对应航向变化绝对值4.321°–135.000° |
| 代表机制角 | B：β135°，(3500,7400)；C：β90°，(5100,7400)；D：β45°，(7495.584412,7200) |
| 圆弧 | 两段90°弧，半径600、800 mm，切向连接相邻直线 |
| 起终点 | (1800,7800) mm；静止起动，完成一圈后制动停稳 |
| 静态核验 | 无自交、无180°方向歧义、无相邻门冲突；非相邻原语保守间距≥80 mm |
| 共同参考 | 2364个点，直线采样步长≤25 mm，弧最大弦高误差≤0.002 mm |

**路径不只包含135°、90°、45°。** 全部不同夹角（显示到0.1°）为：{angles}。`corner_catalogue.csv`逐一列出34个角的准确夹角、带符号航向变化、XY位置和参考弧长。左右转均存在；角度分布不均匀，不声称均衡覆盖每个角度区间。

原始geometry.json登记38个直线原语，其中起终点拆开同一直线，另有一个共线连接；绘图与论文按合并后的36个真实直线段报告，详见`geometry_reporting.json`。原始几何和收据不改写。长短阈值在策略运行前确定；“短”表示相对几何长度，不自动代表全部短段均受动态制动距离限制。全部角点、长短段和圆弧分别统计，避免仅靠长直线拉高巡航总量。

时间下界使用有序门截面的包围圆半径ε/cos(δ/2)：相邻必经门中心距离减去两端半径，求和后除以100 mm/s。因此超过300 s由几何和完整有序覆盖保证，与策略周期无关。实际允差切角不会把该案例缩成300 s以内。

所有方法共用同一参考CSV与解析线/弧。弧离散误差不加到1.25 mm允差上；实际运动仍需检查解析几何、平面V/A/J、有序覆盖与制动停稳。静态相切和无自交不等于已经验证运动可行。

先前短开放、往返、四圆弧和矩形微台阶草案全部保留为已被用户修正的设计。新参考和角点窗口在策略运行前登记，最终方法尚未冻结。

"""
    text = text[:start] + geometry_section + text[end:]
    replace("三类角全部列出；最低速度与持续时间共同判断，不能只看一个瞬时谷值",
            "34个角点全部列出，并按三类归纳；最低速度与持续时间共同判断，不能只看一个瞬时谷值")
    replace("完整单圈闭合参考；标记长/短段、三类内角、圆弧半径、方向与超过300 s的时间下界；",
            "完整自然闭合参考；标记长/短段、多种夹角、圆弧半径、方向与超过300 s的时间下界；附34角/26种夹角分布；")
    replace("观测/记忆→三动作→引导、约束与制动；20 ms/1 ms；PPO、拟合和参数搜索来源分列",
            "观测/记忆→三动作→引导、约束与制动；标明最终实际决策、执行和审核时序；PPO、拟合和参数搜索来源分列")
    replace("三类角低速时长/速度亏损；≥99高速区间、最长连续平台与高速覆盖比例",
            "全部34角的低速时长/速度亏损，按准确夹角与类别展示；≥99高速区间、最长连续平台与高速覆盖比例")
    replace("同一主方法的20 ms动作阶梯", "同一主方法按真实决策时刻绘制；保持/插值形式据实")
    replace("1 ms执行数据计算的有符号曲率", "由真实执行数据及足够细的审核计算有符号曲率")
    replace("每列对应Fig.5中的同一真实角点和同一段完整执行记录。",
            "三个代表列之外，其余31个角点也进入Fig.7与表3的完整统计；完整角度图册可放补充材料。新增代表窗口必须依据预先声明的几何规则，不从结果中只挑好角。\n\n每列对应Fig.5中的同一真实角点和同一段完整执行记录。")
    replace("所有1 ms数据、加速、制动和停稳保留；V/A/J用平面范数，不能裁去峰值或对XY作美化平滑。",
            "保留全部真实时间戳、动作更新和执行记录，覆盖加速、制动和停稳；按已声明的物理审核方式检查平面V/A/J及几何峰值。制图降采样不能取代审核，不能裁去峰值或对XY作美化平滑。")
    replace("并已完成上述36.015 m长短直线交替闭合案例的静态几何审核", "并已完成上述44.421 m自然复杂、多角度闭合案例的静态几何审核")
    replace("已完成闭合长路径设计与静态审核，主案例和三列角点已固定；旧往返草案与四圆弧方案停止采用。",
            "已完成自然闭合长路径与全部34角的静态审核，代表窗口已登记；使用36个真实直线段的计数口径，旧草案保留。")
    replace("核对同案例执行、基线接口、解析圆弧距离与预算方案；保持共同参考和窗口，不重跑已完成实验。",
            "核对同案例执行、基线接口、解析圆弧距离与预算方案；按算法需要确定决策/执行/审核时序，保持共同参考和窗口，不重跑已完成实验。")
    replace("本次仅修订目标、图表和写作指南，不启动实验。",
            "本次仅完成新几何、角度覆盖、控制时序研究边界及写作指南，不启动模型实验。允许调整控制周期不等于自动增加无限训练轮数。")
    replace("保守时间下界358.428 s。", "34个角点、26种夹角，保守时间下界442.726 s。")
    replace("1.25 mm、20/1 ms、平面V/A/J及制动停稳全部通过。",
            "1.25 mm、平面V/A/J及制动停稳全部通过；实际控制时序公开，时间审核足够。")
    replace("最新执行状态以本指南及长短直线交替案例的范围更新收据为准。",
            "矩形微台阶时期的指南及状态保存在`paper_evidence_review/natural_closed_case/guide_before.md`与`execution_state_before.json`。最新范围以本指南和该目录的scope_update_receipt.json为准；用户允许改变控制周期的指令覆盖旧文档中固定20/1 ms要求，历史模型的真实时序记录保持原样。")
    replace("【工具调用简报】本地指南、图合同与收据复核；更新指南与状态并核验UTF-8无BOM/哈希。未新增训练、策略评价或物理仿真；实时额度周剩余48%、已有重置0，未购买或使用重置。",
            "【工具调用简报】本地conda PPO完成静态几何、全角度与共线计数审核；实际查看预览图，核对模型及历史边界哈希，未发现PPO实验进程。更新同名指南与状态，UTF-8无BOM；未新增训练、策略评价或物理仿真。Codex实时额度本次查询周剩余47%、已有重置0，未购买或使用重置。")
    for stale in ["36.015", "358.428", "358.427", "20/1 ms、", "20 ms动作阶梯", "所有1 ms数据"]:
        assert stale not in text, stale
    GUIDE.write_text(text, encoding="utf-8")

    case_doc = f"""# PPO_article：自然复杂、多角度的单圈长轨迹案例

本案例使用不对称凸起、深凹槽、长短直线和两段圆弧构成一个闭合圈。以下是参考几何设计，尚无策略运行结果。

![闭合参考轨迹]({(OUT/'trajectory_case.png').as_posix()})

| 特征 | 已核对的设计 |
|---|---|
| 总长 | 44.420514 m |
| 完整时间下界 | 442.725521 s；使用100 mm/s上限、有序覆盖和1.25 mm允差，尚未计加减速及停稳 |
| 直线 | 共线合并后36段：20长、16短；32处相邻长短切换 |
| 直线长度 | 短段≤700 mm、长段>700 mm；范围80.000–2262.742 mm；直线占总长95.049% |
| 拐角 | 34个角，按0.1°显示26种不同夹角；2锐角、3直角、29钝角 |
| 圆弧 | 仅两段，半径600/800 mm，各90°，与相邻直线相切 |
| 起终点 | (1800,7800) mm；运动需从静止开始，走完一圈并制动停稳≥20 ms |
| 静态几何 | 无自交、无门冲突、无180°方向歧义；不等于实际运动已通过审核 |

上部包含明显凸起和约1800 mm深的锐角凹槽；下部是深度不同的折线凹槽与短平台。夹角β是入边反向射线与出边射线的较小夹角，β=180°−|δ|；凹多边形区域内角另有定义，不能混用。

**不只三个固定角度。** 全部不同夹角为：{angles}。存在左转与右转，角度分布不均衡。三个机制代表为135°/90°/45°，对应航向变化+45°/+90°/+135°；其他31个角全部进入统计。

![完整角度分布]({(OUT/'angle_coverage.png').as_posix()})

`corner_catalogue.csv`保存34个角的准确角度、带符号转角、坐标和参考弧长。`geometry.json`保存原始38个直线原语及两条弧；其中一个共线连接和起终点拆分不构成新几何直线。`geometry_reporting.json`记录合并后的36段与统计依据。本说明与trajectory_case.*采用合并口径；初始路径设计说明及path_preview.*保留原语计数，不作为最终统计口径。

20 ms策略决策/1 ms执行已按用户要求改为可设计项，可采用其他周期、多速率或事件触发方式。必须说明实际决策时间戳、动作保持或插值、执行更新和物理审核方式；几何与V/A/J不能因低采样率被漏检。1.25 mm、100/2000/62500平面V/A/J上限、完整有序覆盖、完整制动及至少20 ms物理停稳仍保留。

所有方法使用相同reference.csv与解析几何。局部展示窗40×40 mm，机制统计窗在参考弧长上每侧最多10 mm并按相邻段裁剪；圆弧弦高误差不额外放宽允差。短段分类是几何长度分类，不等于已证明动力学受限。

研究目标是由同一学得策略在完整一圈中协调转向、进给、前瞻，实现较少拐角降速、更多接近100 mm/s的持续巡航，以及相对可信对照的完整时间收益。当前只完成几何和论文方案，不能据此宣称目标已经实现。

研究目标、九幅主图、三张表、20–25页英文写作和完成标准详见[{GUIDE.stem}]({GUIDE.as_posix()})。原始收据和模型保留，未运行新训练、拟合、搜索或物理仿真。
"""
    (OUT / "轨迹案例.md").write_text(case_doc, encoding="utf-8")
    write(OUT / "visual_qa.json", dict(reviewed_utc=now, status="reviewed_design_previews",
        reviewed_pngs_sha256={rel(OUT / n): sha(OUT / n) for n in ["path_preview.png", "trajectory_case.png", "angle_coverage.png"]},
        observations=["Final preview has one irregular closed loop, macroscopic recesses/projections and two arcs.",
                      "Final counts merge collinear splits: 36 spans, 20 long/16 short, 32 direct switches.",
                      "Multiple additional included angles appear on the overall contour; the coverage plot shows all 34 corners and 26 distinct displayed angles.",
                      "Three representative XY windows use equal aspect and a labelled 1.25 mm corridor.",
                      "Geometry lower bound is labelled; no policy performance is depicted."],
        manuscript_pdf_review=False, submission_figure_review=False))
    user_scope = ("One natural irregular closed long contour with frequent long/short straight transitions, many different "
                  "angles within obtuse/right/acute categories, two arcs, and complete duration >300 s. "
                  "Policy decision and execution periods may be designed to achieve the objective; historical 20 ms/1 ms are not hard constraints.")
    timing_scope = dict(control_periods_are_design_choices=True, actual_new_periods_selected=False,
        historical_policy_decision_ms=20, historical_execution_ms=1,
        historical_records_relabelled=False, actual_runtime_modified=False,
        audit_resolution_separate_from_control_period=True, physical_rest_minimum_ms=20,
        timing_fairness="Share timing/executor for action ablations where feasible; label different-timing comparisons as system-level and report actual computation.")
    next_action = ("Use the registered natural closed case (44.420514 m, 36 maximal straight spans, 34 corners/26 displayed angles, "
                   "two arcs; complete-lap speed-only lower bound 442.725521 s). Prepare a finite implementation/comparison plan "
                   "including algorithm-appropriate decision, execution and audit timing. Achieve complete learned-policy motion "
                   "and measured corner-speed, cruise and full-time gains before claiming goal success. "
                   "Do not automatically launch a third optimization round or resume the exposed sealed set.")
    c = dict(created_utc=now, status="natural_multi_angle_closed_geometry_and_guide_complete_policy_goal_pending",
        user_scope=user_scope, required_primary_cases=1, closed=True, lap_count=1,
        minimum_complete_duration_s=300, duration_requirement_strict=True,
        duration_lower_bound_s=a["ordered_gate_time_lower_bound_s"], analytic_length_mm=a["analytic_length_mm"],
        reference=rel(OUT / "reference.csv"), reference_sha256=sha(OUT / "reference.csv"),
        geometry=rel(OUT / "geometry.json"), geometry_sha256=sha(OUT / "geometry.json"),
        geometry_receipt=rel(OUT / "receipt.json"), geometry_receipt_sha256=sha(OUT / "receipt.json"),
        geometry_reporting=rel(OUT / "geometry_reporting.json"), geometry_reporting_sha256=sha(OUT / "geometry_reporting.json"),
        case_description=rel(OUT / "轨迹案例.md"), figure_preview=rel(OUT / "trajectory_case.png"),
        angle_coverage=rel(OUT / "angle_coverage.png"), straight_count=36,
        long_straight_count=20, short_straight_count=16, straight_count_convention=r["count_convention"],
        registered_straight_primitives=38, short_long_neighbor_switches=32,
        straight_length_fraction=a["straight_length_fraction"], corner_counts=a["sharp_corner_counts"],
        total_corner_count=34, distinct_included_angle_count=26,
        distinct_included_angles_deg_rounded_0_1=r["distinct_included_angles_deg_rounded_0_1"],
        arc_count=2, mechanism_columns=g["mechanism_columns"], all_corners_in_statistics=True,
        control_timing_scope=timing_scope, guide=GUIDE.name, guide_sha256=sha(GUIDE),
        goal_complete=False, method_frozen=False, formal_model_adopted=False,
        new_policy_evaluations=0, new_physics_ticks=0, new_training_updates=0,
        algorithm_rounds_used=2, algorithm_rounds_cap=2,
        additional_algorithm_round_authorized=False, next_action=next_action,
        superseded_geometry_preserved=["paper_evidence_review/composite_long_case", "paper_evidence_review/composite_closed_case",
                                       "paper_evidence_review/composite_closed_alternating_case"])
    write(OUT / "continuation.json", c)
    source_paths = ["record_natural_closed_design.py", "render_natural_case_report.py", "prepare_natural_closed_design.py",
                    "design_natural_closed_case.py", "paper_evidence_review/composite_closed_alternating_case/scope_update_receipt.json",
                    rel(OUT / "receipt.json")]
    output_paths = [GUIDE] + [OUT / name for name in ["轨迹案例.md", "geometry_reporting.json", "corner_catalogue.csv",
                    "visual_qa.json", "continuation.json", "trajectory_case.png", "trajectory_case.svg", "trajectory_case.pdf",
                    "angle_coverage.png", "angle_coverage.svg", "angle_coverage.pdf"]]
    update = dict(created_utc=now, status="completed_natural_multi_angle_case_and_flexible_timing_guide",
        sources_sha256={**verified_sources, **{p: sha(BASE / p) for p in source_paths}},
        outputs_sha256={rel(p): sha(p) for p in output_paths},
        previous_guide_preserved_at=rel(OUT / "guide_before.md"), previous_guide_sha256=sha(OUT / "guide_before.md"),
        original_geometry_receipt_unchanged=True, geometry_only=True, goal_complete=False, new_training_or_simulation=False,
        quota_check=dict(weekly_remaining_percent=47, reset_credits=0, reset_used=False, purchase_made=False),
        process_check="No python/conda process with PPO or joint_policy in its command line was found before report generation.")
    write(OUT / "scope_update_receipt.json", update)
    latest = dict(path=rel(OUT / "continuation.json"), sha256=sha(OUT / "continuation.json"),
                  receipt=rel(OUT / "scope_update_receipt.json"), receipt_sha256=sha(OUT / "scope_update_receipt.json"), status=c["status"])
    state = read(STATE)
    state.update(updated_utc=now, goal_complete=False, next_action=next_action, next=next_action,
        current_action=c["status"], current="Natural closed reference: 44.421 m, 36 true straight spans, 34 corners/26 displayed angles, two arcs; no policy rollout yet.",
        paper_closed_case_design=c, latest_execution_findings=c, paper_control_timing_scope=timing_scope)
    single = state["paper_single_case_scope"]
    single.update(updated_utc=now, status=c["status"], latest_user_instruction=user_scope,
        hard_constraints_unchanged=False, physical_limits_unchanged=True,
        hard_constraints_change_explanation="Latest user instruction makes fixed 20 ms/1 ms timing optional; physical constraints and >=20 ms rest duration remain.",
        control_timing_scope=timing_scope, closed=True, minimum_complete_duration_s=300,
        geometry_proposal=dict(status="static_design_completed_no_policy_execution", reference=c["reference"], reference_sha256=c["reference_sha256"],
            closed=True, analytic_length_mm=a["analytic_length_mm"], time_lower_bound_s=a["ordered_gate_time_lower_bound_s"],
            long_straight_count=20, short_straight_count=16, arc_count=2, corner_count=34, distinct_displayed_angle_count=26),
        guide_sha256=sha(GUIDE), next_action=next_action)
    single["mechanism_figure"].update(angle_convention="smaller included angle beta = 180 degrees - abs(signed heading change delta); not polygon reflex angle",
        representative_columns_not_exhaustive=True, all_corner_statistics=34, distinct_displayed_angle_count=26)
    resume = state["latest_resume_20260919"]
    resume.update(latest_execution_utc=now, next_action=next_action, learning_status="no_learning_running_natural_multi_angle_geometry_and_timing_scope_updated")
    resume["paper_evidence_review"].update(latest_continuation=latest, next_action=next_action,
        current_goal_success_witness_verified=False, current_guide_sha256=sha(GUIDE), primary_case=c["reference"],
        primary_case_closed=True, primary_case_minimum_duration_s=300,
        primary_case_static_time_lower_bound_s=a["ordered_gate_time_lower_bound_s"], primary_case_corner_count=34,
        control_timing_scope=timing_scope)
    for boundary in [state["paper_research_boundary"], resume["paper_research_boundary"]]:
        boundary.update(latest_user_clarification=user_scope, user_clarification_date="2026-09-21",
            required_primary_geometry_cases=1, required_case_closed=True,
            required_complete_duration_greater_than_s=300, frequent_long_short_straight_alternation_required=True,
            multiple_distinct_angles_required=True, representative_angle_columns_are_not_full_angle_set=True,
            all_three_historical_long_paths_must_succeed=False, goal_and_writing_guide=GUIDE.name,
            guide_sha256=sha(GUIDE), latest_scope_receipt=latest["receipt"], latest_scope_receipt_sha256=latest["receipt_sha256"],
            control_periods_are_design_choices=True, historical_fixed_20ms_1ms_requirement_superseded=True,
            physical_rest_minimum_ms=20, hard_physical_constraints_unchanged=True,
            control_timing_scope=timing_scope,
            source_priority="Latest user instructions and current guide override earlier multiple-case and fixed 20 ms/1 ms defaults. Physical limits, ordered completion, braking and >=20 ms physical rest remain; immutable historical records retain actual past settings.")
    write(STATE, state)
    write(OUT / "state_update_receipt.json", dict(created_utc=now,
        state_before_sha256=sha(OUT / "execution_state_before.json"), state_after_sha256=sha(STATE),
        scope_receipt_sha256=sha(OUT / "scope_update_receipt.json")))
    for mapping in [update["sources_sha256"], update["outputs_sha256"], original["outputs_sha256"]]:
        for path, digest in mapping.items():
            assert sha(BASE / path) == digest
    for p in [GUIDE, STATE, OUT / "轨迹案例.md", OUT / "corner_catalogue.csv"]:
        assert not p.read_bytes().startswith(b"\xef\xbb\xbf")
        assert "\ufffd" not in p.read_text(encoding="utf-8")
    print(json.dumps(dict(guide_updated=True, state_updated=True, reference_unchanged=True,
        straight_spans=36, corners=34, distinct_angles=26, timing_design_allowed=True,
        new_training_or_simulation=False, goal_complete=False), ensure_ascii=False))


if __name__ == "__main__":
    main()
