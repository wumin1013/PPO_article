"""Record the user's changed research priority; no training or simulation."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


BASE = Path(r"D:\PPO_article_work\论文项目\joint_policy_20260915_v1")
OUT = Path(r"D:\PPO_article_joint_runs\natural_closed_resume")
RESEARCH = OUT / "compact_dense_research"
GUIDE = BASE / "PPO_article_研究目标_图表与写作指南.md"
LOG = RESEARCH / "research_rounds_20260922.md"
STATES = [
    BASE / "当前执行状态.json",
    OUT / "live_research_summary.json",
    OUT / "all34_dominance_research/active_research_checkpoint.json",
    RESEARCH / "current_progress.json",
]
now = datetime.now(timezone.utc)
stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
backup_dir = RESEARCH / "state_history" / stamp
backup_dir.mkdir(parents=True, exist_ok=False)
backups = []
for path in [*STATES, GUIDE, LOG]:
    raw = path.read_bytes()
    destination = backup_dir / path.name
    destination.write_bytes(raw)
    backups.append({"original": str(path), "path": str(destination),
                    "sha256": hashlib.sha256(raw).hexdigest()})

target_text = """# PPO_article：当前有效研究目标与论文修改范围

更新日期：2026-09-22。依据用户最新指示：允许加入有效规则，优先使用此前案例上效果最好的主方法；以原18页稿为唯一基稿，主要修改实验。以下目标取代旧指南中冲突的要求；旧文本保留在本文件后部作为历史。项目与分支不变，未冻结、未完成。

1. **恢复历史最佳方法作为主线。** 优先核验 `joint_transit_retained_seed922222_full` 的模型、源码、规则、参数和轨迹绑定。历史原长路径完整时间444.302s，S2为445.185s；这些是已存结果，不是本轮重跑。允许保留或增加有证据支持的有效规则，不再把去规则或从零纯PPO作为前置条件。紧凑模型研究完整保留，不自动续训配对课程支线。

2. **保留原案例，使用已固定的新增密集案例。** 原路径44.420514m及全部34角不变；新增案例为已登记的1438.009122984mm、84角开发案例。不得根据方法输赢更改其几何或统计窗口，不称为未见测试。原案例验证完整运动与长直线巡航，新案例检验频繁转弯下的效率。

3. **保留已经约定的效率目标。** 原案例须完整成功、时间可信优于S2及历史强Fixed，全部34角速度亏损不高于二者，零亏损允许并列。5%要求适用于新增密集案例：对可信S2及真实两动作方法的完整时间均降低至少5%。当前新案例S2为19.957s，对它的5%门槛是18.95915s；最终同时满足两动作对照对应的门槛。原长路径不承诺物理下界排除的5%收益。原34角指标和窗口不改。

4. **核心消融采用真实两动作模型。** 转向与进给由独立训练的两个输出产生，去掉可学习的前瞻动作，保留可比的公共观测、执行链和适用规则；给予合理参数选择、训练和检查点选择机会。不能仅锁住三动作模型的一个输出后冒充重训练消融。影响比较的实现定义保留在附录或复现材料中，正文不必增加专门小节。历史强Fixed作为已有比较证据，不替代该消融；最多三项关键消融。已有D=4mm、25.334s的两动作开发结果必须保留。

5. **用完整运动证据判断外摆。** 保留允许外摆与禁止外摆的可比研究，比较完整时间、实际路径、曲率变化和V/A/J，不预设零外摆必然最好。允许在1.25mm允差内选择有利路径，不把小幅外摆自动判为失败；禁止外摆组使用额外形状约束并明确记录。已有禁止组未完整成功，尚不能证明任何一组的最终优越性或物理不可行性。

6. **状态简化服从当前方法优先级。** 先核对历史方法实际起作用的观测、记忆与规则，不为减少维度而直接替换已有效的方法。用户此前要求降维；关于其是否仍为最终硬要求的异步澄清尚无回答，不能静默宣布该要求已取消。在澄清前，核验历史最佳方法可以继续，不宣称72维最优、14维最小或已完成降维验收。

7. **原18页稿的格式与结构优先。** 唯一基稿是坚果云投稿包中的18页匿名PDF及已核验一致源码、参考文献和图件。保留原题目，只有实际贡献不匹配才提出最小修改建议；沿用引言、问题定义、方法、实验和结论的原结构及已有讨论安排。重点重写原实验设置、主要比较、消融结果，非实验部分仅作真实实现和结果一致性所必需的最小修改。规则无需逐条展开或新增主文小节；会影响轨迹、时间及贡献归因的组成须作必要简述，细节可放附录，不能把规则收益全部归为PPO。KCM算法和训练参数放附录，正文不单列PPO训练小节。实验不讨论泛化，突出实测支持的优势，不杜撰或掩盖与比较结论直接有关的事实。

8. **图件与最终验收。** 原前三幅图文件不动，需要更改只提出建议由用户执行；全稿仍为九幅主图，实验图绑定真实数据。轨迹采用135°、90°、45°典型局部放大，机制图为三列六行（转向、进给、前瞻、实际航向、曲率、速度），另列全部34角统计。案例图标注最长/最短直线、最大/最小拐角和典型圆弧，采用10mm十字网格；删去图4b多余10mm文字，避免图7密集填充。默认三个独立训练种子，历史条件搜索不冒充独立PPO重复；同时进行1ms/0.5ms核验并检查收益超过数值与执行分辨率。若最终仅有单模型证据，说明重复性限制，不冒充达成三种子验收。只有完整成功、真实三动作机制和相对S2及两动作的时间目标成立后才冻结，随后完成20–25页英文PDF、引用核验和逐页检查。

共同物理边界：1.25mm单侧欧氏允差；平面向量V/A/J上限100/2000/62500；完整有序覆盖、完整制动及至少20ms物理停稳；高速巡航主阈值≥99mm/s。不得削弱S2、全局人为降速、缩短任务或删除不利终段。PPO训练、离线拟合、参数优化和诊断探针分别记录。

操作边界：沿用 `D:\\PPO_article_work\\论文项目\\joint_policy_20260915_v1` 和 `codex/complex-path-adaptive-preview`，不新增项目编号、不派子代理、不切换或重置分支、不删除/清理/提交/推送。每轮新增训练或仿真≤30分钟，事前登记证据缺口、方案、预算、采用和停止条件。conda PPO，UTF-8无BOM，中间材料在非同步D盘。实时额度只查询，不购买；剩余≤2%且存在重置信用时也须取得该次确认。

下一步先核对历史最佳方法的完整来源与两案例适配条件，再登记密集案例复现实验。当前没有活动训练；本次仅更新目标与状态，新增训练/仿真为零。

---

## 以下为2026-09-21历史指南，冲突处以本文件顶部的当前有效目标为准

"""
old_guide = GUIDE.read_text(encoding="utf-8")
GUIDE.write_text(target_text + old_guide, encoding="utf-8", newline="\n")

selection = RESEARCH / "two_action_distance_selection_after80/result.json"
paired = RESEARCH / "paired_preview_curriculum_from_two_action80/result.json"
selected = json.loads(selection.read_text(encoding="utf-8"))
paired_data = json.loads(paired.read_text(encoding="utf-8"))
historical = OUT / "all34_dominance_research/joint_transit_retained_seed922222_full"

for path in STATES:
    state = json.loads(path.read_text(encoding="utf-8"))
    existing = {item["directory"]: item for item in state.get("stage_records", [])}
    for directory in sorted(RESEARCH.glob("ppo14_*")):
        if not directory.is_dir() or not (directory / "result.json").is_file():
            continue
        entry = existing.setdefault(str(directory), {"directory": str(directory)})
        for key, filename in [("result", "result.json"), ("receipt", "receipt.json"), ("live", "live.json")]:
            source = directory / filename
            if source.exists():
                entry[key] = json.loads(source.read_text(encoding="utf-8"))
        entry["effective_status"] = "finished" if entry.get("receipt", {}).get("exit_code") == 0 else "inspect_receipt"
        entry["live_snapshot_scope"] = "Historical training snapshot; receipt and result determine current stage status."
    state.update({
        "updated_utc": now.isoformat(),
        "current": "按用户最新指示优先核验历史最有效主方法，允许有效规则；原18页稿与前三图保护，重点修改实验。当前无活动训练，新密集案例5%等验收尚未通过。",
        "goal_complete": False,
        "method_frozen": False,
        "research_acceptance_passed": False,
        "final_manuscript_accepted": False,
        "final_pdf": None,
        "active_stages": [],
        "stage_records": list(existing.values()),
        "effective_goal_document": str(GUIDE),
        "effective_goal_document_sha256": hashlib.sha256(GUIDE.read_bytes()).hexdigest(),
        "effective_goal_priority": "Reuse and verify the historical best controller; effective rules are allowed. Do not require rule removal or pure PPO retraining as a prerequisite.",
        "goal_tool_status": "active",
        "goal_tool_objective_note": "The goal API cannot edit an active objective; latest user instructions and effective_goal_document supersede its older rule-removal wording. No new goal created.",
        "selected_research_mainline": str(historical),
        "historical_candidate_complete_time_s": 444.302,
        "compact_branch_status": "retained_development_evidence_no_active_training",
        "dimension_requirement_clarification": "Awaiting existing optional question response; do not silently remove earlier final-dimension requirement. Historical-controller verification may proceed.",
        "latest_two_action_parameter_selection": {
            "result": str(selection),
            "distance_mm": selected["selected"]["distance_mm"],
            "complete_time_s": selected["selected"]["result"]["complete_time_s"],
            "ppo_updates_this_selection": 0,
        },
        "paired_curriculum_initialization": {
            "result": str(paired),
            "initial_executions_matched": paired_data["initial_executions_matched"],
            "new_ppo_updates": paired_data["new_ppo_updates"],
            "subsequent_training_started": False,
            "scope": paired_data["limitation"],
        },
        "next_steps": [
            "Verify historical best model, rules, source, parameters and traces as one bound implementation.",
            "Register <=30 minute reproduction on the already fixed dense case; preserve original case and all34 checks.",
            "Build a comparable genuinely retrained two-output ablation with applicable shared rules and fair selection opportunity.",
            "Resolve allowed-versus-forbidden outward-swing comparison using complete successful trajectories; preserve current incomplete evidence.",
            "Complete independent seeds, dual-resolution and mechanism/time acceptance before freezing.",
            "Revise experiments on the unique18-page source; preserve first3 figures and minimize necessary non-experimental changes.",
        ],
        "latest_state_backup_receipt": str(backup_dir / "receipt.json"),
    })
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

addition = f"""

## {now.isoformat()}：训练已结束及用户最新目标转向

此前“正在训练/待启动”的段落是历史记录，当前没有活动训练。

- 允许外摆三动作完成累计80次PPO更新、163840决策，完整时间26.673s；本次继续训练工作进程墙钟567.78s。
- 两输出完成相同累计预算，D=1.5mm时28.273s；本次工作进程墙钟572.18s。随后九个全局距离的参数选择（0次PPO更新）选得D=4mm、25.334s，快于紧凑三动作。不得丢弃这一强对照。
- 禁止外摆soft-prior50轮完成16更新、32768决策，累计64更新、131072决策，工作进程246.09s；14圈物理完整均违反额外外摆条件，严格验收仍无成功完整时间。
- 两输出模型扩展成两/三输出的共同起点核验已结束，两支均25.334s，rows/actions/observations最大差异均0，新增PPO更新0。共享同一个预训练父模型，不能称独立从零重复；尚未开始后续训练。
- 用户新指示允许规则并优先采用此前案例上最有效的主方法。主线改为核验历史 `joint_transit_retained_seed922222_full`；紧凑研究保留，不继续强制去规则路线。
- 当前有效目标见 `{GUIDE}`。原34角、固定密集案例、对S2及真两输出的密集案例5%、公平性和重复性目标未被撤销。降维是否仍为最终硬要求的已有异步澄清待答复。
- 原18页稿和前三图不改动；本次仅更新目标/状态/记录，未运行新训练或仿真。状态与文档修改前均保存原始字节及SHA256，备份位于 `{backup_dir}`。

实现保持简单：复用已有指南和状态结构，补充有效目标与完成收据，不新增控制模块；历史结果与当前验收分离，避免重复或相互矛盾的有效状态。
"""
with LOG.open("a", encoding="utf-8", newline="\n") as stream:
    stream.write(addition)

receipt = {
    "updated_utc": now.isoformat(),
    "kind": "User-directed effective-target and state update only",
    "backups": backups,
    "files_updated": [str(path) for path in [*STATES, GUIDE, LOG]],
    "new_training_or_simulation_count": 0,
    "goal_complete": False,
    "method_frozen": False,
}
(backup_dir / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
for path in [*STATES, GUIDE, LOG]:
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), path
    raw.decode("utf-8")
for path in STATES:
    state = json.loads(path.read_text(encoding="utf-8"))
    assert state["active_stages"] == [] and not state["goal_complete"] and not state["method_frozen"]
print(json.dumps({"effective_targets": str(GUIDE), "backup_receipt": str(backup_dir / "receipt.json"),
                  "state_files_updated": len(STATES), "new_training_or_simulation_count": 0,
                  "utf8_no_bom_verified": True}, ensure_ascii=False, indent=2))
