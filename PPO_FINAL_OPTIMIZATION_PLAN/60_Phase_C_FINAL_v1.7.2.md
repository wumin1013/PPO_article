# Phase C：论文表格与图（从 Run Bundle 自动生成）（FINAL v1.7.2）
版本日期：2025-12-30  
依赖：你已经拥有一批 Run Bundle（至少 P0_L2 + 若干 PhaseB run）。

---

## 1) Phase C 的目标

### A1 归档要求（本阶段强制）

- **本阶段每一次 run（训练 / 评估 / 消融），无论 PASS/FAIL，都必须按 A1 归档规范生成 Run Bundle（Level-2）。**
- PASS 的前提：该 run 已形成完整 Run Bundle，且 summary.verdict=PASS。  
  （换句话说：没有归档，就不存在“通过”。）
- 失败 run 也必须归档：这是定位退化/漂移/偶发性的唯一可靠证据链。

把 PhaseB 的结果变成“审稿人看得懂、你自己复现得了”的论文材料：

- 主表（性能对比）：success / max_abs_e_n / time(steps) / KCM 等
- 平滑图（拐角段）：ω(t)、|Δω|、corner_mask 分段统计
- 轨迹对比图：overlay（同一轨迹族下 baseline vs ours）

关键原则：**所有表格与图都只从 Run Bundle 读取数据**，不直接读临时训练目录。

---

## 2) 你需要的两个聚合产物（最小集）
1) paper_table.csv（或等价）：每个 Run Bundle 一行  
2) plot_data/：为每张图准备的干净数据（来自 trace 或 summary 聚合）

---

## 3) 主表字段建议（与你的验收维度对齐）
- run_id / tag / baseline_ref
- success_rate / reached_target
- max_abs_e_n（全程 + 分段可选）
- cycle_time_steps（或 steps）
- stall_rate / done_reason（可选）
- smoothness 主指标（拐角段）
- 关键开关摘要（Hard-impact/Shaping 的启用状态）

---

## 4) Phase C 的验收
- 给定同一批 Run Bundle，多次运行聚合应得到一致表格（除非你明确改变聚合规则）
- 从表格能直接选出论文主线的“最佳模型”，且理由清晰（不靠肉眼挑）

