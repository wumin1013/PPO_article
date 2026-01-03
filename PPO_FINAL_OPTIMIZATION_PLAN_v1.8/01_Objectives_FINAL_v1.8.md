# PPO_project 目标文档（FINAL v1.8）
版本日期：2026-01-03  
面向：RCIM 投稿 + 可复现 Artifact（代码/数据/图表一键生成）

---

## 1) 总目标（系统层面）
在轮廓误差容许带（tolerance band，±ε/2）约束下学习策略 π，使其在不同轨迹族上同时满足：

- **直线段**：高进给、贴线（误差不超带）
- **拐角段**：容许带内自动平滑（“转弯更不激烈”，且降速尽量少）
- **全程**：到达终点、无停滞、运动学约束零违规（由 KCM/action shielding 保证或统计证明）

---

## 2) 论文要回答的科学问题（对应你 .tex 的核心叙事）
1) **端到端 RL 是否能在捷度约束/硬约束下学到更优运动规划？**  
2) **它是否能在“直线快 + 拐角顺 + 出弯快回线”三者之间找到更优折中？**  
3) **哪些组件真正贡献了提升（消融/对比）？**

---

## 3) 验收指标：必须项（PASS/FAIL）
> 必须项用于判定“没有退化”。改善项用于判定“确实更好”。

### 3.1 必须项（硬门槛）
- **可达性**：reached_target_rate（或 reached_target=true）
- **贴线**：max_abs_e_n（最大法向误差）不劣于 baseline_ref 的容忍比例（建议 ≤ 1.05×）
- **效率**：steps / cycle_time 不劣于 baseline_ref 的容忍比例（建议 ≤ 1.05×）
- **硬约束**：kcm_violation_rate = 0（或机制保证 + 明确说明）

### 3.2 改善项（至少满足 1 条，才算“进步”）
- **拐角平滑性**：corner_mask 段 peak(|omega|)、mean(|domega|) 或 jerk_proxy 显著下降
- **降速更少**：corner_mask 段 min(v) 更高 / v_drop_ratio 更低
- **出弯回线更快**：corner_end 后固定窗口内 e_n 收敛更快、带外步数减少
- **直线更快**：non-corner 段 mean(v) 更高，且贴线不退化

---

## 4) 证据链（必须能从产物自动计算出来）
为保证“可计算、可复现、可对比”，每个 Run Bundle 必须包含：
- `summary.json`：上述必须项 + 改善项（含分段统计）
- `trace.csv`：至少包含 corner_mask / dist_to_corner / omega(或 proxy) / mode(或 proxy)
- `plots/`：自动生成的对比图（P0 vs 当前 run）

---

## 5) 与论文产物的映射（强制一一对应）
你的论文（tex）里，实验与结果至少需要这些“自动生成”的东西：

### 图（Figures）
- Fig.\*（路径与轨迹对比）：S形、蝶形等典型路径的 overlay、v(t)、e_n(t)、omega/domega（对应 `fig:curve_s` / `fig:curve_butterfly`）
- Fig.\*（消融可视化）：组件打开/关闭对指标的影响（对应 `fig:ablation_vis`）

### 表（Tables）
- `tab:results`：不同方法/阶段在两种路径下的量化对比（来自 main_table.csv）
- `tab:ablation`：关键组件消融（来自 ablation_table.csv）
- `tab:hyperparams`：训练超参（来自 config.yaml 快照 + manifest）

> 原则：论文表格与图必须能从 artifacts 一键重建，避免“手工抄数”。

---
