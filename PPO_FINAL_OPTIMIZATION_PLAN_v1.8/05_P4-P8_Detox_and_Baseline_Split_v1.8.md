# Patch 05：Detox & Baseline Split（决策版）FINAL v1.8
版本日期：2026-01-03

> 这份文档在 v1.8 中被“重新定位”：  
> **不再要求大规模重构**，只保留对 PhaseB 必需的部分：  
> **Observability（观测/口径对齐） + Baseline Isolation（基线隔离）**。

---

## 0) 什么时候必须执行这份文档？
满足任意一条就执行：
- 你无法从 trace 计算 corner_mask 分段指标（平滑/降速/回线）
- 你无法稳定复现 baseline（P0_L2）并对比
- plotter 因字段名/路径结构不一致而无法一键出图

否则：不要做大重构，直接进入 PhaseB（B0/B2a...）。

---

## 1) 本补丁的交付目标（只做三件事）
1) **基线隔离**：baseline_ref 永远指向 P0_L2 Bundle（不可漂移）
2) **观测字段补齐**：trace 至少具备 corner_mask、dist_to_corner、omega/proxy、mode/proxy
3) **口径一致**：plotter 与 trace 字段对齐，P0_L2 一键出图

---

## 2) 修改流程（不写具体代码，只写步骤）
### 2.1 基线隔离（一次性）
- 将当前最优 P0 产物固化为 P0_L2 Bundle
- 在评测/聚合流程中加入“baseline_ref 显式字段”
- 验证：相同 baseline_ref + 相同 seed → eval 指标一致（允许轻微随机波动）

### 2.2 Observability：trace 增列（只增不改语义）
- 定义 corner_mask 的判定口径（基于现有 corner_mode 或几何 lookahead）
- 明确 omega/domega/jerk_proxy 的计算口径（使用你系统已有角运动量或等价量）
- 增加 mode/proxy（normal/corner/recovery）用于分段统计解释

### 2.3 Plotter 对齐
- 统一 trace 字段名映射（例如 position_x vs pos_x）
- plotter 输入：一个 baseline bundle + 一个 candidate bundle
- plotter 输出：overlay、v(t)、e_n(t)、omega/domega（若有）

---

## 3) 验收标准（必须通过才进入 PhaseB2a）
- P0_L2 在新日志/新 plotter 下 eval 不退化（必须项不变）
- 可以从 trace 自动得到：
  - corner_mask 分段的平滑指标
  - corner_mask 分段的降速指标
  - corner_end 后窗口的回线指标

---

## 4) 自动论文产物生成（本补丁完成后立刻执行）
- 生成 P0_L2 的“论文基线图包”（用于后续所有对比）
- 更新 main_table.csv 的 baseline 行（确保列齐全）

---
