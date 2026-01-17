# Phase 40：论文输出（v3.0）
版本日期：2026-01-18  
依赖：Phase 30, 32, 33 完成

---

## 目标

整理实验结果，生成 RCIM 投稿所需的图表和文本。

---

## 1) Figure 列表

### Figure 1: 系统架构图
- 前瞻向量引导机制
- KCM Shielding
- 管道奖励

### Figure 2: 轨迹对比
- (a) P0_L2 Baseline（尖角）
- (b) v3.0 Lookahead+Tube（平滑）

### Figure 3: 角速度曲线对比
- 拐角处 ω 峰值对比

### Figure 4: 消融实验结果
- corner_peak_ω 柱状图
- inside_ratio 柱状图

---

## 2) Table 列表

### Table 1: 性能指标对比

| 方法 | success_rate | corner_peak_ω | inside_ratio | mean_velocity |
|------|--------------|---------------|--------------|---------------|
| P0_L2 | 0.98 | 0.95 | 0.3 | 45.2 |
| v3.0 Full | 0.98 | 0.72 | 0.65 | 52.1 |

### Table 2: 消融实验

| 配置 | Lookahead | Tube | corner_peak_ω |
|------|-----------|------|---------------|
| A | ❌ | ❌ | 0.95 |
| B | ✅ | ❌ | 0.82 |
| C | ❌ | ✅ | 0.88 |
| D | ✅ | ✅ | 0.72 |

---

## 3) 论文话术模板

### Abstract
> We propose a lookahead-guided reinforcement learning approach for CNC trajectory smoothing. By introducing preview control and tolerance-based rewards, the agent autonomously discovers smooth cornering strategies while maintaining manufacturing accuracy.

### Introduction Contribution
> 1. A **Kinematic Constraint Module (KCM)** that guarantees motion feasibility through action shielding.
> 2. A **lookahead observation mechanism** that provides continuous reference directions at sharp corners.
> 3. A **tolerance-based reward function** that encourages geometric optimization within manufacturing tolerances.

### Results
> The proposed method achieves a 24% reduction in peak angular velocity at corners (from 0.95 to 0.72 of maximum), while increasing inside-cutting ratio from 0.3 to 0.65, demonstrating emergent smooth trajectory generation.

---

## 4) 交付物

- `figures/` 目录下所有图表
- `tables/` 目录下所有表格数据
- `paper_draft.md` 论文草稿
