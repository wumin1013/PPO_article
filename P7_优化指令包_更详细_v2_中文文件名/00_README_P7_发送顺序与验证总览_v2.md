# P7 发送顺序与验证总览（v2，更详细）

**最终目标（不可偏离）**  
1) 直线段：快速进给（接近 MAX_VEL，且不越界/不抖动）  
2) 弯前：提前感知并降速，降速来源是“可控性边界”（不是奖励硬压慢）  
3) 拐角：允差带内**自选内切幅度**，并形成**平滑圆角轨迹**（连续航向、连续曲率或近似连续）  
4) 出弯：能主动回到参考路径附近，保持长段跟踪稳定  
5) 全程：可靠到终点（open-path success 触发稳定；不被 stall 误杀）

---

## 为什么要做 P7（从根因出发）
你当前版本存在两类“结构性限制”，导致“尖角内切 + 不回中 + 不到终点”：
- **动力学结构**：`calculate_new_position()` 每步把航向基准重置为参考路径切向（拐角处切向跳变→尖角折线；偏离后难回中）。  
- **动作量纲**：`step()` 把 `policy_length∈[0,1]` 当作 `vel_action` 直接喂给 `apply_kinematic_constraints()`，但该函数把 `vel_action` 当作与 `MAX_VEL` 同单位的绝对速度 → 速度上不去、stall 更容易触发、终点更难到。

P7 的第一优先级是：**让环境在物理上“可学”**，然后再让奖励/边界“学得对”。

---

## 发送给 Codex 的顺序（强制）
1. **P7.0 动力学与动作尺度修复**（必须先做；否则后面全是空谈）
2. **P7.2 LOS 可控性边界（内切→更快的硬耦合）**
3. **P7.1 走廊奖励重构（自选幅度 + 出弯回中）**
4. **P7.3 平滑与终点可靠性收口（曲率连续 / stall / success）**

> 约束：每一步修改都要“自验证通过后再做下一步”。不要把 4 个包一次性塞进同一个 PR。

---

## 每一步的交付物要求（给 Codex 的明确清单）
每个 P7.x 必须同时提交：
- 代码修改（明确到文件/函数）
- 新增或更新的验收脚本（放 `tools/accept_p7_x_*.py`）
- 关键日志打印（每个 episode 的 summary + 单条 trace 导出 csv）
- 通过/失败时的退出码（用于 CI/脚本自动化）

---

## 总体验收：最终“必须通过”的自动化检查（建议写成 tools/accept_p7_all.py）
### 场景 A：Line（直线）
**指标（建议 10 episodes, seed 固定，禁随机起点）**
- success_rate ≥ 0.95  
- mean(v_ratio_exec_last20%) ≥ 0.85（最后 20% 步平均速度比）  
- mean(|e_n|) ≤ 0.2 * half_epsilon（或 RMS contour_error ≤ 0.2*half_epsilon）  
- stall_triggered_rate = 0

### 场景 B：Square / Right-angle corner（直角拐角）
- success_rate ≥ 0.8（训练后目标 ≥0.9）  
- **平滑性**：  
  - max(|Δheading|/dt) ≤ MAX_ANG_VEL + small_eps（物理一致性）  
  - max(|Δkappa|) 在角点附近不出现尖峰（需要定义 kappa=|omega|/(v+eps)）  
  - 轨迹在拐角附近呈圆弧（可视化 + 曲率曲线双证据）
- **出弯回中**：离开 corner_phase 后 N 步内  
  - median(|e_n|) 回落到 ≤ 0.2*half_epsilon（N 建议 50~150 步，按 dt/速度定）

### 场景 C：不同允差带宽 / 不同角速度上限
- 对同一路径，调整 half_epsilon、MAX_ANG_VEL：策略行为应“可解释变化”：  
  - half_epsilon ↑ → 内切幅度分布变宽（可更激进）  
  - MAX_ANG_VEL ↓ → 弯前降速更早、更强（cap 下压更明显）

---

## 推荐训练/自检命令（按你项目脚本）
（示例，按你 configs 实际文件名调整）
- 训练：`python main.py --config configs/train_square.yaml --mode train --experiment_name p7_square`
- 自检（corridor/终点/越界）：用 `tools/accept_p7_all.py` 或分模块 accept 脚本跑。
- 常量动作 sanity：`python tools/sanity_constant_action.py --config ...`（必要时新增参数 a_theta/a_v）

