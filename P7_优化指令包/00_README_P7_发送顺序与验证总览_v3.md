# P7 发送顺序与验证总览（v3，加入实现风险点与补丁）

**最终目标**  
1) 直线段：快速进给（接近 MAX_VEL），稳定到终点  
2) 弯前：提前感知并按“可控性边界”降速（不是奖励硬压慢）  
3) 拐角：允差带内自选内切幅度 + 平滑圆角（连续航向、近似连续曲率）  
4) 出弯：回到参考路径附近，长段跟踪稳定  
5) 全程：可靠到终点（success 稳、stall 不误杀、数值不爆）

---

## 发送顺序（强制）
1. **P7.0 动作尺度与航向积分**（物理上可学；消除尖角根因）
2. **P7.2 LOS 可控性边界（含自适应预瞄）**（让“更聪明→更快”可学）
3. **P7.1 走廊奖励重构（含回中 ramp + 滞回）**（自选幅度 + 出弯回中稳定）
4. **P7.3 平滑与终点可靠性（含奇异点保护）**（曲率连续 + 不 NaN + 终点稳）

> 注意：P7.0 是环境地震。**默认从头训练**（见 P7.0 课程学习建议）。

---

## 统一的“数值安全”要求（所有阶段必须遵守）
- 禁止 reward/obs/info 出现 NaN/Inf：一旦出现立刻 assert 并输出 trace。
- 所有除法必须加 eps（推荐 1e-6 到 1e-4 量级，按单位选择）。
- 所有 wrap 统一使用 wrap_to_pi，避免角度跳变导致异常数值。

建议在 `Env.step()` 末尾做一次全局检查：
- `assert np.isfinite(reward)`
- `assert np.all(np.isfinite(state))`
- `assert np.isfinite(self.velocity) and np.isfinite(self.angular_vel)`

---

## 课程学习（Curriculum）建议（强烈建议执行）
由于 P7.0 改了动力学，旧模型直接迁移往往等于“灾后重建”。建议 curriculum：
1) Stage-1：Line（无拐角），训练到 success_rate≥0.98，平均 v_ratio_exec_last20%≥0.9  
2) Stage-2：Gentle turns（大圆角/小曲率），引入 LOS cap，训练到稳定提前减速  
3) Stage-3：Square（直角），打开走廊奖励与回中，训练到 success_rate≥0.9  
4) Stage-4：提高 MAX_VEL/收紧 half_epsilon/提高 jerk 约束（逐步加难）

每个 stage 的 checkpoint 作为下一 stage 初始化。

---

## 总体验收（建议写 tools/accept_p7_all.py）
### A. Line
- success_rate ≥ 0.95  
- mean(v_ratio_exec_last20%) ≥ 0.85  
- stall_triggered_rate = 0  
- NaN/Inf = 0

### B. Square
- success_rate ≥ 0.8（训练后目标 ≥0.9）  
- 平滑：dkappa_p95 明显下降；且 kappa 全程 finite  
- 出弯回中：exit 后 N 步内 median(|e_n|) ≤ 0.2*corridor_half  
- NaN/Inf = 0

### C. 泛化（参数扰动）
- half_epsilon ↑ → 内切幅度分布变宽  
- MAX_ANG_VEL ↓ → cap 更早下压、弯前降速更强  
- 同时保持 success_rate 不崩盘
