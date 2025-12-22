# Patch Runbook：PhaseA → 通过 P8.1（Corner Exit Recenter v1）

> 目标：修复 `accept_p8_1_observation_and_corner_phase.py` 仍“越界导致 reached_target=False”的问题，达到进入 PhaseB 的门槛。

## 0. 你当前的失败是什么（结论先行）

自动化验收失败点很明确：
- `reached_target=False`
- `max_abs_e_n=0.8859685`，而门槛是 `0.98*half_eps = 0.735`（因为 `epsilon=1.5 → half_eps=0.75`）
- 运行总步数 `steps=54`
- `corner_mode_ratio=0.1296` 且 `cap_ang_active_ratio=0.1296`（只在约 7/54 步处于 corner_mode / ang cap 生效）
- 直线段速度利用率不差：`mean_speed_util_in_band=0.9059`

这些数字都来自你的 `summary.json`：fileciteturn2file0L2-L19

**推断**：问题不是“角速度 cap 没生效”，而是**“拐角退出（corner exit）后仍然发生横向漂移/转向未收敛，导致在直线段后段越界”**。  
从你给的 `square_e_n(t)` 曲线也能看出：前半段在界内，尾段 e_n 快速发散并在最后一步超门槛。

---

## 1. 根因定位（基于 trace 的关键现象）

> 下述现象来自你生成的 `trace.csv`（本补丁就是针对这些“模式”下药）。

### 1.1 “corner_mode 退出时刻”与“速度 cap 跳变”同发生
在 `corner_mode=1` 期间，`v_ratio_cap` 被压到很低（例如 0.01），但在退出 `corner_mode` 后，`v_ratio_cap` 很快跳回较高值（例如 0.04），导致车辆在**横向误差尚未回到小范围时就加速**。  
这类“退出瞬间加速”会放大横向误差，最终在直线段末端超出 `±0.98*half_eps`。

### 1.2 退出拐角后，e_n 逐步走向越界（不是瞬间炸）
你的越界是“慢慢漂出去”的：直到最后一步才超过 0.735。  
这非常像 **turn completion/steering neutralization 不充分**：拐完弯后转向没有及时归零或姿态没有收敛，导致在直线段继续带着一点“残余转向/残余角速度”走，越走越偏。

---

## 2. 进入 PhaseB 的门槛应该是什么

PhaseB 不应该接手“专家策略都跑不通 / 越界”的系统。建议 PhaseA 的硬门槛（以 open_square 为例）至少要满足：

1) **P8.1 通过**：`accept_p8_1_observation_and_corner_phase.py` 必须 PASS（包含 reached_target=True 和 e_n 门禁）  
2) **越界门禁**：`max_abs_e_n ≤ 0.98*half_eps`，并且“越界步数=0”（不是只靠最后一步侥幸）  
3) **corner_mode 行为合理**：ang cap 只在拐角附近生效，不应直线段长期压死（你已做到这点）fileciteturn2file0L12-L16  
4) **轨迹可解释**：`square_v_cap_breakdown.png` 中，corner 期间 v_cap 受 ang/brake 其中之一约束；退出后 v_cap **平滑回升**（不允许阶跃跳变）

---

## 3. 需要再打哪些补丁？（最小集合）

下面给出两个补丁，按优先级从“最该先做”到“增强稳健性”。你可以先只打 Patch-1，通常就能过线；若仍偶发越界，再加 Patch-2。

### Patch-1：Corner Exit Recenter（拐角退出缓冲 + cap 释放迟滞）【强烈建议】

**目的**：解决“corner_mode 一退出就加速 → 误差放大 → 直线段越界”。

#### 3.1 修改点 A：给 corner_mode 加退出迟滞（hysteresis）
把 `corner_mode` 从“纯距离门控”升级为**状态机**：

- **进入条件（ON）**：`dist_to_turn < d_on`  或  `corner_phase==1` 且 `turn_angle` 非零
- **退出条件（OFF）**：同时满足：
  - `dist_to_turn > d_off`（d_off > d_on，避免抖动）
  - `abs(e_n) < e_release`（比如 0.6*half_eps）
  - `abs(heading_error) < psi_release`（比如 5°~10°）
  - 连续满足 N 步（比如 N=3~5）

> 解释：这样做的关键是——**拐完弯还没“站稳”之前，不让系统进入“直线高速模式”**。

#### 3.2 修改点 B：v_ratio_cap 在退出 corner_mode 时“平滑释放”
即使 corner_mode 关闭，也不要让 `v_ratio_cap` 从 0.01 直接跳到 0.04。  
建议加一个统一的 **rate limiter**：

- `v_ratio_cap(t) ≤ v_ratio_cap(t-1) + dv_cap_up_max`
- `v_ratio_cap(t) ≥ v_ratio_cap(t-1) - dv_cap_down_max`（下调可以更快）

经验值：`dv_cap_up_max` 取 0.001~0.004（按你的步长/频率微调）。

#### 3.3 修改点 C：越界风险时触发“恢复限速”（recovery cap）
当 `abs(e_n) > e_warn`（比如 0.85*half_eps）时：
- 强制 `v_ratio_cap = min(v_ratio_cap, v_recovery)`（例如 0.015）
- 直到 `abs(e_n) < e_release`（比如 0.6*half_eps）再退出恢复模式

> 这招非常“工程”，但它能把最后那 0.15 的越界吃掉，直接把你推过 P8.1 门槛。

#### 3.4 验收标准（Patch-1）
- `accept_p8_1_observation_and_corner_phase.py` PASS
- `max_abs_e_n ≤ 0.98*half_eps`，且越界步数=0
- `reached_target=True` fileciteturn2file0L4-L18
- `square_v_ratio_cap(t)` 不出现“corner exit 阶跃上跳”（肉眼可见的台阶）

---

### Patch-2：Turn Completion Clamp（转弯完成/转向归零）【若 Patch-1 仍偶发越界再上】

**目的**：解决“退出拐角后仍带着残余转向/残余角速度，直线段漂移”。

实现思路二选一（优先选更贴近你当前专家策略实现的那种）：

#### 方案 2A：转弯进度 θ 限幅到 turn_angle
如果你的专家策略内部用“转弯进度”变量（例如 ω/θ）在累加：
- 让它**最多累加到** `turn_angle`，不要累加到 `2π`
- 一旦达到 turn_angle：强制将曲率/角速度命令衰减到 0（或直线段跟踪值）

#### 方案 2B：退出 corner_mode 后，转向用一阶回零滤波
- `steer = (1-α)*steer + α*steer_target`
- 在 corner exit 的 K 步内，把 `steer_target` 线性/指数地拉回 0

#### 验收标准（Patch-2）
- e_n 在 corner exit 后呈衰减（回到 0），而不是持续漂移
- `omega_exec`（或等价姿态量）在拐角结束后不再单调累加/饱和

---

## 4. 改动落点建议（结合你当前仓库结构）

你现在的改动点主要在：
- `cnc_env.py`（v_ratio_cap 生成、corner_mode）
- `accept_p8_1_observation_and_corner_phase.py`（门禁与诊断输出）

本补丁建议继续保持：
- cap 仍由 env 单点生成（DRY）
- 验收脚本只读字段，不再二次推导

**具体落点**：
- Patch-1 的状态机与 rate limiter：放 `cnc_env.py` 的 cap 生成位置（你说“同一处加入 angular cap 的 fillet 距离门控”的那一段）
- Patch-2 的转向回零：如果专家策略在 env 内实现，就放 env；若在 expert policy 单独模块，就放 expert policy 模块。

---

## 5. 回滚策略（必留）

所有行为改动都建议加 feature flag：
- `USE_CORNER_EXIT_HYSTERESIS`
- `USE_VCAP_RATE_LIMIT`
- `USE_RECOVERY_CAP`
- `USE_TURN_COMPLETION_CLAMP`

这样一旦发现副作用（例如过度保守导致速度 util 下滑），你可以快速对比定位。

---

## 6. 你下一步怎么做（最短路径）

1) 先打 Patch-1（退出迟滞 + cap 平滑释放 + 恢复限速）
2) 直接跑 `accept_p8_1_observation_and_corner_phase.py`
3) 如果仍偶发越界，再上 Patch-2（turn completion / 转向回零）

---

### 附：你当前 summary 的关键指标（用于对照）
- epsilon=1.5 → half_eps=0.75
- max_abs_e_n=0.8859（超门槛）
- mean_speed_util_in_band=0.9059（速度并不差）
- corner_mode_ratio=cap_ang_active_ratio=0.1296（角速度 cap 生效窗口合理）
- reached_target=false（验收失败的直接原因）

以上均引自 summary.json：fileciteturn2file0L2-L19
