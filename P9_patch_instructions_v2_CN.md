# P9 补丁指令文档 v2（SoftCap + Deadzone + 死区内弱向心力 + 尺度归一化）
日期：2025-12-24

> 说明：本文件设计为**可直接粘贴给 Codex / Claude** 执行的“行级 patch 指令”。  
> 核心原则：**默认不破坏现有 Hard 模式**（便于回归），同时新增 **SoftCap + Deadzone** 训练通路，让 RL 真正学到“入弯点选择/切弯/出弯加速”。

---

## 目标（DoD）
1. 修复 v_cap（P4/P6）预瞄采样间距 bug：让 cap 能在拐角前提前“看见”曲率变化。
2. 新增 SoftCap 模式：turning-feasible cap（几何/角速度相关）不再硬截断动作，而是变成 reward 的软惩罚；Hard 模式保持默认以保证回归安全。
3. 重写 tracking reward：实现 Deadzone（允差带内允许切弯），并在 deadzone 内加入**微弱向心力**，防止直线段随机游走/蛇形跑偏。
4. 提供尺度匹配训练配置：避免 `MAX_VEL=1000` + 小路径 + 有限角速度导致学习难度爆炸。
5. 增加自动化验收脚本与明确的人为验收指标（轨迹/分解项）。

---

## 影响文件清单
### 必须修改
- `PPO_project/src/environment/cnc_env.py`
- `PPO_project/src/environment/reward.py`

### 建议新增（不破坏旧配置）
- `PPO_project/configs/train_square_softcap_scaled.yaml`
- `PPO_project/tools/accept_p9_softcap_deadzone.py`
- （可选）`PPO_project/tools/run_p9_suite.py`（一键跑回归 + 新验收）

---

## Step 0：基线回归（必须先跑，便于对比）
在仓库根目录 `PPO_project` 执行：

```bash
python tools/accept_p7_0_dynamics_and_scale.py --config configs/train_square.yaml --outdir artifacts/p7_0_baseline
python tools/p6_feasibility_cap_report.py --config configs/train_square.yaml --outdir artifacts/p6_0_baseline --episodes 10
python tools/accept_p8_1_observation_and_corner_phase.py --config configs/train_square.yaml --out artifacts/phaseA_baseline
```

**通过标准**
- 三条命令均返回 exit code 0
- 保留输出 artifacts 目录作为 patch 前基线

---

## Step 1：修复 PreviewSpacing bug（关键致命伤）
### 背景（必须理解）
- `_p4_speed_cap_s`：语义应是“总预瞄弧长（window）”
- `_p6_speed_cap_spacing`：语义应是“采样间距 ds”
- 当前仓库逻辑把两者混用，导致采样跨度过大，拐角曲率被“平均掉” → cap 误判为直线

### 1A）`cnc_env.py`：`_init_p4_config()` 修正默认 spacing 语义
文件：`PPO_project/src/environment/cnc_env.py`  
函数：`def _init_p4_config(self) -> None:`

将 preview spacing 默认逻辑改成：
- 如果用户配置了 `speed_cap_preview_spacing`：将其视为 `ds`
- 否则默认 `ds = s_lookahead / preview_points`
- 做防御性钳制：`ds <= s_lookahead/pts`（避免用户误填巨大 ds）

建议直接使用以下代码块替换原逻辑：

```py
preview_spacing = cfg.get("speed_cap_preview_spacing", None)
s_lookahead = float(getattr(self, "_p4_speed_cap_s", 0.0))
pts = int(getattr(self, "_p6_speed_cap_points", 8))
pts = max(1, pts)

if preview_spacing is None:
    preview_spacing = s_lookahead / pts
else:
    preview_spacing = float(preview_spacing)

ds_max = s_lookahead / pts if s_lookahead > 0 else preview_spacing
preview_spacing = float(min(preview_spacing, max(ds_max, 1e-6)))

self._p6_speed_cap_spacing = float(max(preview_spacing, 1e-6))
```

### 1B）`cnc_env.py`：`_compute_p4_pre_step_status()` 使用 `_p6_speed_cap_spacing`
函数：`def _compute_p4_pre_step_status(self) -> Dict[str, float]:`

将错误的全局间距：

```py
preview_spacing = float(max(float(getattr(self, "lookahead_spacing", 1.0)), EPS))
```

替换为正确的局部采样间距：

```py
preview_spacing = float(max(float(getattr(self, "_p6_speed_cap_spacing", 0.0)), EPS))
```

并确保采样距离不超过窗口：

```py
s_i = float(min(preview_spacing * float(i + 1), s_lookahead, s_remain))
```

（建议）加入 debug 字段，方便验收定位：

```py
status["preview_spacing"] = float(preview_spacing)
status["preview_points"] = float(preview_points)
```

---

## Step 2：新增 SoftCap 模式（Hard 默认不变）
### 设计原则
- Hard：保持旧逻辑（回归安全）
- Soft：
  - **不再**用 turning-feasible cap（`v_ratio_cap`）硬截断 policy 输出
  - **仍保留**物理硬约束：MAX_VEL + 刹车包络（确保可停） + KCM（加速度/jerk 等）
  - 超 cap 变成 reward 里的软惩罚（cap_violation）

### 2A）`cnc_env.py`：读取新配置（`_init_p4_config()`）
在 `_init_p4_config()` 新增：

```py
self._p4_cap_mode = str(cfg.get("cap_mode", "hard")).lower()  # hard|soft

self._p4_deadzone_ratio = float(cfg.get("deadzone_ratio", 0.8))
self._p4_deadzone_center_weight = float(cfg.get("deadzone_center_weight", 0.1))
self._p4_deadzone_speed_weight = float(cfg.get("deadzone_speed_weight", 0.0))

self._p4_cap_violation_weight = float(cfg.get("cap_violation_weight", 0.0))
self._p4_cap_violation_power = float(cfg.get("cap_violation_power", 2.0))
self._p4_kcm_weight = float(cfg.get("kcm_weight", 2.0))
```

### 2B）`cnc_env.py`：pre-step 里 speed_target 的 cap 分支
在 `_compute_p4_pre_step_status()` 中，将：

```py
speed_target = float(min(speed_target_brake, float(status["v_ratio_cap"])))
```

改成：

```py
cap_mode = str(getattr(self, "_p4_cap_mode", "hard")).lower()
if cap_mode == "soft":
    # Soft：speed_target 只受刹车包络约束（硬安全），不再被 turning-feasible cap 教条式压死
    speed_target = float(speed_target_brake)
else:
    speed_target = float(min(speed_target_brake, float(status["v_ratio_cap"])))

status["speed_target"] = float(speed_target)
status["cap_mode_soft"] = 1.0 if cap_mode == "soft" else 0.0
```

### 2C）`cnc_env.py`：`step()` 中取消 turning-feasible 的硬 min（仅 soft）
在 `step()` 中实现如下语义（Hard 维持旧行为；Soft 不 min）：

```py
v_ratio_cap = float(p4_status.get("v_ratio_cap", 1.0))
v_ratio_brake = float(p4_status.get("v_ratio_brake", 1.0))
cap_mode = str(getattr(self, "_p4_cap_mode", "hard")).lower()

if cap_mode == "soft":
    v_u_exec = float(v_u_policy)

    # Hard safety：只保留刹车包络与 MAX_VEL（KCM 仍然会进一步约束）
    v_ratio_cap_hard = float(np.clip(v_ratio_brake, 0.0, 1.0))
    max_vel_cap_phys = float(min(float(self.MAX_VEL), v_ratio_cap_hard * float(self.MAX_VEL)))

    # 记录“超出 turning-feasible cap 的程度”，用于 reward 软惩罚
    cap_violation_ratio = float(max(0.0, v_u_policy - v_ratio_cap))
else:
    v_u_exec = float(min(v_u_policy, v_ratio_cap))
    max_vel_cap_phys = float(min(float(self.MAX_VEL), float(v_ratio_cap) * float(self.MAX_VEL)))
    cap_violation_ratio = 0.0
    v_ratio_cap_hard = float(v_ratio_cap)

p4_status["cap_violation_ratio"] = float(cap_violation_ratio)
p4_status["v_ratio_cap_hard"] = float(v_ratio_cap_hard)
p4_status["max_vel_cap_phys"] = float(max_vel_cap_phys)
p4_status["cap_mode_soft"] = 1.0 if cap_mode == "soft" else 0.0
```

注意：后续如果还有 `v_target = clip(..., max_vel_cap_phys)`，Soft 模式下它的含义变为“硬安全上限”（刹车包络+MAX_VEL），而不是 turning-feasible cap。

---

## Step 3：Reward 改造（Deadzone + 死区内弱向心力 + SoftCap 惩罚）
文件：`PPO_project/src/environment/reward.py`

### 3A）扩展 `calculate_reward()` 参数
在 `RewardCalculator.calculate_reward(...)` 增加参数（放末尾即可）：

```py
deadzone_ratio: float = 0.8,
deadzone_center_weight: float = 0.1,
deadzone_speed_weight: float = 0.0,
cap_violation_ratio: float = 0.0,
cap_violation_weight: float = 0.0,
cap_violation_power: float = 2.0,
kcm_weight: float = 2.0,
```

### 3B）实现 tracking reward：deadzone 内给“微弱向心力”，死区外用势垒
**必须行为：**
- deadzone 内：不做强误差惩罚，但给一个小的中心偏好，避免直线段“允差内无所谓→随机游走/蛇形”
- deadzone 外：靠近边界时惩罚迅速变陡（softplus barrier），防止越界

建议直接替换非 corridor 的 tracking 逻辑为以下代码（默认 `deadzone_center_weight=0.1`，可配置）：

```py
tracking_reward = 0.0
if not use_corridor:
    e = float(contour_error)
    e_abs = float(abs(e))
    dz = float(np.clip(deadzone_ratio, 0.0, 0.999)) * float(self.half_epsilon)
    dz = float(max(dz, 1e-6))

    if e_abs <= dz:
        # Deadzone 内：弱向心力（防止直线段漂移/蛇形）
        ratio_in_dz = float(e_abs / dz)
        w_center = abs(float(deadzone_center_weight))
        tracking_reward = w_center * (1.0 - ratio_in_dz ** 2)
    else:
        # Deadzone 外：softplus 势垒，越接近边界越陡
        m = float(self.half_epsilon - e_abs)            # 到边界剩余 margin
        m_safe = float(self.half_epsilon - dz)          # deadzone 边界处 margin
        s = float(max(0.05 * self.half_epsilon, 1e-6))  # 势垒陡峭度
        x = float((m_safe - m) / s)

        if x > 50.0:
            softplus = x
        elif x < -50.0:
            softplus = math.exp(x)
        else:
            softplus = float(math.log1p(math.exp(x)))

        tracking_reward = -2.0 * float(softplus)
```

### 3C）可选：deadzone 内速度奖励（慎用）
如果需要，可保留 `deadzone_speed_reward`，但建议 `deadzone_speed_weight <= 1.0`，否则可能诱导“直线贴边跑”。

### 3D）SoftCap penalty + KCM penalty 可配
将硬编码的：

```py
constraint_penalty = -2.0 * kcm_intervention
```

改为可配：

```py
constraint_penalty = -abs(float(kcm_weight)) * float(kcm_intervention)
```

新增 turning-feasible 超限惩罚：

```py
cap_violation_penalty = 0.0
w_cap = abs(float(cap_violation_weight))
if w_cap > 0.0 and float(cap_violation_ratio) > 0.0:
    p = float(cap_violation_power)
    if not math.isfinite(p) or p <= 0.5:
        p = 2.0
    cap_violation_penalty = -w_cap * float(float(cap_violation_ratio) ** p)
```

并加入 total，同时写入 `components`（用于验收/调参）：

```py
components.update({
  "cap_violation_ratio": float(cap_violation_ratio),
  "cap_violation_penalty": float(cap_violation_penalty),
  "deadzone_center_weight": float(deadzone_center_weight),
})
```

---

## Step 4：环境向 reward 透传新参数
文件：`cnc_env.py` 调用 `reward_calculator.calculate_reward(...)` 的位置

新增：

```py
cap_violation_ratio = float(p4_status.get("cap_violation_ratio", 0.0))
```

并在调用中加入：

```py
deadzone_ratio=float(getattr(self, "_p4_deadzone_ratio", 0.8)),
deadzone_center_weight=float(getattr(self, "_p4_deadzone_center_weight", 0.1)),
deadzone_speed_weight=float(getattr(self, "_p4_deadzone_speed_weight", 0.0)),
cap_violation_ratio=float(cap_violation_ratio),
cap_violation_weight=float(getattr(self, "_p4_cap_violation_weight", 0.0)),
cap_violation_power=float(getattr(self, "_p4_cap_violation_power", 2.0)),
kcm_weight=float(getattr(self, "_p4_kcm_weight", 2.0)),
```

---

## Step 5：新增尺度匹配训练配置（不改旧配置）
新建：`PPO_project/configs/train_square_softcap_scaled.yaml`

最小关键项（示例）：
- `MAX_VEL: 20.0`
- `cap_mode: soft`
- `deadzone_center_weight: 0.1`
- `cap_violation_weight: 8.0`

在 `reward_weights.p4` 下增加：

```yaml
cap_mode: soft
deadzone_ratio: 0.8
deadzone_center_weight: 0.1
deadzone_speed_weight: 1.0

cap_violation_weight: 8.0
cap_violation_power: 2.0
kcm_weight: 6.0
```

> 注意：epsilon 建议与路径尺度保持比例一致，不要随意缩到极小；如果要更严格循迹，建议做 curriculum（先宽后紧）。

---

## Step 6：新增自动化验收脚本（必须）
新建：`PPO_project/tools/accept_p9_softcap_deadzone.py`

**自动化断言（硬指标）**
1. Soft 模式：多步验证 `v_u_exec == v_u_policy`（误差 < 1e-6），确保已取消 turning-feasible 的硬截断。
2. 在拐角前强制 `v_u_policy=1.0`，至少一次出现 `cap_violation_ratio > 0`（说明 cap 在算且能“看见弯”）。
3. reward components 必须包含 `cap_violation_penalty`，且 violation 时该项为负。
4. 当 `abs(e) <= dz` 时，`tracking_reward` 必须为正，且 `tracking_reward <= deadzone_center_weight`（避免退回全 0 或强负值）。

**输出要求**
- `summary.json`：pass/fail + 关键统计（violation 次数、均值等）
- `trace.csv`：包含至少以下列：`cap_violation_ratio, tracking_reward, contour_error, v_u_policy, v_u_exec, v_ratio_cap, v_ratio_brake`

**退出码**
- pass：0
- fail：2

---

## Step 7：回归 + 新验收（推荐一键跑）
### 回归（Hard 默认）
```bash
python tools/accept_p7_0_dynamics_and_scale.py --config configs/train_square.yaml --outdir artifacts/p7_0_p9
python tools/p6_feasibility_cap_report.py --config configs/train_square.yaml --outdir artifacts/p6_0_p9 --episodes 10
```

### 新验收（Soft）
```bash
python tools/accept_p9_softcap_deadzone.py --config configs/train_square_softcap_scaled.yaml --outdir artifacts/p9_softcap
```

**通过标准**
- 所有命令 exit code = 0

---

## 人为验收指标（必须人工看图/日志）
1. **直线段严格循迹**：轨迹应收敛到参考路径中心线，不能长期蛇形漂移；即使速度饱和，也应因为“弱向心力”回中心。
2. **拐角切弯**：轨迹在允差带内呈内切弧线趋势（而非死贴拐点或直接 OOB）。
3. **出弯加速**：出弯后速度在几十步内恢复（不能长时间龟速漂移）。
4. **奖励分解项**：
   - `tracking_reward` 在直线段是小的正值（上限约等于 `deadzone_center_weight`）
   - `cap_violation_penalty` 只在冲弯时阶段性出现，不应全程巨大负值
   - KCM penalty 不应成为长期主导负项（否则尺度/约束仍不匹配）

---

## 调参建议（安全默认）
- 直线仍漂：`deadzone_center_weight` 从 0.1 提到 0.2
- 切弯不明显：降低 `deadzone_center_weight` 或提高进度/速度奖励权重
- 策略无视 cap 出界：提高 `cap_violation_weight`（8 → 12）或略降低 `deadzone_ratio`

（文档结束）
