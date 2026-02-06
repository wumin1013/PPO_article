# Phase 31：前瞻作为观测（v3.2 可执行稿）

版本日期：2026-02-06  
依赖：Phase 30（已跑通的 P0 / corridor+KCM 基线）  
目标：把 **lookahead** 从“指令式参考（θ_ref）”彻底退场，改为 **观测特征（obs）**，并用 **几何 cornerness 连续加权奖励**实现：  
- **非拐角段：精度优先（强跟踪）**  
- **拐角段：平滑优先（鼓励内切/抑制角速度峰值）**  
- 全程 **KCM** 保证运动学约束可行域

> v3.2 的核心：**多尺度观测落地一致 + corridor 保留检测、惩罚降级 + 避免 Tube 平地 + 验收阈值绑定 ε**。


---

## 0. v3.2 相比 v3.1 的修复清单（必须读）

### 修复 1：多尺度观测“文档/代码/维度”一致
- v3.2 **明确 obs_scales**，并要求 `_compute_lookahead_observation()` 按尺度循环返回 `2 * N_scales` 维特征。
- observation_dim 的增量与 N_scales 一致，避免“文档说 3 个尺度、代码只算 1 个”的维度错配。

### 修复 2：θ_ref 与 lookahead 完全解耦
- θ_ref **只来自切线（tangent）**。
- lookahead 只进 obs，不得再参与 θ_ref（防止“名义回归切线、实际仍在用 lookahead”）。

### 修复 3：corridor 默认保留检测，上层惩罚降级
- `corridor.enabled: true`（保留 turn_info/corner_phase/corridor_status 作为上下文检测）。
- corridor 奖励塑形项（outside/barrier/center/dir_pref）默认置 0，避免“靠规则惩罚跑出来”的质疑。

### 修复 4：主线禁用 minimal_mode + tube（Tube 只做对照组）
- v3.2 主线默认 `minimal_mode: false`、`tube.enabled: false`。
- Tube 平地会抹平梯度，不适合作为“直线精度优先”的主方案；如需对照，放进 ablation。

### 修复 5：验收阈值绑定 ε，避免尺度漂移
- 主阈值改为 `max_abs_e_n ≤ 0.30 × ε`（示例：ε=2.5 → 0.75mm）。

---

## 1. 设计原则（保证论文与工程都站得住）

1) **拐角判定必须客观（不可被策略操控）**  
   cornerness 由几何计算得到；策略不能通过动作改变“自己处于哪个奖励模式”，规避 reward hacking。

2) **“自适应前瞻”用多尺度观测实现，而不是让策略输出 L 当 action**  
   PPO 对“改变观测目标点”的间接因果学习不稳定；多尺度 obs 稳定、可消融、可解释。

3) **直线必须有连续强梯度**  
   避免全局 Tube/走廊平地，防止直线段出现“管道内摆动”或“精度退化”。

---

## 2. 配置新增与约定（YAML）

### 2.1 新增/启用：多尺度前瞻观测
建议新增字段（推荐做法）：

```yaml
environment:
  lookahead_obs_enabled: true           # v3.2: 固定观测维度，off 时输出全 0
  lookahead_obs_scales: [0.5, 1.0, 2.0] # v3.2: 多尺度相对系数（乘以 base_lookahead）
```

其中 base_lookahead 来自既有字段（保持兼容）：

```yaml
reward_weights:
  lookahead:
    enabled: true
    distance: 2.5   # base_lookahead（单位同轨迹/ε尺度）
```

> 兼容策略：若没有 lookahead_obs_scales，则默认 `[1.0]`（单尺度）；若 lookahead_obs_enabled=false，则仍保留维度但输出零向量。

### 2.2 corridor：检测保留，惩罚降级（主线）
```yaml
reward_weights:
  corridor:
    enabled: true
    # 注意：当前代码里 corridor.enabled 同时控制“拐角检测/phase”与“corridor 奖励塑形”。
    # 为了保留检测但移除强规则惩罚，主线做法是保持 enabled=true，
    # 同时将塑形权重置 0（与 PPO_project/src/environment/reward.py 的字段一致）。
    outside_penalty_weight: 0.0
    barrier_weight: 0.0
    center_weight: 0.0
    dir_pref_weight: 0.0
```

### 2.3 主线禁止 Tube 平地（Tube 只做对照）
```yaml
reward_weights:
  minimal_mode: false
  tube:
    enabled: false
```

---

## 3. 代码改动（最小可控）

> 文件路径以你当前工程结构为准：`PPO_project/src/environment/cnc_env.py` 与 `PPO_project/src/environment/reward.py`。

### Step 1：恢复/实现多尺度 lookahead 观测（cnc_env.py）

#### 1.1 读取配置并固定观测维度
- 新增成员：
  - `self.lookahead_obs_enabled`（bool）
  - `self.lookahead_obs_scales`（List[float]）
  - `self.lookahead_obs_dim = 2 * len(scales)`
- **重要：不要再硬编码** `self.lookahead_points = 0`。

#### 1.2 观测特征定义（每个尺度两维）
对每个尺度 `s`：
- `L = base_lookahead * s`
- `theta_L = heading_at_arc_length(s+L)`（或用既有 lookahead dir 的角度）
- `angle_diff = wrap(theta_L - theta_tangent) / π`  → [-1, 1]
- `dist_ratio = clamp(dist_to_lookahead_point / L, 0, 1)` → [0, 1]

拼接得到：`[angle_diff_near, dist_ratio_near, angle_diff_mid, dist_ratio_mid, ...]`

#### 1.3 固定维度的“关闭方式”
- 当 `lookahead_obs_enabled=false` 时：
  - 仍然输出同维度全 0 向量（避免网络结构随消融变化）。

#### 1.4 θ_ref 解耦：只用切线
- 将“参考方向”函数拆成两类语义（命名要明确）：
  - `get_tangent_direction()`：只返回切线（用于 θ_ref）
  - `get_lookahead_observation()`：只用于 obs 特征
- 确保奖励/控制使用的是 tangent 参考，不再调用 lookahead 来设置 θ_ref。

---

### Step 2：曲率/拐角强度 cornerness（客观）与连续加权（reward.py）

#### 2.1 cornerness 定义（推荐：前瞻角变化而非数值曲率）
用 “far 尺度”（例如 `max(lookahead_obs_scales)`）计算：
- `delta_theta = abs(wrap(theta_far - theta_tangent))`
- `c_raw = delta_theta / theta0`
- `c = clip(EMA(c_raw), 0, 1)`

参数建议：
- `theta0 = 30deg`（≈0.5236 rad）
- `EMA` 时间常数 5~20 steps（建议先 10）

#### 2.2 连续加权奖励（直线精度优先、拐角平滑优先）
设：
- `e`：轮廓误差（例如 e_n）
- `smooth_term`：平滑惩罚（如 |ω|、|Δω| 或 jerk proxy）
- `v`：速度项（已有）

推荐形式：
```text
w_track(c)  = w_track0 * (1 - c)^p + w_track_min
w_smooth(c) = w_smooth0 * c^p
```

然后：
```text
r = - w_track(c) * e^2  - w_smooth(c) * smooth_term  + w_v * v  + (其他你已有的稳态项)
```

**关键护栏：**
- `w_track_min > 0`（拐角也不能完全放弃跟踪，否则会飞）
- `c` 必须 clip + 平滑，否则权重抖动会导致 PPO 不稳定

#### 2.3 与 corridor 的关系（主线策略）
- v3.2 主线：corridor 只提供 `corner_phase / status` 作为上下文（如果你已有），但 **不再提供强惩罚塑形**（权重为 0）。
- cornerness 负责连续权重调度，成为“论文贡献点”。

---

## 4. 训练与验收（必须按顺序做）

### 4.1 Smoke Test（不训练/少训练）
目的：确认 obs 维度正确、θ_ref 没被 lookahead 污染、cornerness 不抖动。
- 运行 1~3 个 episode（或 very short rollout）
- 检查：
  - obs 维度 = base_dim + 2 * N_scales
  - lookahead_obs_enabled=false 时，维度不变但值全 0
  - θ_ref 在直线段不随 lookahead distance 改变
  - cornerness `c` 在直线≈0，在拐角上升且平滑

### 4.2 主线训练（v3.2）
- corridor.enabled=true，但 corridor 相关奖励权重 = 0
- minimal_mode=false，tube=false
- 先在 square/path 这种可复现任务上训练到稳定收敛

### 4.3 验收指标（绑定 ε）
令 `epsilon` 为容差带尺度（单位同 e_n）。
- Hard Gate：
  - `reached_target = true`
  - `max_abs_e_n ≤ 0.30 × epsilon`  
    例：ε=2.5 → 0.75（同 epsilon 单位）
- Soft Gate（与基线对比）：
  - `corner_peak_omega` 下降（建议 ≥10%）
  - `corner_min_v` 上升或不下降
  - 直线段平均误差不退化（例如直线窗口 mean|e|）

> 强烈建议：所有 “更好” 指标都与 **v3.0 基线（corridor+KCM，固定 lookahead）** 做对比，并做多 seed。

---

## 5. 消融矩阵（论文最值钱的证据）

保持网络结构固定（因为 lookahead obs 关闭时输出零向量）。

| 组别 | lookahead obs | cornerness 自适应权重 | corridor 惩罚塑形 | 目的 |
|---|---|---|---|---|
| A 基线 | off | off | on（v3.0 原样） | 你当前最好结果 |
| B | on | off | on | 仅验证多尺度观测是否有益 |
| C | on | on | on | 叠加几何自适应但仍有规则保底 |
| D（主线） | on | on | off（权重=0） | 证明“去规则惩罚”也能保持性能 |
| E（对照） | on | on | off + minimal/tube on | 证明 Tube 平地确实伤梯度/伤精度 |

> corridor “on/off” 更推荐做成 **惩罚权重 on/off**，而不是 enabled=false（enabled=false 会连检测上下文一起丢掉，早期更易发散）。

---

## 6. 失败模式与快速诊断（出现就按此排查）

1) 直线段出现摆动 / 精度退化  
- 先检查：是否误启用了 tube/minimal_mode  
- 再检查：`w_track_min` 是否太小（拐角放松影响到直线？）、`w_track0` 是否过低  
- 再检查：θ_ref 是否被 lookahead 重新污染（必须只用 tangent）

2) 训练早期发散，弯道入口偏离  
- corridor 检测是否被关掉（enabled=false）  
- cornerness 是否未 clip/未 EMA 平滑（权重抖动）  
- 适当提升 `w_track_min`，并把 `theta0` 调大（让 c 更“保守”）

3) 拐角仍尖锐  
- `w_smooth0` 太小或 smooth_term 定义太弱（只惩罚 |ω| 不够？加 |Δω|）  
- far 尺度不够远（lookahead_obs_scales 的最大值太小）  
- cornerness 计算的 far theta 不正确（取点/取角错误）

---

## 7. 交付物清单（落盘/可复现）

- 配置：`train_square_v31_v3_2.yaml`（主线） + ablation configs（A~E）
- 日志：summary.json（包含 epsilon、N_scales、obs_scales、cornerness 参数）
- trace.csv：至少包含 `e_n, v, omega, cornerness(c), corner_phase/status`
- 图：  
  1) 轨迹对比（baseline vs v3.2）  
  2) cornerness(c) 与权重 w_track/w_smooth 的时间序列  
  3) 指标柱状图（多 seed）

---

## 8. 一句话给审稿人（写在方法/贡献里）

我们将前瞻信息从启发式参考中剥离，作为多尺度几何观测输入策略；并以客观几何 cornerness 连续调度“直线精度—拐角平滑”的奖励权重，在 KCM 约束屏蔽下实现可部署的安全强化学习控制。
