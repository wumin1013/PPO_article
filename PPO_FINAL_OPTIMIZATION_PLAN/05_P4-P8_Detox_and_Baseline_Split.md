# P4/P6/P7/P8 规则栈瘦身与“基线-学习”拆分（Patch 05 / 建议 PhaseB 前必做）
版本日期：2025-12-30

**目标**：把“安全约束（必须留）”与“启发式/滤波（默认不参与学习）”拆开，降低耦合与调试成本，避免“手工规划掩盖 RL 贡献”。  
**你要的最终效果**：同一套 PPO 策略学习问题里，环境只保留 *Safety*（KCM + 刹车包络 + turning-feasible cap 可选），其余启发式统一用开关控制并默认关闭；同时日志能明确指出“到底是谁在限制速度/抑制抖动/做平滑”。

---

## 0) 先讲清楚：哪些东西会真的影响行为？
在当前代码中（`src/environment/cnc_env.py`）：

### 0.1 会改变 action→state 动力学的（“硬影响”，必须可开关）
- **P4 turning-feasible cap**（`reward_weights.p4.speed_cap_enabled`）：直接限制 `v_ratio_exec` 上限（即使关掉，该代码也强制保留刹车包络 `v_ratio_brake`）。  
- **P6 目标速度平滑器**（`reward_weights.p6_1.v_target_smoother_enabled`）：把 policy 的 `v_des` 变成更慢/更平滑的 `v_target` 再进 KCM（这是典型“隐藏二阶系统”）。  
- **P7.3 曲率平滑**（`reward_weights.p7_3.kappa_smoothing_enabled`）：对 `omega_intent` 做温和滤波，可能“掩盖抖动来源”。  
- **P4 stall 终止**（`reward_weights.p4.stall_enabled`）：不是规划，但会改变 episode 何时结束（影响训练分布与调试体验）。

### 0.2 主要影响“角点阶段判定/日志”的（建议保留但默认不参与 reward shaping）
- **P8 几何扫描**：你观测里的 `distance_to_next_turn` 与 `next_angle` 就来自 `_scan_for_next_turn`，这是 PhaseB 的信息基础（不建议删）。  
- **VirtualCorridor**（`reward_weights.corridor.enabled`）：当前 RewardCalculator 主体几乎不使用 corridor 的 shaping 字段，但 corridor 会影响 `corner_phase` 等状态与日志（也可能用于你后续的分段评估）。

---

## 1) 你现在会觉得“冗余且难调”的根因
同一个现象（例如“出弯慢漂”“速度上不去”“抖动”）可能同时被以下东西影响：
- cap（P4/P6.0）  
- v_target 平滑器（P6.1）  
- 曲率平滑（P7.3）  
- KCM（物理约束）  
- 终止逻辑（stall/max_steps）

这会导致：**单次失败无法归因**，论文也很难证明“RL 真学到了什么”。

---

## 2) 必须加入的“开关选项”（本补丁的核心）
下面这些开关在你项目里**基本已经存在**（多数来自 `reward_weights.*`），但现在缺的是：**文档化 + 默认策略 + 两套 profile**。

> 你只要在 YAML 里改这些 key 就能开/关；不需要大改代码。  
> 这些 key 都能在 `cnc_env.py` 的 `_init_p4_config/_init_p6_1_config/_init_p7_3_config/_init_corridor_config` 里找到对应。

### 2.1 开关矩阵（建议你贴在论文/补丁首页）
| 层级 | 组件 | YAML Key | Legacy 默认（当前 default.yaml） | Detox 默认（建议） | 作用/备注 |
|---|---|---|---:|---:|---|
| Safety | turning-feasible cap | `reward_weights.p4.speed_cap_enabled` | True | **True** | 速度上限裁剪；关掉也仍保留刹车包络（更安全） |
| Safety | stall 终止 | `reward_weights.p4.stall_enabled` | True | True / False* | True 更省训练资源；False 便于调试“慢漂”原因 |
| Heuristic | v_target 平滑器 | `reward_weights.p6_1.v_target_smoother_enabled` | True | **False** | 强烈建议 Detox 关闭，避免隐藏二阶系统 |
| Regularizer | Δu 抑制 | `reward_weights.p6_1.du_enabled` | True | True | 是否启用动作变化率惩罚（注意：RewardCalculator 若未使用该项，则等同占位） |
| Heuristic | 曲率平滑 | `reward_weights.p7_3.kappa_smoothing_enabled` | True | **False** | Detox 关闭，让抖动由策略/奖励直接负责 |
| Meta/日志 | Corridor 阶段判定 | `reward_weights.corridor.enabled` | True | True / False | 若你要用 corridor 做 corner_phase 分段，保留 True；否则可 False 以减少耦合 |
| Heuristic | 方向偏好 | `reward_weights.corridor.dir_pref_weight` | 0.6 | **0.0** | Detox 置零；避免“手工偏好”替代 RL |
| Meta | P4 诊断输出 | `reward_weights.p4.debug` | False | True(可选) | 只影响打印，不影响学习 |

\*：`stall_enabled` 在 Detox 里建议先保持 True；只有当你怀疑 stall 提前终止掩盖了问题时，临时关掉做定位。

---

## 3) 两套建议 Profile（直接复制到新 YAML）
你现在的工程最好同时维护两份配置，用于“回归”和“真学习”。

### 3.1 Legacy Profile（用于回归：必须能复现 P0_gold）
> 这就是你当前 `configs/default.yaml` 的含义（保持不动），关键是把它固定为“金标准”。

### 3.2 Detox Profile（用于 PhaseB：启发式默认关闭，只留 safety）
建议你新建：`configs/detox.yaml`（以 default.yaml 为底，覆盖以下字段）：

```yaml
reward_weights:
  p4:
    speed_cap_enabled: true      # Safety：建议保留
    stall_enabled: true          # 先保留；怀疑误杀再关
    exit_boost_enabled: false    # 即使当前 reward 不用，也建议关掉减少心智负担
    debug: false

  p6_1:
    v_target_smoother_enabled: false   # 关键：关闭隐藏平滑器
    du_enabled: true                   # 规则化项，可保留
    w_du: 0.01
    du_mode: l1

  p7_3:
    kappa_smoothing_enabled: false     # 关键：关闭曲率平滑
    kappa_smoothing_beta: 0.25

  corridor:
    enabled: true                # 若 PhaseB 用 corner_phase 分段评估则保留
    dir_pref_weight: 0.0         # 关键：去掉方向偏好
```

---

## 4) 执行顺序（最省时间、最少返工）
1) **冻结 P0_gold**：保存最优 checkpoint 与对应 eval 产物（summary/trace/plots）。  
2) **跑一次 Legacy 验收**：作为“回归锚点”。  
3) **切到 Detox 配置跑验收**：允许短期指标下降，但必须满足：
   - 不崩溃、不 NaN/Inf（P7.3 trace dump 不能触发）
   - `info["p4_status"]` 里能看到 `v_ratio_policy/v_ratio_cap/v_target` 等关键字段（方便归因）
   - 训练/评估能稳定完成 episode（不会陷入“永远跑不完/永远 stall”）

---

## 5) 通过后，P4/P6/P7 代码要不要“删除”？
建议分两步走（更稳）：
1) **先分层 + 用开关把启发式默认关掉 + 让日志可解释**（这就是本 05）  
2) 等 PhaseB 主线跑通、论文定稿后，再把长期不用的启发式挪到 `baselines/` 或直接删（风险更低）

