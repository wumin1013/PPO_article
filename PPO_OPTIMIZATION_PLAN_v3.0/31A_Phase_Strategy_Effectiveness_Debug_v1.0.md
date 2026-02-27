# 31A：策略不生效与早收敛问题处置指南（v1.0）

版本日期：2026-02-27  
适用阶段：Phase31（可学习前瞻距离主线）

---

## 1. 问题定义

当前现象：
- 第 1 个迭代（或极少 episode）后，策略表现几乎不再变化。
- 训练曲线看似“稳定”，但性能不提升（成功率低、progress 停滞、拐角不平滑）。

这不是“收敛”，而是“**学习停滞**”（learning stagnation）。

---

## 2. 根因分层（按优先级）

### R1. 训练预算不足（最常见）
- 只训练 10 episode 或极短 rollout，PPO 尚未形成有效策略更新。
- 表现为：训练日志几乎无统计意义，评估跨回合高度同质。

### R2. 探索不足
- `ent_coef` 过低（甚至 0）导致策略过早确定化。
- action 分布很快塌缩，`u_lookahead` 不再探索。

### R3. 奖励可辨识度不足
- 直线/拐角奖励差异太弱，导致策略学不到“何时拉长前瞻”。
- `lookahead_reward` 权重过小，或 `straight_target/corner_target` 距离太近。

### R4. 控制权被强规则掩盖
- 速度上限/约束过强、KCM 干预过高，使策略动作对轨迹影响被吞掉。
- 表现为动作变化存在，但执行后状态变化很小。

### R5. 观测尺度与归一化问题
- lookahead 观测信号过小或过噪，网络难以利用。
- `use_obs_normalizer` 关闭时，特征量纲差异过大也会造成训练困难。

---

## 3. 先验判定：不是“策略学会了”，而是“训练无效”

满足以下任意 2 条即可判定“训练无效”：
- `success_rate < 0.3` 且长期不升。
- `mean_progress_final` 在多个 checkpoint 间变化 < 0.02。
- `u_lookahead` 分布标准差 < 0.05（几乎常数）。
- `lookahead_dist_active` 对 episode 的分布无明显迁移。

---

## 4. 修复方案（必须按顺序）

### Step A：先修训练可学习性
1. 训练预算提升到可学习区间：
   - `num_episodes >= 500`（建议 800~1500）
2. 增加探索：
   - `ent_coef: 0.002 ~ 0.01`
3. 保留 3 维动作：
   - `action_space_dim == 3`（含 `u_lookahead`）

### Step B：增强“分区奖励可辨识度”
1. 加大分区目标间隔：
   - `straight_target=0.20`
   - `corner_target=0.80`
2. 保证分区权重有效：
   - `region_source: cornerness`
   - `cornerness` 必须 EMA 平滑 + clip
3. 增强前瞻奖励权重：
   - `w_straight=0.6`, `w_corner=1.2`（可逐步上调）

### Step C：防止控制权被掩盖
1. corridor 保留检测但惩罚降级（权重为 0）。
2. 保持 `tube=false`、`minimal_mode=false`。
3. 监控 `mean_kcm_intervention`：
   - 若长期 > 0.75，优先检查约束是否过紧。

---

## 5. 必做监控（每次训练都要产出）

新增三条诊断曲线（按 episode）：
1. `u_lookahead` 均值/方差
2. `lookahead_dist_active` 在直线区与拐角区的双均值
3. `corner_peak_abs_omega` 与 `straight_mean_abs_error`

判定“策略真的在学”需要同时满足：
- `u_lookahead` 方差先升后稳（不是立即塌缩）
- 拐角前瞻均值 > 直线前瞻均值
- 拐角平滑指标改善且直线误差不退化

---

## 6. 验收门槛（针对“策略有效性”）

在原 Phase31 门槛外，追加：
- `action_dim == 3`
- `std(u_lookahead) >= 0.08`（中后期）
- `mean(lookahead_dist | corner) - mean(lookahead_dist | straight) >= 0.15`
- 与固定前瞻基线比：
  - `corner_peak_abs_omega` 至少下降 10%
  - `progress_final` 不低于基线

---

## 7. 推荐执行节奏（最小闭环）

1. `phase31_learnable_validate.py`（控制权冒烟）
2. 200 episode 短训（检查学习是否发生）
3. 800+ episode 主训（看最终性能）
4. 与固定前瞻基线做 A/B + 多 seed

若第 2 步仍无学习迹象，不进入第 3 步，先回到 Step B 调奖励辨识度。

---

## 8. 写作建议（论文口径）

不要把“首轮即稳定”描述为快速收敛。应明确区分：
- **Optimization convergence**（优化收敛）
- **Policy effectiveness**（策略有效）

论文中应报告“策略有效性证据”：
- 控制量分布演化
- 分区行为差异
- 指标提升而非仅曲线平稳
