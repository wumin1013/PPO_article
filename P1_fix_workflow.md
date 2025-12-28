# P1 修复工作流

## 概述

本工作流分为三个阶段修复 P1 验收失败问题：
1. **Phase A**：Smoke 参数微调（不需要训练）
2. **Phase B**：迁移学习再训练（基于 P0 成功模型）
3. **Phase C**：重新验收

---

## Phase A：Smoke 参数微调

> [!IMPORTANT]
> 目标：通过调整 LOS 参数，让 Smoke 测试能够完成整个轨迹而不出界。

### A.1 当前 LOS 参数问题诊断

当前配置（`train_square_p1.yaml`）：
```yaml
p1:
  use_los: true
  L0: 4.4       # 前瞻距离基础值 - 过大
  Lmin: 2.2     # 最小前瞻 - 合理
  Lmax: 13.2    # 最大前瞻 - 过大
  kL: 0.01      # 速度增益
  deadzone_corner_ratio: 0.3  # 拐角允差 - 可能过大
```

**问题**：`L0=4.4` 在 scale=10 的方形路径上，前瞻距离占边长 44%，导致拐角处提前转向过多。

### A.2 建议的参数调整

```yaml
p1:
  use_los: true
  L0: 2.0       # 降低：从 4.4 → 2.0（占边长 20%）
  Lmin: 1.0     # 降低：从 2.2 → 1.0
  Lmax: 6.0     # 降低：从 13.2 → 6.0
  kL: 0.005     # 降低：从 0.01 → 0.005（减少速度对前瞻的影响）
  deadzone_corner_ratio: 0.15  # 降低：从 0.3 → 0.15
```

### A.3 执行 Smoke 测试验证

```bash
# 激活环境
conda activate PPO

# 进入项目目录
cd PPO_project

# 运行 Smoke 测试
python tools/acceptance_suite.py ^
    --phase p1_smoke ^
    --config configs/train_square_p1.yaml ^
    --episodes 1 ^
    --out artifacts/p1_smoke_v2
```

### A.4 检查点

- [ ] `artifacts/p1_smoke_v2/summary.json` 中 `done_reason` 不是 `oob`
- [ ] `progress_final` ≥ 0.9（完成90%以上轨迹）
- [ ] `max_abs_contour_error` < 0.75（在 epsilon/2 以内）
- [ ] `theta_ref_max_step` ≤ 0.35（参考信号连续）

> [!TIP]
> 如果仍然出界，继续降低 `L0` 和 `deadzone_corner_ratio`，直到 Smoke 通过。

---

## Phase B：迁移学习再训练

> [!IMPORTANT]
> 关键：P1 训练必须基于 P0 成功模型的权重继续训练，而非从头开始。

### B.1 确认 P0 最佳模型路径

P0 验收通过的模型：
```
saved_models/train/20251227_092744/checkpoints/best_model.pth
```

### B.2 修改 P1 配置以加载 P0 模型

在 `configs/train_square_p1.yaml` 中添加：
```yaml
experiment:
  mode: train
  model_path: saved_models/train/20251227_092744/checkpoints/best_model.pth  # 新增
  enable_kcm: true
```

### B.3 执行 P1 训练

```bash
# 激活环境
conda activate PPO

# 进入项目目录
cd PPO_project

# 启动训练（迁移学习）
python main.py ^
    --mode train ^
    --config configs/train_square_p1.yaml
```

### B.4 训练监控检查点

训练过程中观察：
- [ ] Episode 0 的 reward 应接近 P0 水平（约 -800）
- [ ] Reward 随 episode 增加应有下降趋势
- [ ] 无剧烈波动（方差应逐渐减小）

### B.5 训练完成条件

- [ ] 训练完成 1000 episodes
- [ ] `best_model.pth` 已生成
- [ ] 训练日志中 reward 呈收敛趋势

---

## Phase C：重新验收

### C.1 执行 P1 验收

```bash
# 激活环境
conda activate PPO

# 进入项目目录
cd PPO_project

# 获取最新训练目录（替换为实际路径）
# 假设新训练目录为 saved_models/train/YYYYMMDD_HHMMSS

# 运行验收
python tools/acceptance_suite.py ^
    --phase p1_eval ^
    --config configs/train_square_p1.yaml ^
    --model saved_models/train/<NEW_TRAIN_DIR>/checkpoints/best_model.pth ^
    --episodes 50 ^
    --baseline artifacts/p0_accept/summary.json ^
    --out artifacts/p1_accept_v2 ^
    --deterministic
```

### C.2 验收通过条件

根据 P1 文档定义：
- [ ] `success_rate` ≥ P0 baseline（不退化）
- [ ] `stall_rate` ≤ P0 baseline（不退化）
- [ ] `mean_progress_final` ≥ P0 baseline（不退化）
- [ ] `corner_peak_ang_acc` 较 P0 下降 ≥ 30%
- [ ] `corner_mean_speed` ≥ P0 的 80%
- [ ] `theta_ref_max_step` ≤ 0.35 rad/step

---

## 故障排除

### 问题：Smoke 仍然出界

1. 进一步降低 `L0`（尝试 1.5、1.0）
2. 增大 `epsilon`（从 1.5 → 2.0，临时放宽边界）
3. 检查 `cnc_env.py` 中 `_compute_los_metrics` 的实现

### 问题：训练不收敛

1. 降低学习率 `actor_lr`（从 2e-5 → 1e-5）
2. 增加 `ent_coef`（从 0.01 → 0.02）增加探索
3. 检查是否正确加载了 P0 模型权重

### 问题：验收指标退化

1. 检查 LOS 参数是否过于激进
2. 减少 `w_ang_acc` 权重（从 0.5 → 0.2）
3. 增加训练 episodes（从 1000 → 2000）

---

## 总结

| 阶段 | 目标 | 预计时间 |
|-----|-----|---------|
| Phase A | Smoke 测试通过 | 15-30 分钟 |
| Phase B | 迁移学习训练收敛 | 2-4 小时 |
| Phase C | P1 验收通过 | 30 分钟 |
