# Phase 33：消融实验（Ablation Study）
版本日期：2026-01-17  
依赖：Phase 32 完成（多路径验证通过）

---

## 0) 目标（一句话）

**通过消融实验验证各设计决策的贡献**：移除/修改各组件，量化其对性能的影响。

---

## 1) 消融实验设计

### 1.1 消融点

| ID | 消融项 | 变体 | 目的 |
|----|--------|------|------|
| A1 | KCM Shielding | 禁用 | 验证运动学约束保护的必要性 |
| A2 | Time Penalty | 减弱 (-0.01) | 验证效率压力驱动平滑的假设 |
| A3 | Time Penalty | 增强 (-0.05) | 验证效率压力的边际效应 |
| A4 | Boundary Penalty | 软边界 | 验证硬边界 vs 梯度边界的效果 |
| A5 | Curvature State | 禁用（若 Phase 31 启用） | 验证曲率状态的贡献 |
| A6 | Baseline P0_L2 | 原始 v1.9 P0 | 与新策略对比 |

### 1.2 核心对比

```
Minimal (Phase 30/31) vs P0_L2 (v1.9 baseline)
│
├─ 如果 Minimal 更好 → 证明极简路线有效
├─ 如果相当 → 证明复杂规则是冗余的
└─ 如果更差 → 需要分析原因（但预期不会）
```

---

## 2) 消融配置

### 2.1 A1: No KCM

```yaml
# configs/ablation_no_kcm.yaml
experiment:
  name: ablation_no_kcm
  enable_kcm: false  # 禁用运动学约束保护

# 其他继承 train_square_minimal.yaml
```

**预期**：性能严重退化，可能出现运动学违规或轨迹不稳定。

### 2.2 A2: Weak Time Penalty

```yaml
# configs/ablation_weak_time.yaml
reward_weights:
  p4:
    time_penalty: -0.01  # 原为 -0.02

experiment:
  name: ablation_weak_time
```

**预期**：平滑性可能下降（效率压力不足），但基础功能保持。

### 2.3 A3: Strong Time Penalty

```yaml
# configs/ablation_strong_time.yaml
reward_weights:
  p4:
    time_penalty: -0.05  # 原为 -0.02

experiment:
  name: ablation_strong_time
```

**预期**：平滑性可能提升，但可能出现"贪快"导致的边界问题。

### 2.4 A4: Soft Boundary

```yaml
# configs/ablation_soft_boundary.yaml
reward_weights:
  boundary:
    enabled: true
    mode: soft          # 软边界（梯度惩罚）
    penalty: -100.0     # 峰值惩罚
    soft_margin: 0.8    # 从 0.8 × half_epsilon 开始惩罚

experiment:
  name: ablation_soft_boundary
```

**实现**（在 reward.py 中）：
```python
if mode == "soft":
    soft_start = soft_margin * self.half_epsilon
    if abs_error > soft_start:
        ratio = (abs_error - soft_start) / (self.half_epsilon - soft_start)
        r_boundary = penalty * (ratio ** 2)
```

**预期**：轨迹可能更"敢"接近边界，平滑性可能提升。

### 2.5 A5: No Curvature State（若 Phase 31 启用）

```yaml
# configs/ablation_no_curvature.yaml
environment:
  curvature_observation:
    enabled: false

experiment:
  name: ablation_no_curvature
```

**预期**：与 Phase 30 结果相当（验证 Phase 31 的贡献）。

---

## 3) 执行脚本

```powershell
# 消融实验批处理
$ablations = @(
    "ablation_no_kcm",
    "ablation_weak_time",
    "ablation_strong_time",
    "ablation_soft_boundary"
)

foreach ($abl in $ablations) {
    Write-Host "Running $abl..."
    python main.py --config configs/$abl.yaml --mode train
    python tools/a1_pack_run.py --run_dir artifacts/$abl --baseline artifacts/P0_L2/P0_12d_gold_20260114_174552
}

# 聚合消融结果
python tools/a3_aggregate_runs.py --run_dirs artifacts/ablation_* --out artifacts/ablation_aggregate
```

---

## 4) 结果表格

### 4.1 消融汇总表

| 变体 | success_rate | max_e_n | mean_steps | corner_ω_peak | 结论 |
|------|--------------|---------|------------|---------------|------|
| Minimal (main) | - | - | - | - | 主方法 |
| A1: No KCM | - | - | - | - | KCM 必要性 |
| A2: Weak Time | - | - | - | - | 效率压力敏感性 |
| A3: Strong Time | - | - | - | - | 效率压力边际效应 |
| A4: Soft Boundary | - | - | - | - | 边界策略对比 |
| P0_L2 Baseline | - | - | - | - | v1.9 对比基线 |

### 4.2 论文消融表（Tab. 3）

| Method | Succ. ↑ | MaxErr ↓ | Steps ↓ | ω_peak ↓ | Smooth ↑ |
|--------|---------|----------|---------|----------|----------|
| **Ours (Minimal)** | **X%** | **X** | **X** | **X** | **X** |
| w/o KCM | X% | X | X | X | X |
| w/ Weak Time | X% | X | X | X | X |
| w/ Strong Time | X% | X | X | X | X |
| Soft Boundary | X% | X | X | X | X |
| Baseline (P0_L2) | X% | X | X | X | X |

---

## 5) 分析要点

### 5.1 KCM 消融（A1）

**问题**：如果禁用 KCM，会发生什么？

**预期分析**：
- 运动学违规率 > 0
- 轨迹可能出现不物理的急转
- 证明 KCM 是安全保障的必要组件

**论文叙事**：
> "消融实验表明，运动学约束模块（KCM）是保证策略安全性的关键组件。禁用 KCM 后，运动学违规率从 0% 上升到 X%，轨迹出现明显的不连续和振荡。"

### 5.2 Time Penalty 敏感性（A2/A3）

**问题**：时间惩罚权重如何影响平滑性？

**预期分析**：
- 权重过低 → 策略"不着急" → 可能用更多步数绕路
- 权重过高 → 策略"贪快" → 可能频繁触碰边界

**论文叙事**：
> "时间惩罚权重是效率-安全权衡的关键参数。实验表明，权重为 0.02 时取得最佳平衡，过低导致步数增加 X%，过高导致边界违规率上升 X%。"

### 5.3 Boundary 策略（A4）

**问题**：硬边界 vs 软边界？

**预期分析**：
- 硬边界：策略"保守"，远离边界
- 软边界：策略可以"试探"边界，可能更平滑

**论文叙事**：
> "对比硬边界和软边界策略，我们发现硬边界在本任务中表现更稳定，因为软边界可能导致策略过度利用边界裕量。"

---

## 6) 交付物

| 文件 | 说明 |
|------|------|
| `configs/ablation_*.yaml` | 各消融配置 |
| `artifacts/ablation_*` | 各消融 Run Bundle |
| `artifacts/ablation_aggregate/ablation_table.csv` | 消融汇总表 |
| `paper_assets/tables/tab_ablation.csv` | 论文消融表 |

---

## 7) 时间估算

| 步骤 | 时间 |
|------|------|
| 创建配置 | 30 分钟 |
| 运行 4 个消融实验（并行可加速） | 2-4 小时 |
| 分析结果 | 1 小时 |
| 生成表格 | 30 分钟 |
| **总计** | **4-6 小时** |

---

## 8) 论文映射

### 8.1 节次

消融实验结果放在论文的 **Section IV-C: Ablation Study** 或 **Section V: Discussion**。

### 8.2 讨论点

1. **KCM 的必要性**：证明强化学习需要机制保护来满足硬约束
2. **极简奖励的有效性**：证明复杂 reward shaping 可能是冗余的
3. **效率压力假设**：验证"时间惩罚驱动平滑"的假设
4. **设计决策的鲁棒性**：展示参数敏感性
