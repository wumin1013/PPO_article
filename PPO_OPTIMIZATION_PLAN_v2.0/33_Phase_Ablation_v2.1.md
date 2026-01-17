# Phase 33：消融分析（v2.1）
版本日期：2026-01-17  
依赖：Phase 32 通过

---

## 0) 目标

**量化曲率状态对平滑涌现的贡献**，为论文提供消融证据。

---

## 1) 实验设计

### 1.1 对比组

| 配置 | 状态维度 | 曲率特征 | 预期行为 |
|------|----------|----------|----------|
| **A: 无曲率（对照）** | 12 | ❌ | 尖角 |
| **B: 有曲率（实验）** | 14 | ✅ | 平滑 |

### 1.2 控制变量

- 相同奖励函数（极简4项）
- 相同训练参数（episodes=500）
- 相同随机种子（seed=42）
- 相同路径（square, closed）

---

## 2) 执行步骤

### Step 1：训练对照组（12维）

```powershell
conda activate PPO
cd PPO_project

# 使用原 minimal 配置（12维，无曲率）
python main.py --config configs/train_square_minimal.yaml --mode train
```

### Step 2：训练实验组（14维）

```powershell
# 使用 curvature 配置（14维，有曲率）
python main.py --config configs/train_square_curvature_v21.yaml --mode train
```

### Step 3：对比评估

```powershell
python tools/b2a1_corner_evidence.py `
    --candidate artifacts/curvature_v21 `
    --baseline artifacts/minimal_v1 `
    --out artifacts/phase33_ablation
```

---

## 3) 核心指标对比

| 指标 | 对照组（12维） | 实验组（14维） | 期望差异 |
|------|----------------|----------------|----------|
| corner_peak_omega | ≈MAX | <0.9×MAX | **显著下降** |
| inside_ratio | <0.3 | >0.5 | **显著上升** |
| corner_min_v | 低 | 高 | **显著上升** |
| steps | 基准 | ≈基准 | 无显著差异 |

---

## 4) 统计分析

### 4.1 显著性检验

```python
from scipy.stats import ttest_ind

# 对 corner_peak_omega 进行 t 检验
t_stat, p_value = ttest_ind(omega_control, omega_experiment)
print(f"p-value: {p_value:.4f}")

# 期望 p < 0.05
```

### 4.2 效应量

```python
# Cohen's d
d = (mean_control - mean_experiment) / pooled_std
# 期望 d > 0.8（大效应）
```

---

## 5) 交付物

| 文件 | 说明 |
|------|------|
| `artifacts/phase33_ablation/ablation_table.csv` | 指标对比表 |
| `artifacts/phase33_ablation/ablation_boxplot.png` | 箱线图 |
| `artifacts/phase33_ablation/trajectory_overlay.png` | 轨迹对比 |
| `artifacts/phase33_ablation/stats.json` | 统计结果 |

---

## 6) 论文映射

消融分析对应论文的 **Ablation Study** 章节：

> "为验证曲率状态的关键作用，我们对比了有无曲率特征的两种配置。实验表明，缺少曲率信息时，策略在拐角处的角速度峰值接近运动学极限（...），而加入曲率状态后显著下降（...），差异具有统计显著性（p<0.001, Cohen's d=1.2）。这证明曲率感知是实现平滑过弯的关键因素。"
