# Phase 32：多路径验证（v2.1）
版本日期：2026-01-17  
依赖：Phase 30 通过（曲率感知基线验证成功）

---

## 0) 目标

**验证 Phase 30 训练的策略在不同路径几何上的泛化能力**。

---

## 1) 测试路径

| 路径 | 配置 | 几何特点 | 验收重点 |
|------|------|----------|----------|
| square | scale=10, closed | 4个90°直角 | 基线（已验证） |
| s_shape | - | 连续曲线 | 曲线段连续性 |
| sharp_angle | angle=45° | 锐角转弯 | 极端情况 |
| line | - | 纯直线 | 消融（无拐角） |

---

## 2) 执行步骤

### Step 1：使用 Phase 30 模型评估不同路径

```powershell
conda activate PPO
cd PPO_project

# 使用同一模型测试不同路径
$model = "saved_models/curvature_v21/*/checkpoints/best_model.pth"

foreach ($path in @("square", "s_shape", "sharp_angle", "line")) {
    python tools/acceptance_suite.py `
        --phase p0_eval `
        --config configs/train_square_curvature_v21.yaml `
        --model $model `
        --path_type $path `
        --episodes 20 `
        --out "out/phase32_$path" `
        --deterministic
}
```

### Step 2：汇总结果

```powershell
python tools/a3_aggregate_runs.py `
    --runs out/phase32_square out/phase32_s_shape out/phase32_sharp_angle out/phase32_line `
    --out artifacts/phase32_multipath
```

---

## 3) 验收标准

### 3.1 必须项

| 路径 | success_rate | max_abs_e_n |
|------|--------------|-------------|
| 所有 | ≥ 0.90 | ≤ ε/2 |

### 3.2 观察项

| 指标 | 期望趋势 |
|------|----------|
| corner_peak_omega | 所有路径均低于 0.9×MAX |
| steps | 无显著增加 |

---

## 4) 结果处理

### 4.1 如果通过
→ 进入 Phase 33（消融分析）

### 4.2 如果失败
- 分析失败路径的特点
- 考虑路径类型特化训练
- 或调整训练数据分布

---

## 5) 交付物

| 文件 | 说明 |
|------|------|
| `artifacts/phase32_multipath/summary.json` | 汇总指标 |
| `artifacts/phase32_multipath/comparison.png` | 对比图 |
