# Phase 32：多路径验证（v3.0）
版本日期：2026-01-18  
依赖：Phase 30 完成

---

## 目标

验证 Phase 30 训练的策略在不同路径上的泛化能力。

---

## 测试路径

| 路径类型 | 拐角角度 | 配置文件 |
|----------|----------|----------|
| 方形 | 90° | `train_square_v30.yaml` |
| 梯形 | 60°, 120° | `train_trapezoid_v30.yaml` |
| 圆形 | 连续曲率 | `train_circle_v30.yaml` |

---

## 执行命令

```powershell
# 方形评估
python tools/acceptance_suite.py `
    --config configs/train_square_v30.yaml `
    --model saved_models/v30_lookahead_tube/best_model.pth `
    --episodes 50

# 梯形评估
python tools/acceptance_suite.py `
    --config configs/train_trapezoid_v30.yaml `
    --model saved_models/v30_lookahead_tube/best_model.pth `
    --episodes 50
```

---

## 验收标准

| 路径 | success_rate | corner_peak_ω |
|------|--------------|---------------|
| 方形 | ≥ 0.95 | < 0.85×MAX |
| 梯形 | ≥ 0.90 | < 0.90×MAX |
| 圆形 | ≥ 0.95 | 连续 |

---

## 输出

- 多路径对比表格
- 轨迹可视化
