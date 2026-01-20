# Phase 22：P0 基线重训（基于 12 维状态空间）（v1.9）
版本日期：2026-01-07  
依赖：21_StateSpace_Redesign 已完成（observation_dim == 12）

---

## 0) 目标
在新的 12 维状态空间上重新训练 P0 基线，验证：
1. 状态空间改动不破坏基础能力
2. 新基线可达到 P0 原有水平

---

## 1) 执行步骤

### Step 1：确认 21 已完成
```powershell
conda activate PPO
python -c "from src.environment import Env; print('observation_dim:', Env(...).observation_dim)"
# 预期输出：observation_dim: 12
```

### Step 2：训练新 P0
```powershell
python main.py --config configs/train_square_p0_12d.yaml --mode train --episodes 500
```

### Step 3：评估
```powershell
python tools/acceptance_suite.py --config configs/train_square_p0_12d.yaml \
    --phase p0_eval --model saved_models/P0_12d/.../best_model.pth \
    --episodes 20 --deterministic --out artifacts/P0_12d/eval
```

---

## 2) 验收标准
| 指标 | 条件 |
|------|------|
| `success_rate` | ≥ 0.95 |
| `max_abs_contour_error` | ≤ half_epsilon |
| `mean_progress_final` | ≥ 0.99 |

---

## 3) 交付物
- `saved_models/P0_12d/` 目录
- `artifacts/P0_12d/eval/summary.json`
