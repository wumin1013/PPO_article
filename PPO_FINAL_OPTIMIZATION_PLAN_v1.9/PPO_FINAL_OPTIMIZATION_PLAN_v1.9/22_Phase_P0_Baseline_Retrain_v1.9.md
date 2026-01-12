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
cd PPO_project
conda run -n PPO python -c "import yaml; from src.utils.path_generator import get_path_by_name; from src.environment import create_environment_from_config; cfg=yaml.safe_load(open('configs/train_square_p0_12d.yaml','r',encoding='utf-8')); p=cfg['path']; extra={k:v for k,v in p.items() if k not in {'type','scale','num_points'}}; Pm=get_path_by_name(str(p['type']), scale=float(p.get('scale',10.0)), num_points=int(p.get('num_points',200)), **extra); env=create_environment_from_config(cfg, Pm, device=None); s=env.reset(); print('observation_dim:', env.observation_dim, 'len(reset_obs):', len(s))"
# 预期输出：observation_dim: 12 len(reset_obs): 12
```

### Step 2：训练新 P0
```powershell
cd PPO_project
python main.py --config configs/train_square_p0_12d.yaml --mode train
# 训练回合数在 configs/train_square_p0_12d.yaml: training.num_episodes 中设置
```

### Step 3：评估
```powershell
cd PPO_project
python tools/acceptance_suite.py --config configs/train_square_p0_12d.yaml `
    --phase p0_eval --model saved_models/P0_12d/.../best_model.pth `
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
