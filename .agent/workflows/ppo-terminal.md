---
description: PPO Project Terminal Commands - All terminal operations must use PPO virtual environment
---

# PPO 项目终端操作规则

## 虚拟环境激活

**所有终端命令必须在 PPO conda 环境中执行。**

### 激活命令

```powershell
# 激活 PPO conda 环境
conda activate PPO

# 进入 PPO_project 目录
cd "c:\Users\wumin\Nutstore\1\DDPG的轨迹平滑\基于强化学习的轨迹平滑\PPO_project"
```

### 验证环境

```powershell
# 确认 Python 路径指向虚拟环境
python -c "import sys; print(sys.executable)"
# 应输出: c:\Users\wumin\...\PPO_project\.venv\Scripts\python.exe
```

## 常用命令模板

// turbo-all

### Smoke Test

```powershell
python tools/acceptance_suite.py --phase p0_smoke --config configs/p0_l2_gold.yaml --episodes 5 --out out/phase20_gate
```

### P0 Eval

```powershell
python tools/acceptance_suite.py --phase p0_eval --config configs/p0_l2_gold.yaml --model artifacts/P0_gold_20251230_034122/checkpoint.pth --episodes 10 --deterministic --out out/phase20_gate/p0_eval
```

### P0_L2 Eval

```powershell
python tools/acceptance_suite.py --phase p0_eval --config configs/p0_l2_gold.yaml --model artifacts/P0_L2/P0_gold_20251230_034122/checkpoint.pth --episodes 10 --deterministic --out out/phase20_gate/p0_l2_eval
```
