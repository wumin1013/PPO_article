# Phase 40：论文输出（v2.1）
版本日期：2026-01-17  
依赖：Phase 33 通过

---

## 0) 目标

**生成 RCIM 投稿所需的全部图表、数据和代码存档**。

---

## 1) 论文结构映射

| 论文章节 | 对应 Phase | 核心内容 |
|----------|------------|----------|
| Introduction | - | 问题定义、动机 |
| Related Work | - | 文献调研 |
| **Method** | Phase 30 | 曲率感知状态 + 极简奖励 |
| **Experiments** | Phase 32 | 多路径验证 |
| **Ablation Study** | Phase 33 | 曲率状态消融 |
| Conclusion | - | 总结与展望 |

---

## 2) 必需图表

### 2.1 方法图

| 图编号 | 内容 | 来源 |
|--------|------|------|
| Fig.1 | 系统架构图 | 手绘/PPT |
| Fig.2 | 状态空间示意图（14维） | 手绘/PPT |
| Fig.3 | 奖励函数公式 | LaTeX |

### 2.2 实验图

| 图编号 | 内容 | 生成脚本 |
|--------|------|----------|
| Fig.4 | 训练曲线（reward vs episode） | `tools/plot_training.py` |
| Fig.5 | 轨迹对比（baseline vs ours） | `tools/plot_trajectory_overlay.py` |
| Fig.6 | 多路径验证结果 | `tools/plot_multipath.py` |

### 2.3 消融图

| 图编号 | 内容 | 生成脚本 |
|--------|------|----------|
| Fig.7 | 有/无曲率状态对比 | `tools/plot_ablation.py` |
| Fig.8 | 角速度分布箱线图 | `tools/plot_omega_boxplot.py` |

---

## 3) 必需表格

| 表编号 | 内容 | 格式 |
|--------|------|------|
| Table 1 | 状态空间定义（14维） | LaTeX |
| Table 2 | 奖励函数参数 | LaTeX |
| Table 3 | 定量评估指标 | LaTeX |
| Table 4 | 多路径验证结果 | LaTeX |
| Table 5 | 消融分析统计 | LaTeX |

---

## 4) 执行步骤

### Step 1：生成图表

```powershell
conda activate PPO
cd PPO_project

# 训练曲线
python tools/plot_training.py `
    --log artifacts/curvature_v21/logs/training_log.csv `
    --out paper/figures/fig4_training.pdf

# 轨迹对比
python tools/plot_trajectory_overlay.py `
    --baseline artifacts/P0_L2/rollout.csv `
    --ours artifacts/curvature_v21/rollout.csv `
    --out paper/figures/fig5_trajectory.pdf

# 多路径
python tools/plot_multipath.py `
    --data artifacts/phase32_multipath/summary.json `
    --out paper/figures/fig6_multipath.pdf

# 消融
python tools/plot_ablation.py `
    --data artifacts/phase33_ablation/ablation_table.csv `
    --out paper/figures/fig7_ablation.pdf
```

### Step 2：生成表格

```python
# 使用 pandas + tabulate 生成 LaTeX 表格
import pandas as pd
from tabulate import tabulate

df = pd.read_csv("artifacts/phase33_ablation/ablation_table.csv")
latex = tabulate(df, headers="keys", tablefmt="latex_booktabs")
with open("paper/tables/table5_ablation.tex", "w") as f:
    f.write(latex)
```

### Step 3：代码存档

```powershell
# 创建可复现代码包
git archive --format=zip --prefix=ppo_curvature/ HEAD > paper/code/ppo_curvature_v21.zip

# 添加配置快照
Copy-Item configs/train_square_curvature_v21.yaml paper/code/
Copy-Item artifacts/curvature_v21/checkpoints/best_model.pth paper/code/
```

---

## 5) 交付物清单

### 5.1 figures/
```
fig1_architecture.pdf
fig2_state_space.pdf
fig3_reward.pdf
fig4_training.pdf
fig5_trajectory.pdf
fig6_multipath.pdf
fig7_ablation.pdf
fig8_omega_boxplot.pdf
```

### 5.2 tables/
```
table1_state_space.tex
table2_reward.tex
table3_metrics.tex
table4_multipath.tex
table5_ablation.tex
```

### 5.3 code/
```
ppo_curvature_v21.zip
train_square_curvature_v21.yaml
best_model.pth
README.md
```

---

## 6) 论文核心叙事（v2.1）

### 6.1 Abstract 关键句

> "We propose a curvature-aware state representation that enables end-to-end reinforcement learning of smooth cornering behaviors without explicit geometric rules."

### 6.2 Method 关键段落

> "Unlike prior work that relies on complex reward shaping with 15+ tunable weights, our method uses a minimal reward function (progress + boundary + time) combined with a 14-dimensional state space that includes curvature features. This allows the policy to perceive the geometric shape of its executed trajectory and autonomously discover that smooth paths are more efficient."

### 6.3 Contribution 列表

1. 曲率感知状态设计（14维）
2. 极简奖励函数（4项）
3. 端到端平滑涌现验证
4. 多路径泛化实验
5. 消融分析证明曲率状态的关键作用
