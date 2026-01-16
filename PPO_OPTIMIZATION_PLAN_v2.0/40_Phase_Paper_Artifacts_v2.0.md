# Phase 40：论文产物生成（Paper Artifacts）
版本日期：2026-01-17  
依赖：Phase 33 完成

---

## 0) 目标（一句话）

**一键生成论文所需的全部图表和数据**，确保可复现性。

---

## 1) 论文结构与产物映射

### 1.1 RCIM 论文结构

| Section | 内容 | 产物 |
|---------|------|------|
| I. Introduction | 背景、问题、贡献 | 无图表 |
| II. Related Work | 相关工作 | 无图表 |
| III. Method | 方法描述 | Fig. 1: 系统架构图 |
| IV. Experiments | 实验结果 | Fig. 2-5, Tab. 1-2 |
| IV-A. Setup | 实验设置 | Tab. 1: 参数表 |
| IV-B. Main Results | 主要结果 | Fig. 2-4, Tab. 2 |
| IV-C. Ablation | 消融实验 | Tab. 3 |
| V. Conclusion | 结论 | 无图表 |

### 1.2 产物清单

| 类型 | 文件 | 来源 |
|------|------|------|
| **图表** | | |
| Fig. 1 | `fig_system_arch.pdf` | 手绘/tikz |
| Fig. 2 | `fig_trajectory_square.pdf` | Phase 32 |
| Fig. 3 | `fig_trajectory_line.pdf` | Phase 32 |
| Fig. 4 | `fig_trajectory_s_shape.pdf` | Phase 32 |
| Fig. 5 | `fig_velocity_profile.pdf` | Phase 32 |
| Fig. 6 | `fig_learning_curve.pdf` | Phase 30 |
| **表格** | | |
| Tab. 1 | `tab_parameters.csv` | 配置汇总 |
| Tab. 2 | `tab_main_results.csv` | Phase 32 |
| Tab. 3 | `tab_ablation.csv` | Phase 33 |
| **数据** | | |
| manifest | `reproducibility_manifest.json` | 全流程 |

---

## 2) 产物生成脚本

### 2.1 主脚本 `scripts/generate_paper_assets.py`

```python
#!/usr/bin/env python
"""一键生成论文产物"""

import argparse
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

# 论文级绘图设置
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'figure.figsize': (3.5, 2.5),  # 单栏宽度
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def generate_trajectory_figure(run_dir: Path, path_type: str, out_dir: Path):
    """生成轨迹对比图"""
    # 加载 trace
    trace_path = run_dir / "rollout_det" / "trace.csv"
    if not trace_path.exists():
        print(f"Warning: {trace_path} not found")
        return
    
    trace = pd.read_csv(trace_path)
    
    fig, ax = plt.subplots()
    
    # 绘制参考轨迹
    ax.plot(trace["ref_x"], trace["ref_y"], 'k--', label='Reference', linewidth=0.8)
    
    # 绘制实际轨迹
    ax.plot(trace["x"], trace["y"], 'b-', label='Executed', linewidth=1.0)
    
    # 绘制允差带（简化）
    # ...
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_aspect('equal')
    ax.legend(loc='best')
    ax.set_title(f'{path_type.capitalize()} Trajectory')
    
    fig.savefig(out_dir / f"fig_trajectory_{path_type}.pdf")
    plt.close(fig)


def generate_velocity_profile(run_dir: Path, out_dir: Path):
    """生成速度剖面图"""
    trace_path = run_dir / "rollout_det" / "trace.csv"
    if not trace_path.exists():
        return
    
    trace = pd.read_csv(trace_path)
    
    fig, axes = plt.subplots(2, 1, figsize=(7, 4), sharex=True)
    
    # 速度
    axes[0].plot(trace["step"], trace["velocity"], 'b-', linewidth=0.8)
    axes[0].set_ylabel('Velocity (mm/s)')
    axes[0].axhline(y=100, color='r', linestyle='--', linewidth=0.5, label='MAX_VEL')
    
    # 角速度
    axes[1].plot(trace["step"], trace["angular_vel"], 'g-', linewidth=0.8)
    axes[1].set_ylabel('Angular Vel (rad/s)')
    axes[1].set_xlabel('Step')
    
    # 标记拐角区域
    if "corner_phase" in trace.columns:
        corner_mask = trace["corner_phase"] == 1
        for ax in axes:
            ax.fill_between(trace["step"], ax.get_ylim()[0], ax.get_ylim()[1],
                          where=corner_mask, alpha=0.2, color='orange', label='Corner')
    
    fig.tight_layout()
    fig.savefig(out_dir / "fig_velocity_profile.pdf")
    plt.close(fig)


def generate_learning_curve(run_dir: Path, out_dir: Path):
    """生成学习曲线"""
    log_path = run_dir / "training_log.csv"
    if not log_path.exists():
        return
    
    log = pd.read_csv(log_path)
    
    fig, ax = plt.subplots()
    
    ax.plot(log["episode"], log["reward"], 'b-', alpha=0.3, linewidth=0.5)
    
    # 平滑曲线
    window = 20
    smoothed = log["reward"].rolling(window=window, min_periods=1).mean()
    ax.plot(log["episode"], smoothed, 'b-', linewidth=1.0, label='Smoothed')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Reward')
    ax.legend()
    
    fig.savefig(out_dir / "fig_learning_curve.pdf")
    plt.close(fig)


def generate_main_table(multipath_dir: Path, out_dir: Path):
    """生成主结果表"""
    results = []
    
    for path_type in ["square", "line", "s_shape"]:
        summary_path = multipath_dir / path_type / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            results.append({
                "Path Type": path_type,
                "Success Rate": f"{summary.get('success_rate', 0):.2%}",
                "Max Error (mm)": f"{summary.get('max_abs_contour_error', 0):.3f}",
                "Mean Steps": f"{summary.get('mean_steps', 0):.0f}",
                "Mean Velocity": f"{summary.get('trace_mean_velocity', 0):.1f}",
                "Corner ω Peak": f"{summary.get('trace_corner_peak_abs_omega', 0):.2f}",
            })
    
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "tab_main_results.csv", index=False)
    
    # 生成 LaTeX 表格
    latex = df.to_latex(index=False, escape=False)
    with open(out_dir / "tab_main_results.tex", "w") as f:
        f.write(latex)


def generate_ablation_table(ablation_dir: Path, out_dir: Path):
    """生成消融表"""
    ablation_csv = ablation_dir / "ablation_table.csv"
    if not ablation_csv.exists():
        return
    
    df = pd.read_csv(ablation_csv)
    
    # 格式化
    df.to_csv(out_dir / "tab_ablation.csv", index=False)
    
    # LaTeX
    latex = df.to_latex(index=False, escape=False)
    with open(out_dir / "tab_ablation.tex", "w") as f:
        f.write(latex)


def generate_manifest(out_dir: Path):
    """生成可复现性清单"""
    import subprocess
    
    manifest = {
        "version": "2.0",
        "date": pd.Timestamp.now().isoformat(),
        "git_hash": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "python_version": subprocess.check_output(
            ["python", "--version"], text=True
        ).strip(),
        "phases_completed": ["20", "21", "22", "23", "30", "32", "33", "40"],
        "artifacts": {
            "figures": [
                "fig_trajectory_square.pdf",
                "fig_trajectory_line.pdf",
                "fig_trajectory_s_shape.pdf",
                "fig_velocity_profile.pdf",
                "fig_learning_curve.pdf",
            ],
            "tables": [
                "tab_main_results.csv",
                "tab_ablation.csv",
            ],
        },
    }
    
    with open(out_dir / "reproducibility_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--multipath_dir", type=str, required=True)
    parser.add_argument("--ablation_dir", type=str, required=True)
    parser.add_argument("--out", type=str, default="paper_assets")
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    multipath_dir = Path(args.multipath_dir)
    ablation_dir = Path(args.ablation_dir)
    out_dir = Path(args.out)
    
    # 创建目录
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    
    # 生成产物
    print("Generating trajectory figures...")
    for path_type in ["square", "line", "s_shape"]:
        generate_trajectory_figure(
            multipath_dir / path_type, path_type, out_dir / "figures"
        )
    
    print("Generating velocity profile...")
    generate_velocity_profile(run_dir, out_dir / "figures")
    
    print("Generating learning curve...")
    generate_learning_curve(run_dir, out_dir / "figures")
    
    print("Generating main results table...")
    generate_main_table(multipath_dir, out_dir / "tables")
    
    print("Generating ablation table...")
    generate_ablation_table(ablation_dir, out_dir / "tables")
    
    print("Generating manifest...")
    generate_manifest(out_dir / "data")
    
    print(f"Done! Assets saved to {out_dir}")


if __name__ == "__main__":
    main()
```

---

## 3) 执行步骤

### Step 1：确认依赖产物

```powershell
# 检查必需文件
$required = @(
    "artifacts/minimal_v1/rollout_det/trace.csv",
    "artifacts/minimal_v1/multipath/square/summary.json",
    "artifacts/minimal_v1/multipath/line/summary.json",
    "artifacts/minimal_v1/multipath/s_shape/summary.json",
    "artifacts/ablation_aggregate/ablation_table.csv"
)

foreach ($f in $required) {
    if (!(Test-Path $f)) {
        Write-Host "Missing: $f" -ForegroundColor Red
    }
}
```

### Step 2：运行生成脚本

```powershell
python scripts/generate_paper_assets.py \
    --run_dir artifacts/minimal_v1 \
    --multipath_dir artifacts/minimal_v1/multipath \
    --ablation_dir artifacts/ablation_aggregate \
    --out paper_assets
```

### Step 3：检查产物

```powershell
Get-ChildItem -Recurse paper_assets
```

期望结构：
```
paper_assets/
├── figures/
│   ├── fig_trajectory_square.pdf
│   ├── fig_trajectory_line.pdf
│   ├── fig_trajectory_s_shape.pdf
│   ├── fig_velocity_profile.pdf
│   └── fig_learning_curve.pdf
├── tables/
│   ├── tab_main_results.csv
│   ├── tab_main_results.tex
│   ├── tab_ablation.csv
│   └── tab_ablation.tex
└── data/
    └── reproducibility_manifest.json
```

---

## 4) 论文集成

### 4.1 LaTeX 引用

```latex
% 图
\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{figures/fig_trajectory_square.pdf}
    \caption{Trajectory comparison on square path.}
    \label{fig:trajectory_square}
\end{figure}

% 表
\begin{table}[t]
    \centering
    \caption{Main experimental results.}
    \label{tab:main_results}
    \input{tables/tab_main_results.tex}
\end{table}
```

### 4.2 论文仓库结构

```
论文项目/
├── main.tex
├── references.bib
├── figures/           ← 从 paper_assets/figures 复制
│   └── ...
├── tables/            ← 从 paper_assets/tables 复制
│   └── ...
└── supplementary/
    └── reproducibility_manifest.json
```

---

## 5) 可复现性保证

### 5.1 Manifest 内容

```json
{
  "version": "2.0",
  "date": "2026-01-17T...",
  "git_hash": "abc123...",
  "python_version": "Python 3.10.x",
  "phases_completed": ["20", "21", "22", "23", "30", "32", "33", "40"],
  "artifacts": {
    "figures": [...],
    "tables": [...]
  },
  "configs": {
    "minimal": "configs/train_square_minimal.yaml",
    "ablations": [...]
  }
}
```

### 5.2 复现命令

在论文中或补充材料中提供：

```bash
# 克隆仓库
git clone https://github.com/xxx/PPO_project.git
cd PPO_project

# 切换到论文对应版本
git checkout <git_hash>

# 安装依赖
pip install -r requirements.txt

# 训练主模型
python main.py --config configs/train_square_minimal.yaml --mode train

# 生成论文产物
python scripts/generate_paper_assets.py ...
```

---

## 6) 时间估算

| 步骤 | 时间 |
|------|------|
| 脚本开发/调试 | 1 小时 |
| 运行生成 | 10 分钟 |
| 检查/微调 | 30 分钟 |
| **总计** | **1.5-2 小时** |

---

## 7) 交付物

| 文件 | 说明 |
|------|------|
| `scripts/generate_paper_assets.py` | 生成脚本 |
| `paper_assets/` | 完整论文产物 |
| `paper_assets/data/reproducibility_manifest.json` | 可复现性清单 |

---

## 8) Phase 40 完成标志

当以下条件满足时，v2.0 优化流程**全部完成**：

- [ ] 所有 figures 生成且质量合格
- [ ] 所有 tables 生成且数据正确
- [ ] manifest 记录完整
- [ ] 论文仓库集成完成
