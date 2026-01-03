"""
Utility script to generate paper-quality figures and summary statistics.

Figures:
- Fig 8: S-shape speed/acceleration/jerk comparison across J-NNC, NNC, Traditional.
- Fig 9: Butterfly trajectory with velocity heatmap.
- Fig 11: Contour error vs KCM intervention (dual axes).

It also prints Table 2 style metrics (Time, Max Error, Mean Error, Max Jerk)
for each method/path pair provided.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import sys

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as font_manager
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

STYLE_MAP = {
    "J-NNC": {"linestyle": "-", "color": "#1f77b4"},
    "NNC": {"linestyle": "-.", "color": "#2ca02c"},
    "Traditional": {"linestyle": "--", "color": "#d62728"},
}


def set_academic_style(font: str = "Times New Roman") -> None:
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    chosen_font = font if font in available_fonts else "DejaVu Sans"
    if chosen_font != font:
        print(f"[paper_plotter] Font '{font}' not found, fallback to '{chosen_font}'.")

    mpl.rcParams.update(
        {
            "font.family": chosen_font,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    sns.set_style("whitegrid")


def resolve_csv_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    candidates = [
        (PROJECT_ROOT / path).resolve(),
        (SAVED_MODELS_DIR / path).resolve(),
        (Path(__file__).resolve().parent / path).resolve(),
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]


def _ensure_columns(df: pd.DataFrame, label: str) -> pd.DataFrame:
    fallbacks: Dict[str, List[str]] = {
        "timestamp": ["step", "env_step", "episode_idx", "wall_time"],
        "pos_x": ["position_x", "x", "ref_x", "reference_x"],
        "pos_y": ["position_y", "y", "ref_y", "reference_y"],
        "velocity": ["speed", "mean_velocity"],
        "acceleration": ["acc", "mean_acceleration"],
        "jerk": ["mean_jerk"],
        "contour_error": ["error", "rmse_error"],
        "kcm_intervention": ["kcm", "mean_kcm_intervention"],
        "reward": ["total_reward", "episode_reward"],
    }
    for target, candidates in fallbacks.items():
        if target in df.columns:
            continue
        for cand in candidates:
            if cand in df.columns:
                df[target] = df[cand]
                print(f"[paper_plotter] '{target}' 缺失，使用备选列 '{cand}' (label={label}).")
                break
        else:
            print(f"[paper_plotter] 警告: 数据缺少列 '{target}' (label={label}).")
    return df


def load_run(csv_path: Path, label: str) -> pd.DataFrame:
    if not csv_path.exists():
        print(f"[paper_plotter] CSV not found: {csv_path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"[paper_plotter] 读取失败 {csv_path}: {exc}")
        return pd.DataFrame()
    print(f"[paper_plotter] Loaded {csv_path}, columns={list(df.columns)}")
    df = _ensure_columns(df, label)
    df["method"] = label
    try:
        df["reward_components"] = df["reward_components"].apply(json.loads)
    except Exception:
        pass
    return df


def plot_s_shape_comparison(runs: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    if len(runs) != 3:
        print("Fig8 skipped: need three runs for J-NNC, NNC, Traditional")
        return
    required_cols = {"timestamp", "velocity", "acceleration", "jerk"}
    if not all(required_cols.issubset(df.columns) for df in runs.values()):
        print("Fig8 skipped: 数据缺少必要列 timestamp/velocity/acceleration/jerk")
        return
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    metrics = ["velocity", "acceleration", "jerk"]
    titles = ["Velocity", "Acceleration", "Jerk"]

    for label, df in runs.items():
        style = STYLE_MAP.get(label, {})
        for ax, metric, title in zip(axes, metrics, titles):
            ax.plot(df["timestamp"], df[metric], label=label, **style, linewidth=1.8)
            ax.set_ylabel(title)
            ax.grid(True, linestyle=":", alpha=0.6)

    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title("S-shape Comparison (Fig.8)")
    axes[0].legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    out_path = output_dir / "fig8_s_shape_comparison.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved Fig8 to {out_path}")


def plot_velocity_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    required_cols = {"pos_x", "pos_y", "velocity"}
    if df.empty:
        print("Fig9 skipped: empty dataframe")
        return
    if not required_cols.issubset(df.columns):
        print(f"Fig9 skipped: 缺少列 {required_cols - set(df.columns)}")
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        df["pos_x"],
        df["pos_y"],
        c=df["velocity"],
        cmap="coolwarm",
        s=10,
        alpha=0.9,
        linewidths=0,
    )
    cb = fig.colorbar(sc, ax=ax, label="Velocity")
    cb.ax.tick_params(labelsize=9)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Butterfly Path Velocity Heatmap (Fig.9)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.5)
    out_path = output_dir / "fig9_velocity_heatmap.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved Fig9 to {out_path}")


def plot_kcm_dual_axis(df: pd.DataFrame, output_dir: Path) -> None:
    required_cols = {"timestamp", "contour_error", "kcm_intervention"}
    if df.empty:
        print("Fig11 skipped: empty dataframe")
        return
    if not required_cols.issubset(df.columns):
        print(f"Fig11 skipped: 缺少列 {required_cols - set(df.columns)}")
        return
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    ax1.plot(df["timestamp"], df["contour_error"], color="#1f77b4", label="Contour Error")
    ax2.plot(
        df["timestamp"],
        df["kcm_intervention"],
        color="#ff7f0e",
        linestyle="--",
        label="KCM Intervention",
    )

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Contour Error")
    ax2.set_ylabel("KCM Intervention")
    ax1.set_title("KCM Mechanism Analysis (Fig.11)")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.5)

    out_path = output_dir / "fig11_kcm_dual_axis.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved Fig11 to {out_path}")


def calc_stats(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "Time": float(df["timestamp"].max() if "timestamp" in df else 0.0),
        "Max Error": float(df["contour_error"].abs().max() if "contour_error" in df else 0.0),
        "Mean Error": float(df["contour_error"].abs().mean() if "contour_error" in df else 0.0),
        "Max Jerk": float(df["jerk"].abs().max() if "jerk" in df else 0.0),
    }


def print_table(config: Dict[str, Dict[str, Path]]) -> None:
    if not config:
        print("Table 2 skipped: no config provided")
        return
    print("\nTable 2 metrics (Time / Max Error / Mean Error / Max Jerk):")
    for path_name, runs in config.items():
        print(f"\nPath: {path_name}")
        for label, path in runs.items():
            df = load_run(path, label)
            stats = calc_stats(df)
            print(
                f"  {label:<12} | Time: {stats['Time']:.3f}s | MaxErr: {stats['Max Error']:.4f} | "
                f"MeanErr: {stats['Mean Error']:.4f} | MaxJerk: {stats['Max Jerk']:.4f}"
            )


def parse_table_config(config_path: Optional[str]) -> Dict[str, Dict[str, Path]]:
    if not config_path:
        return {}
    cfg_path = resolve_csv_path(config_path)
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}")
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        if cfg_path.suffix.lower() in {".yaml", ".yml"}:
            raw = yaml.safe_load(f)
        else:
            raw = json.load(f)
    parsed: Dict[str, Dict[str, Path]] = {}
    for path_name, mapping in raw.items():
        parsed[path_name] = {label: Path(p) for label, p in mapping.items()}
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures from CSV logs")
    parser.add_argument("--jncc_csv", type=str, help="CSV for J-NNC (S-shape)")
    parser.add_argument("--nnc_csv", type=str, help="CSV for NNC (S-shape)")
    parser.add_argument("--traditional_csv", type=str, help="CSV for Traditional (S-shape)")
    parser.add_argument("--heatmap_csv", type=str, help="CSV for heatmap (butterfly)")
    parser.add_argument("--fig11_csv", type=str, help="CSV for KCM analysis")
    parser.add_argument(
        "--table_config",
        type=str,
        help="YAML/JSON mapping: {path_name: {label: csv_path}}",
    )
    parser.add_argument("--output_dir", type=str, default="paper_figures")
    parser.add_argument("--font", type=str, default="Times New Roman")
    args = parser.parse_args()

    set_academic_style(args.font)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs: Dict[str, pd.DataFrame] = {}
    if args.jncc_csv and args.nnc_csv and args.traditional_csv:
        runs = {
            "J-NNC": load_run(resolve_csv_path(args.jncc_csv), "J-NNC"),
            "NNC": load_run(resolve_csv_path(args.nnc_csv), "NNC"),
            "Traditional": load_run(resolve_csv_path(args.traditional_csv), "Traditional"),
        }
        plot_s_shape_comparison(runs, output_dir)

    heatmap_csv = args.heatmap_csv or args.jncc_csv
    if heatmap_csv:
        plot_velocity_heatmap(load_run(resolve_csv_path(heatmap_csv), "Heatmap"), output_dir)

    fig11_csv = args.fig11_csv or args.jncc_csv
    if fig11_csv:
        plot_kcm_dual_axis(load_run(resolve_csv_path(fig11_csv), "J-NNC"), output_dir)

    table_cfg = parse_table_config(args.table_config)
    if not table_cfg and args.jncc_csv and args.nnc_csv and args.traditional_csv:
        table_cfg = {
            "s_shape": {
                "J-NNC": resolve_csv_path(args.jncc_csv),
                "NNC": resolve_csv_path(args.nnc_csv),
                "Traditional": resolve_csv_path(args.traditional_csv),
            }
        }
        if args.heatmap_csv and args.heatmap_csv != args.jncc_csv:
            table_cfg["butterfly"] = {
                "J-NNC": resolve_csv_path(args.heatmap_csv),
            }
    print_table(table_cfg)


if __name__ == "__main__":
    main()
