import base64
import copy
import io
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import torch
import yaml

from main import _build_path, load_config
from src.algorithms.baselines import NNCAgent
from src.algorithms.ppo import PPOContinuous
from src.environment import Env
from src.utils.logger import DataLogger
from src.utils.metrics import PaperMetrics

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
CONFIG_DIR = BASE_DIR / "configs"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
MAIN_SCRIPT = BASE_DIR / "main.py"
PYTHON_CMD = ROOT_DIR / "python.cmd"

SCENARIOS: Dict[str, Path] = {
    "Line (直线)": CONFIG_DIR / "default.yaml",
    "Square (正方形)": CONFIG_DIR / "default.yaml",
    "S-shape (S形)": CONFIG_DIR / "default.yaml",
}

PATH_TYPES: List[str] = ["line", "square", "s_shape"]

KCM_FIELDS: List[Tuple[str, str]] = [
    ("MAX_VEL", "最大线速度 MAX_VEL"),
    ("MAX_ACC", "最大线加速度 MAX_ACC"),
    ("MAX_JERK", "最大线跃度 MAX_JERK"),
    ("MAX_ANG_VEL", "最大角速度 MAX_ANG_VEL"),
    ("MAX_ANG_ACC", "最大角加速度 MAX_ANG_ACC"),
    ("MAX_ANG_JERK", "最大角跃度 MAX_ANG_JERK"),
]

st.set_page_config(page_title="Trajectory Industrial Dashboard", layout="wide")


# --------------------------------------------------------------------------------------
# Sidebar Utilities
# --------------------------------------------------------------------------------------
def ensure_state_defaults() -> None:
    st.session_state.setdefault("is_training", False)
    st.session_state.setdefault("train_pid", None)
    st.session_state.setdefault("log_dir", None)
    st.session_state.setdefault("config_path", None)
    st.session_state.setdefault("experiment_name", None)
    st.session_state.setdefault("paper_results", None)


def resolve_python() -> str:
    if PYTHON_CMD.exists():
        return str(PYTHON_CMD)
    return sys.executable or "python"


def read_live_csv(path: Path) -> pd.DataFrame:
    """无缓存读取CSV，锁定或空文件时返回空表。"""
    if not path or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except (pd.errors.EmptyDataError, pd.errors.ParserError, PermissionError, OSError):
        return pd.DataFrame()


def _build_env_from_config(config: dict, device: torch.device = torch.device("cpu")) -> Env:
    env_cfg = config["environment"]
    kcm_cfg = config["kinematic_constraints"]
    path_cfg = config["path"]
    reward_weights = config.get("reward_weights", {})
    pm_points = _build_path(path_cfg)
    return Env(
        device=device,
        epsilon=env_cfg["epsilon"],
        interpolation_period=env_cfg["interpolation_period"],
        MAX_VEL=kcm_cfg["MAX_VEL"],
        MAX_ACC=kcm_cfg["MAX_ACC"],
        MAX_JERK=kcm_cfg["MAX_JERK"],
        MAX_ANG_VEL=kcm_cfg["MAX_ANG_VEL"],
        MAX_ANG_ACC=kcm_cfg["MAX_ANG_ACC"],
        MAX_ANG_JERK=kcm_cfg["MAX_ANG_JERK"],
        Pm=pm_points,
        max_steps=env_cfg["max_steps"],
        lookahead_points=env_cfg.get("lookahead_points", 5),
        reward_weights=reward_weights,
    )


def _apply_path_override(config: dict, path_override: Optional[dict]) -> dict:
    """复制配置并应用路径覆盖项（仅用于前端/启动前的动态调整）。"""
    if not path_override:
        return config
    cfg = copy.deepcopy(config)
    override_path = path_override.get("path") if isinstance(path_override, dict) else None
    if override_path:
        merged_path = copy.deepcopy(cfg.get("path", {}))
        merged_path.update(override_path)
        cfg["path"] = merged_path
    return cfg


def _build_geometry_from_config(config: dict) -> Dict[str, object]:
    env = _build_env_from_config(config)
    ref_points = [(float(p[0]), float(p[1])) for p in env.Pm]
    pl = [(float(p[0]), float(p[1])) for p in env.cache.get("Pl", [])]
    pr = [(float(p[0]), float(p[1])) for p in env.cache.get("Pr", [])]

    xs = [p[0] for p in ref_points]
    ys = [p[1] for p in ref_points]
    x_range = [min(xs) - 0.5, max(xs) + 0.5] if xs else None
    y_range = [min(ys) - 0.5, max(ys) + 0.5] if ys else None

    return {"ref_points": ref_points, "pl": pl, "pr": pr, "ranges": {"x": x_range, "y": y_range}}


def load_reference_geometry(config_path: Path, path_override: Optional[dict] = None) -> Dict[str, object]:
    use_cache = path_override is None
    cache_key = f"ref_geom::{config_path}"
    cached = st.session_state.get(cache_key) if use_cache else None
    if cached:
        return cached

    config, _ = load_config(str(config_path))
    config = _apply_path_override(config, path_override)
    result = _build_geometry_from_config(config)
    if use_cache:
        st.session_state[cache_key] = result
    return result


def _safe_log_dir(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    log_dir = Path(path)
    return log_dir if log_dir.exists() else None


def _find_latest_log_dir() -> Optional[Path]:
    if not SAVED_MODELS_DIR.exists():
        return None
    candidates: List[Tuple[float, Path]] = []
    for exp_dir in SAVED_MODELS_DIR.iterdir():
        if not exp_dir.is_dir():
            continue
        for run_dir in exp_dir.iterdir():
            log_dir = run_dir / "logs"
            if log_dir.exists():
                candidates.append((log_dir.stat().st_mtime, log_dir))
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: x[0], reverse=True)[0][1]


def _latest_experiment_name(config_path: Path) -> str:
    label = config_path.stem.replace(" ", "_")
    return f"exp_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _render_path_override_form(config: dict, path_type: str) -> dict:
    """渲染路径参数表单，返回覆盖后的 path 片段。"""
    path_cfg = config.get("path", {})

    st.sidebar.markdown("##### 轨迹配置")
    scale = st.sidebar.number_input("路径尺度 (scale/length/side)", value=float(path_cfg.get("scale", 10.0)), step=0.5)
    num_points = int(
        st.sidebar.number_input("采样点数", min_value=10, max_value=20000, value=int(path_cfg.get("num_points", 200)), step=10)
    )

    override = {"path": {"type": path_type, "scale": float(scale), "num_points": num_points}}

    if path_type == "line":
        base_angle = float(path_cfg.get("line", {}).get("angle", 0.0))
        angle = st.sidebar.number_input("线段角度 (rad)", value=base_angle, step=0.1, format="%.4f")
        override["path"]["line"] = {"angle": angle}
    elif path_type == "s_shape":
        s_cfg = path_cfg.get("s_shape", {})
        amp = st.sidebar.number_input("S形振幅", value=float(s_cfg.get("amplitude", scale / 2)), step=0.5)
        periods = st.sidebar.number_input("周期数", value=float(s_cfg.get("periods", 2.0)), step=0.5)
        override["path"]["s_shape"] = {"amplitude": amp, "periods": periods}
    elif path_type == "square":
        # 正方形无需额外参数；侧边输入用 scale 表示边长
        override["path"]["square"] = {}

    return override


# --------------------------------------------------------------------------------------
# Training Logic
# --------------------------------------------------------------------------------------
def start_training(
    config_path: Path,
    experiment_name: str,
    disable_kcm: bool,
    disable_smooth: bool,
    kcm_overrides: Dict[str, float],
    path_override: Optional[dict] = None,
) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = SAVED_MODELS_DIR / experiment_name / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=True)

    runtime_config_path = config_path
    if path_override:
        try:
            base_config, _ = load_config(str(config_path))
            merged_config = _apply_path_override(base_config, path_override)
            runtime_config_path = experiment_dir / "config.runtime.yaml"
            with runtime_config_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(merged_config, f, allow_unicode=True, sort_keys=False)
        except Exception as exc:
            st.warning(f"写入覆盖后的配置失败，将使用原始配置：{exc}")
            runtime_config_path = config_path

    cmd: List[str] = [
        resolve_python(),
        str(MAIN_SCRIPT),
        "--mode",
        "train",
        "--config",
        str(runtime_config_path),
        "--experiment_name",
        experiment_name,
        "--experiment_dir",
        str(experiment_dir),
    ]
    if disable_kcm:
        cmd.extend(["--use_kcm", "False"])
    if disable_smooth:
        cmd.extend(["--use_smoothness_reward", "False"])

    flag_map = {
        "MAX_VEL": "max_vel",
        "MAX_ACC": "max_acc",
        "MAX_JERK": "max_jerk",
        "MAX_ANG_VEL": "max_ang_vel",
        "MAX_ANG_ACC": "max_ang_acc",
        "MAX_ANG_JERK": "max_ang_jerk",
    }
    for cfg_key, value in kcm_overrides.items():
        flag = flag_map.get(cfg_key)
        if flag is not None:
            cmd.extend([f"--{flag}", str(value)])

    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception as exc:
        st.error(f"启动训练失败: {exc}")
        return

    st.session_state["is_training"] = True
    st.session_state["train_pid"] = process.pid
    st.session_state["log_dir"] = str(experiment_dir / "logs")
    st.session_state["config_path"] = str(runtime_config_path)
    st.session_state["active_path_override"] = path_override or None
    st.session_state["experiment_name"] = experiment_name
    st.session_state["command_line"] = " ".join(cmd)
    st.success(f"训练已启动 (PID: {process.pid})")


def stop_training() -> None:
    pid = st.session_state.get("train_pid")
    if pid:
        try:
            os.kill(int(pid), signal.SIGTERM)
        except Exception as exc:
            st.warning(f"终止进程时出现问题: {exc}")
    st.session_state["is_training"] = False
    st.session_state["train_pid"] = None
    st.session_state["log_dir"] = None
    st.session_state["command_line"] = None
    st.session_state["active_path_override"] = None


def _build_reward_loss_fig(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if not df.empty and {"episode_idx", "reward"}.issubset(df.columns):
        fig.add_trace(
            go.Scatter(x=df["episode_idx"], y=df["reward"], name="Reward", mode="lines+markers", line=dict(color="#0b7285")),
            secondary_y=False,
        )
    if not df.empty and {"episode_idx", "actor_loss"}.issubset(df.columns):
        fig.add_trace(
            go.Scatter(x=df["episode_idx"], y=df["actor_loss"], name="Actor Loss", mode="lines", line=dict(color="#f59f00")),
            secondary_y=True,
        )
    if not df.empty and {"episode_idx", "critic_loss"}.issubset(df.columns):
        fig.add_trace(
            go.Scatter(x=df["episode_idx"], y=df["critic_loss"], name="Critic Loss", mode="lines", line=dict(color="#c92a2a")),
            secondary_y=True,
        )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_yaxes(title_text="Reward", secondary_y=False)
    fig.update_yaxes(title_text="Loss", secondary_y=True)
    return fig


def _build_trajectory_fig(traj_df: pd.DataFrame, geom: Dict[str, object]) -> go.Figure:
    ref_points: List[Tuple[float, float]] = geom.get("ref_points", []) if geom else []
    pl: List[Tuple[float, float]] = geom.get("pl", []) if geom else []
    pr: List[Tuple[float, float]] = geom.get("pr", []) if geom else []
    ranges = geom.get("ranges", {}) if geom else {}

    fig = go.Figure()

    if pl and pr:
        band_x = [p[0] for p in pl] + [p[0] for p in pr][::-1]
        band_y = [p[1] for p in pl] + [p[1] for p in pr][::-1]
        fig.add_trace(
            go.Scatter(
                x=band_x,
                y=band_y,
                fill="toself",
                fillcolor="rgba(33, 150, 243, 0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Tolerance Band",
                hoverinfo="skip",
            )
        )

    if ref_points:
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in ref_points],
                y=[p[1] for p in ref_points],
                mode="lines",
                line=dict(color="#1f77b4", dash="dash"),
                name="Reference Path",
            )
        )

    if not traj_df.empty and {"x", "y"}.issubset(traj_df.columns):
        fig.add_trace(
            go.Scatter(
                x=traj_df["x"],
                y=traj_df["y"],
                mode="lines+markers",
                marker=dict(size=4, color="#e03131"),
                line=dict(color="#e03131", width=2),
                name="Agent Trajectory",
            )
        )

    x_range = ranges.get("x")
    y_range = ranges.get("y")
    fig.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="X",
        yaxis_title="Y",
        xaxis_range=x_range,
        yaxis_range=y_range,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def load_live_data(log_dir: Optional[Path], config_path: Path, path_override: Optional[dict] = None) -> Dict[str, object]:
    if log_dir is None:
        return {
            "training": pd.DataFrame(),
            "paper": pd.DataFrame(),
            "trajectory": pd.DataFrame(),
            "geom": load_reference_geometry(config_path, path_override),
        }

    training_df = read_live_csv(log_dir / "training_log.csv")

    paper_path = next(log_dir.glob("paper_metrics*.csv"), None) if log_dir.exists() else None
    paper_df = read_live_csv(paper_path) if paper_path else pd.DataFrame()

    traj_df = read_live_csv(log_dir / "latest_trajectory.csv")

    config_copy = log_dir.parent / "config.yaml"
    geom_path = config_copy if config_copy.exists() else config_path
    geom = load_reference_geometry(geom_path, None if config_copy.exists() else path_override)

    return {"training": training_df, "paper": paper_df, "trajectory": traj_df, "geom": geom}


def render_training_sidebar() -> Dict[str, object]:
    st.sidebar.markdown("### 训练监控 · Training Ops")
    scenario = st.sidebar.selectbox("场景选择", list(SCENARIOS.keys()))
    config_path = SCENARIOS[scenario]
    path_type_map = {
        "Line (直线)": "line",
        "Square (正方形)": "square",
        "S-shape (S形)": "s_shape",
    }
    selected_path_type = path_type_map.get(scenario, "line")

    config, _ = load_config(str(config_path))
    default_name = st.session_state.get("experiment_name") or _latest_experiment_name(config_path)
    experiment_name = st.sidebar.text_input("实验名称", value=default_name)
    if not experiment_name.strip():
        experiment_name = _latest_experiment_name(config_path)
    st.session_state["experiment_name"] = experiment_name

    disable_kcm = st.sidebar.checkbox("Disable KCM", value=False)
    disable_smooth = st.sidebar.checkbox("Disable Smoothness", value=False)

    kcm_overrides: Dict[str, float] = {}
    with st.sidebar.expander("KCM 参数微调", expanded=False):
        kcm_cfg = config.get("kinematic_constraints", {})
        for cfg_key, label in KCM_FIELDS:
            base_value = float(kcm_cfg.get(cfg_key, 0.0))
            val = st.number_input(label, value=base_value, step=0.1, format="%.4f", key=f"kcm_{cfg_key}")
            kcm_overrides[cfg_key] = float(val)

    path_override = _render_path_override_form(config, selected_path_type)

    col_start, col_stop = st.sidebar.columns(2)
    if col_start.button("🚀 启动训练 (Start)", width='stretch'):
        start_training(config_path, experiment_name, disable_kcm, disable_smooth, kcm_overrides, path_override)
    if col_stop.button("🛑 停止训练 (Stop)", width='stretch'):
        stop_training()

    active_log_dir = _safe_log_dir(st.session_state.get("log_dir")) or _find_latest_log_dir()
    st.sidebar.caption(f"日志目录: {active_log_dir}" if active_log_dir else "日志目录: 未找到")
    if st.session_state.get("train_pid"):
        st.sidebar.success(f"运行中 PID: {st.session_state['train_pid']}")

    return {"config_path": config_path, "log_dir": active_log_dir, "path_override": path_override}


def render_training_view() -> None:
    sidebar_state = render_training_sidebar()
    config_path: Path = sidebar_state["config_path"]
    log_dir: Optional[Path] = sidebar_state["log_dir"]
    path_override = sidebar_state.get("path_override")

    # 训练中使用运行时配置，预览时使用当前选择+覆盖
    effective_config_path = Path(st.session_state.get("config_path") or config_path)
    if not st.session_state.get("is_training"):
        effective_config_path = config_path
    active_override = st.session_state.get("active_path_override") if st.session_state.get("is_training") else path_override

    data = load_live_data(log_dir, effective_config_path, active_override)
    training_df: pd.DataFrame = data["training"]
    paper_df: pd.DataFrame = data["paper"]
    traj_df: pd.DataFrame = data["trajectory"]
    geom = data["geom"]

    current_episode = "-"
    if not training_df.empty and "episode_idx" in training_df.columns:
        current_episode = int(training_df["episode_idx"].max())

    last_reward = "-"
    if not training_df.empty and "reward" in training_df.columns:
        last_reward = float(training_df["reward"].iloc[-1])

    mean_error = "-"
    if not paper_df.empty and "rmse_error" in paper_df.columns:
        mean_error = float(paper_df["rmse_error"].iloc[-1])

    m1, m2, m3 = st.columns(3)
    m1.metric("Current Episode", current_episode)
    m2.metric("Last Reward", last_reward)
    m3.metric("Mean Error (RMSE)", mean_error)

    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.markdown("#### Reward & Loss")
        reward_fig = _build_reward_loss_fig(training_df)
        st.plotly_chart(reward_fig, width='stretch')
    with col_right:
        st.markdown("#### Real-time Trajectory")
        traj_fig = _build_trajectory_fig(traj_df, geom)
        st.plotly_chart(traj_fig, width='stretch')

    if st.session_state.get("is_training"):
        time.sleep(1)
        rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
        if rerun:
            rerun()


# --------------------------------------------------------------------------------------
# Paper Logic
# --------------------------------------------------------------------------------------
def list_best_models() -> List[Path]:
    models: List[Tuple[float, Path]] = []
    for best_path in SAVED_MODELS_DIR.glob("**/checkpoints/best_model.pth"):
        try:
            mtime = best_path.stat().st_mtime
            models.append((mtime, best_path))
        except OSError:
            continue
    return [p for _, p in sorted(models, key=lambda x: x[0], reverse=True)]


def _load_effective_config(model_path: Path) -> Tuple[dict, Path]:
    candidate = model_path.parent.parent / "config.yaml"
    if candidate.exists():
        return load_config(str(candidate))[0], candidate
    fallback = CONFIG_DIR / "s_shape.yaml"
    st.warning("未找到模型同目录下的 config.yaml，已回退到 s_shape.yaml。")
    return load_config(str(fallback))[0], fallback


def _build_agent(config: dict, env: Env, device: torch.device):
    exp_cfg = config.get("experiment", {})
    kcm_cfg = config["kinematic_constraints"]
    ppo_cfg = config["ppo"]
    obs_space = getattr(env, "observation_space", None)
    act_space = getattr(env, "action_space", None)
    disable_kcm = exp_cfg.get("enable_kcm") is False or exp_cfg.get("mode") == "ablation_no_kcm"

    if disable_kcm:
        return NNCAgent(
            state_dim=None,
            hidden_dim=ppo_cfg["hidden_dim"],
            action_dim=None,
            actor_lr=ppo_cfg["actor_lr"],
            critic_lr=ppo_cfg["critic_lr"],
            lmbda=ppo_cfg["lmbda"],
            epochs=ppo_cfg["epochs"],
            eps=ppo_cfg["eps"],
            gamma=ppo_cfg["gamma"],
            ent_coef=ppo_cfg.get("ent_coef", 0.0),
            device=device,
            max_vel=kcm_cfg["MAX_VEL"],
            max_ang_vel=kcm_cfg["MAX_ANG_VEL"],
            observation_space=obs_space,
            action_space=act_space,
        )
    return PPOContinuous(
        state_dim=None,
        hidden_dim=ppo_cfg["hidden_dim"],
        action_dim=None,
        actor_lr=ppo_cfg["actor_lr"],
        critic_lr=ppo_cfg["critic_lr"],
        lmbda=ppo_cfg["lmbda"],
        epochs=ppo_cfg["epochs"],
        eps=ppo_cfg["eps"],
        gamma=ppo_cfg["gamma"],
        ent_coef=ppo_cfg.get("ent_coef", 0.0),
        device=device,
        observation_space=obs_space,
        action_space=act_space,
    )


def _load_checkpoint(agent, model_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if "actor" in checkpoint and hasattr(agent, "actor"):
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.actor.eval()
    if "critic" in checkpoint and hasattr(agent, "critic"):
        agent.critic.load_state_dict(checkpoint["critic"])
        agent.critic.eval()


def _rollout_episode(env: Env, agent) -> Tuple[dict, Dict[str, List[float]]]:
    state = env.reset()
    paper_metrics = PaperMetrics()
    trace: Dict[str, List[float]] = {
        "timestamp": [],
        "position_x": [],
        "position_y": [],
        "reference_x": [],
        "reference_y": [],
        "velocity": [],
        "acceleration": [],
        "jerk": [],
        "contour_error": [],
        "kcm_intervention": [],
    }

    done = False
    step_idx = 0
    dt = env.interpolation_period

    with torch.no_grad():
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, info = env.step(action)
            ref_point = DataLogger.project_to_path(
                position=env.current_position,
                path_points=env.Pm,
                segment_index=info.get("segment_idx", getattr(env, "current_segment_idx", 0)),
            )
            trace["timestamp"].append(step_idx * dt)
            trace["position_x"].append(float(env.current_position[0]))
            trace["position_y"].append(float(env.current_position[1]))
            trace["reference_x"].append(float(ref_point[0]))
            trace["reference_y"].append(float(ref_point[1]))
            trace["velocity"].append(float(env.velocity))
            trace["acceleration"].append(float(env.acceleration))
            trace["jerk"].append(float(env.jerk))
            trace["contour_error"].append(float(info.get("contour_error", 0.0)))
            trace["kcm_intervention"].append(float(info.get("kcm_intervention", 0.0)))

            paper_metrics.update(
                contour_error=info["contour_error"],
                jerk=info["jerk"],
                velocity=env.velocity,
                kcm_intervention=info["kcm_intervention"],
            )
            state = next_state
            step_idx += 1

    metrics = paper_metrics.compute()
    metrics["max_error"] = float(max(trace["contour_error"])) if trace["contour_error"] else 0.0
    return metrics, trace


def _summarize_metrics(metrics_list: List[dict]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for key in ("rmse_error", "max_error", "mean_jerk"):
        values = [m.get(key, 0.0) for m in metrics_list]
        summary[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return summary


def _latex_table(summary: Dict[str, Dict[str, float]]) -> str:
    labels = {"rmse_error": "RMSE", "max_error": "Max Error", "mean_jerk": "Mean Jerk"}
    lines = [
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Metric & Mean & Std \\\\",
        "\\midrule",
    ]
    for key in ("rmse_error", "max_error", "mean_jerk"):
        stats = summary.get(key, {"mean": 0.0, "std": 0.0})
        lines.append(f"{labels[key]} & {stats['mean']:.4f} & {stats['std']:.4f} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def _make_fig1_matplotlib(trace: Dict[str, List[float]], geom: Dict[str, object]):
    ref_points: List[Tuple[float, float]] = geom.get("ref_points", [])
    pl: List[Tuple[float, float]] = geom.get("pl", [])
    pr: List[Tuple[float, float]] = geom.get("pr", [])

    fig, ax = plt.subplots(figsize=(6, 6))

    if pl and pr:
        band_x = [p[0] for p in pl] + [p[0] for p in pr][::-1]
        band_y = [p[1] for p in pl] + [p[1] for p in pr][::-1]
        ax.fill(band_x, band_y, color="#2196f3", alpha=0.15, label="Tolerance Band")

    if ref_points:
        ax.plot([p[0] for p in ref_points], [p[1] for p in ref_points], "--", color="#1f77b4", linewidth=1.5, label="Reference Path")

    if trace.get("position_x") and trace.get("position_y"):
        ax.plot(trace["position_x"], trace["position_y"], color="#e03131", linewidth=2.2, label="Agent Trajectory")

    xs = [p[0] for p in ref_points] or trace.get("position_x", [])
    ys = [p[1] for p in ref_points] or trace.get("position_y", [])
    if xs and ys:
        ax.set_xlim([min(xs) - 1, max(xs) + 1])
        ax.set_ylim([min(ys) - 1, max(ys) + 1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Fig1 · Trajectory vs Reference")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig


def _make_fig2_matplotlib(trace: Dict[str, List[float]], kcm_cfg: dict):
    t = trace.get("timestamp", [])
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 7.5), sharex=True)
    axes[0].plot(t, trace.get("velocity", []), color="#1f77b4")
    axes[0].axhline(kcm_cfg["MAX_VEL"], color="red", linestyle="--", label="Vel Limit")
    axes[0].set_ylabel("Velocity")

    axes[1].plot(t, trace.get("acceleration", []), color="#2ca02c")
    axes[1].axhline(kcm_cfg["MAX_ACC"], color="red", linestyle="--", label="Acc Limit")
    axes[1].set_ylabel("Acceleration")

    axes[2].plot(t, trace.get("jerk", []), color="#d62728")
    axes[2].axhline(kcm_cfg["MAX_JERK"], color="red", linestyle="--", label="Jerk Limit")
    axes[2].axhline(-kcm_cfg["MAX_JERK"], color="red", linestyle=":")
    axes[2].set_ylabel("Jerk")
    axes[2].set_xlabel("Time (s)")

    for ax in axes:
        ax.grid(True, linestyle=":", alpha=0.5)
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    return fig


def _fig_to_pdf_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def _pdf_iframe(pdf_bytes: bytes, height: int = 480) -> str:
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    return f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="{height}" type="application/pdf"></iframe>'


def render_paper_sidebar(models: List[Path]) -> Dict[str, object]:
    st.sidebar.markdown("### 论文评估 · Paper Mode")
    model_labels = [str(p.relative_to(SAVED_MODELS_DIR)) for p in models] if models else ["未找到 best_model.pth"]
    model_choice = st.sidebar.selectbox("选择 best_model.pth", model_labels, index=0)
    model_path = models[model_labels.index(model_choice)] if models else None

    runs = int(st.sidebar.number_input("评估次数 (Batch Size)", min_value=1, max_value=50, value=20, step=1))
    device_choice = st.sidebar.selectbox("计算设备", ["auto", "cpu", "cuda"], index=0)

    return {"model_path": model_path, "runs": runs, "device_choice": device_choice}


def render_paper_view() -> None:
    models = list_best_models()
    sidebar_state = render_paper_sidebar(models)
    model_path: Optional[Path] = sidebar_state["model_path"]
    runs: int = sidebar_state["runs"]
    device_choice: str = sidebar_state["device_choice"]

    if st.sidebar.button("开始批量评估 (Start Evaluation)", width='stretch') and model_path:
        st.session_state["paper_results"] = None
        with st.spinner("评估中..."):
            try:
                config, cfg_path = _load_effective_config(model_path)
                device = torch.device(
                    "cuda" if (device_choice == "cuda" or (device_choice == "auto" and torch.cuda.is_available())) else "cpu"
                )
                if device_choice == "cuda" and not torch.cuda.is_available():
                    st.warning("CUDA 不可用，已自动切换到 CPU。")

                env = _build_env_from_config(config, device)
                agent = _build_agent(config, env, device)
                _load_checkpoint(agent, model_path, device)

                metrics_list: List[dict] = []
                sample_trace: Optional[Dict[str, List[float]]] = None
                for _ in range(int(runs)):
                    metrics, trace = _rollout_episode(env, agent)
                    metrics_list.append(metrics)
                    if sample_trace is None:
                        sample_trace = trace

                summary = _summarize_metrics(metrics_list)
                latex_text = _latex_table(summary)

                geom = load_reference_geometry(cfg_path)
                fig1 = _make_fig1_matplotlib(sample_trace or {}, geom)
                fig2 = _make_fig2_matplotlib(sample_trace or {}, config["kinematic_constraints"])
                fig1_pdf = _fig_to_pdf_bytes(fig1)
                fig2_pdf = _fig_to_pdf_bytes(fig2)
                plt.close(fig1)
                plt.close(fig2)

                st.session_state["paper_results"] = {
                    "summary": summary,
                    "latex": latex_text,
                    "fig1_pdf": fig1_pdf,
                    "fig2_pdf": fig2_pdf,
                    "fig1_iframe": _pdf_iframe(fig1_pdf, height=520),
                    "fig2_iframe": _pdf_iframe(fig2_pdf, height=520),
                    "model_path": str(model_path),
                    "config_path": str(cfg_path),
                }
            except Exception as exc:
                st.error(f"评估失败: {exc}")

    results = st.session_state.get("paper_results")
    st.subheader("论文评估结果")
    if not results:
        st.info("点击左侧“开始批量评估”生成结果。")
        return

    summary_df = pd.DataFrame(results["summary"]).T.rename(columns={"mean": "Mean", "std": "Std"})
    st.dataframe(summary_df, width='stretch')
    st.text_area("LaTeX 表格", results["latex"], height=140)
    st.caption(f"模型: {results['model_path']} | 配置: {results['config_path']}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Fig1 轨迹对比 (PDF 预览)")
        st.markdown(results["fig1_iframe"], unsafe_allow_html=True)
        st.download_button("下载 Fig1 (PDF)", data=results["fig1_pdf"], file_name="fig1_trajectory.pdf")
    with col2:
        st.markdown("##### Fig2 运动学曲线 (PDF 预览)")
        st.markdown(results["fig2_iframe"], unsafe_allow_html=True)
        st.download_button("下载 Fig2 (PDF)", data=results["fig2_pdf"], file_name="fig2_dynamics.pdf")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main() -> None:
    ensure_state_defaults()
    mode = st.sidebar.radio("系统模式", ["训练监控 (Training Ops)", "论文评估 (Paper Mode)"], index=0)
    if mode == "训练监控 (Training Ops)":
        render_training_view()
    else:
        render_paper_view()


if __name__ == "__main__":
    main()
