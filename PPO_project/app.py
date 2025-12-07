import os
import signal
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import torch

from main import _build_path, load_config
from src.algorithms.baselines import NNCAgent
from src.algorithms.ppo import PPOContinuous
from src.environment import Env
from src.utils.logger import DataLogger
from src.utils.metrics import PaperMetrics

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "configs"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
MAIN_SCRIPT = BASE_DIR / "main.py"

CONFIG_OPTIONS: Dict[str, Path] = {
    "S-Shape Curve (S形曲线)": CONFIG_DIR / "s_shape.yaml",
    "Butterfly Curve (蝴蝶曲线)": CONFIG_DIR / "butterfly.yaml",
}

KCM_FIELDS: List[Tuple[str, str, str]] = [
    ("MAX_VEL", "max_vel", "Max Velocity (线速度)"),
    ("MAX_ACC", "max_acc", "Max Acceleration (线加速度)"),
    ("MAX_JERK", "max_jerk", "Max Jerk (线跃度)"),
    ("MAX_ANG_VEL", "max_ang_vel", "Max Angular Velocity (角速度)"),
    ("MAX_ANG_ACC", "max_ang_acc", "Max Angular Acceleration (角加速度)"),
    ("MAX_ANG_JERK", "max_ang_jerk", "Max Angular Jerk (角跃度)"),
]

st.set_page_config(page_title="Trajectory Master Dashboard", layout="wide")


@st.cache_data(show_spinner=False)
def _load_kcm_defaults(config_path: str, mtime: float) -> Dict[str, float]:
    config, _ = load_config(config_path)
    return config.get("kinematic_constraints", {})


def init_session_state() -> None:
    """Initialize session state for multi-process tracking."""
    st.session_state.setdefault("running_processes", [])
    if "last_launch" in st.session_state:
        st.session_state.pop("last_launch")


def _trigger_rerun() -> None:
    rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rerun:
        rerun()


def _remove_process(pid: str) -> None:
    st.session_state["running_processes"] = [p for p in st.session_state.get("running_processes", []) if p.get("pid") != pid]


def kill_process(pid_value: str) -> None:
    """Terminate a running process by PID and remove it from session state."""
    if not pid_value:
        return
    init_session_state()
    try:
        pid = int(pid_value)
    except ValueError:
        st.error(f"无法解析 PID: {pid_value}")
        return

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        st.warning(f"进程 {pid} 已不存在，已从列表中移除。")
    except Exception as exc:  # pragma: no cover - defensive UI path
        st.error(f"终止任务失败: {exc}")
        return

    _remove_process(str(pid))
    _trigger_rerun()


def generate_experiment_name(trajectory_label: str, disable_kcm: bool, disable_smooth: bool, customized: bool = False) -> str:
    safe_label = trajectory_label.split("(")[0].strip().replace(" ", "_").replace("-", "_")
    tags: List[str] = []
    if disable_kcm:
        tags.append("NoKCM")
    if disable_smooth:
        tags.append("NoSmooth")
    tag = "_".join(tags) if tags else "Full"
    base = f"exp_{safe_label}_{tag}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    return f"{base}_Custom" if customized else base


def ensure_experiment_name(auto_name: str) -> str:
    if "experiment_name_user_edit" not in st.session_state:
        st.session_state["experiment_name_user_edit"] = False
    if not st.session_state["experiment_name_user_edit"]:
        st.session_state["experiment_name"] = auto_name

    def _mark_user_edit() -> None:
        st.session_state["experiment_name_user_edit"] = True

    exp_name = st.text_input(
        "Experiment Name",
        value=st.session_state.get("experiment_name", auto_name),
        on_change=_mark_user_edit,
        key="experiment_name_input",
    )
    if st.button("重置自动命名", type="secondary"):
        st.session_state["experiment_name_user_edit"] = False
        st.session_state["experiment_name"] = auto_name
        exp_name = auto_name
    return exp_name or auto_name


def list_experiment_runs() -> Dict[str, List[Path]]:
    experiments: Dict[str, List[Path]] = {}
    if not SAVED_MODELS_DIR.exists():
        return experiments
    for exp_dir in SAVED_MODELS_DIR.iterdir():
        if not exp_dir.is_dir():
            continue
        runs = sorted([p for p in exp_dir.iterdir() if p.is_dir()], reverse=True)
        if runs:
            experiments[exp_dir.name] = runs
    return experiments


def pick_log_file(run_dir: Path) -> Optional[Path]:
    logs_dir = run_dir / "logs"
    primary = logs_dir / "training_log.csv"
    if primary.exists():
        return primary
    candidates = sorted(logs_dir.glob("training_*.csv"), reverse=True)
    return candidates[0] if candidates else None


def format_command(cmd: List[str]) -> str:
    return " ".join(map(str, cmd))


def launch_training_process(
    config_path: Path,
    experiment_name: str,
    disable_kcm: bool,
    disable_smooth: bool,
    resume_path: str,
    force_gpu: bool,
    kcm_overrides: Optional[Dict[str, float]] = None,
) -> Optional[Dict[str, str]]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = SAVED_MODELS_DIR / experiment_name / timestamp
    cmd = [
        sys.executable,
        str(MAIN_SCRIPT),
        "--mode",
        "train",
        "--config",
        str(config_path),
        "--experiment_name",
        experiment_name,
        "--experiment_dir",
        str(experiment_dir),
    ]
    if disable_kcm:
        cmd.extend(["--use_kcm", "False"])
    if disable_smooth:
        cmd.extend(["--use_smoothness_reward", "False"])
    if resume_path:
        cmd.extend(["--resume", resume_path])
    if kcm_overrides:
        for arg_name, value in kcm_overrides.items():
            if value is not None:
                cmd.extend([f"--{arg_name}", str(value)])

    env = os.environ.copy()
    if force_gpu:
        env.setdefault("CUDA_VISIBLE_DEVICES", env.get("CUDA_VISIBLE_DEVICES", "0"))

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            cwd=str(BASE_DIR),
            env=env,
            start_new_session=True,
        )
    except Exception as exc:  # pragma: no cover - user runtime protection
        st.error(f"启动训练失败: {exc}")
        return None

    init_session_state()
    launch_info = {
        "pid": str(process.pid),
        "cmd": format_command(cmd),
        "experiment_dir": str(experiment_dir),
        "experiment_name": experiment_name,
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    st.session_state["running_processes"].append(launch_info)
    st.session_state["experiment_name_user_edit"] = True
    st.success(f"已启动训练进程 (PID: {process.pid})")
    return launch_info


def render_training_monitor(default_exp: Optional[str] = None, default_run: Optional[Path] = None) -> None:
    st.subheader("训练监控 · Passive Monitoring")
    experiments = list_experiment_runs()
    if not experiments:
        st.info("尚未发现 saved_models 下的训练记录。")
        return

    exp_names = sorted(experiments.keys(), reverse=True)
    default_exp_idx = exp_names.index(default_exp) if default_exp in exp_names else 0
    exp_choice = st.selectbox("选择实验", exp_names, index=default_exp_idx, key="monitor_exp_choice")

    runs = experiments.get(exp_choice, [])
    if not runs:
        st.warning("该实验下暂无时间戳目录。")
        return

    run_labels = [p.name for p in runs]
    default_run_idx = 0
    if default_run:
        try:
            default_run_idx = run_labels.index(default_run.name)
        except ValueError:
            default_run_idx = 0
    run_choice = st.selectbox("选择时间戳", run_labels, index=default_run_idx, key="monitor_run_choice")
    run_dir = runs[run_labels.index(run_choice)]
    log_path = pick_log_file(run_dir)
    st.caption(f"日志文件: {log_path or '未找到 training_log.csv'}")
    st.button("手动刷新日志", key=f"refresh_{exp_choice}_{run_choice}")

    if log_path is None or not log_path.exists():
        st.info("等待日志产生中...")
        return

    df = pd.read_csv(log_path)
    if df.empty:
        st.info("日志文件为空，可能训练尚未写入。")
        return

    df["reward_smooth"] = df["reward"].ewm(alpha=0.1).mean()
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Reward", "Actor/Critic Loss"])
    fig.add_trace(go.Scatter(x=df["episode_idx"], y=df["reward"], name="Reward", line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["episode_idx"], y=df["reward_smooth"], name="Reward (EMA)", line=dict(color="#ff7f0e", dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["episode_idx"], y=df["actor_loss"], name="Actor Loss", line=dict(color="#2ca02c")), row=1, col=2)
    fig.add_trace(go.Scatter(x=df["episode_idx"], y=df["critic_loss"], name="Critic Loss", line=dict(color="#d62728")), row=1, col=2)
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    latest_episode_idx = int(df["episode_idx"].iloc[-1]) if "episode_idx" in df.columns else None
    st.dataframe(df.tail(15), use_container_width=True)
    render_realtime_trajectory(run_dir, latest_episode_idx)


def render_realtime_trajectory(run_dir: Path, latest_episode: Optional[int] = None) -> None:
    traj_path = run_dir / "logs" / "latest_trajectory.csv"
    config_path = run_dir / "config.yaml"

    if not traj_path.exists():
        st.info("等待数据同步：latest_trajectory.csv 尚未生成。")
        return
    if not config_path.exists():
        st.info("未找到 config.yaml，无法绘制参考路径与允差带。")
        return

    st.divider()
    st.subheader(f"最新回合轨迹 (Episode {latest_episode})" if latest_episode is not None else "最新回合轨迹")

    try:
        traj_df = pd.read_csv(traj_path)
        if traj_df.empty or not {"x", "y"}.issubset(traj_df.columns):
            st.info("轨迹文件为空或缺少 x,y 列，等待下一次更新。")
            return

        saved_config, _ = load_config(str(config_path))
        dummy_env = _build_env(saved_config, torch.device("cpu"))
        dummy_env.reset()  # 触发边界计算

        pl = _clean_boundary(dummy_env.cache.get("Pl", [])) if hasattr(dummy_env, "cache") else []
        pr = _clean_boundary(dummy_env.cache.get("Pr", [])) if hasattr(dummy_env, "cache") else []
        ref_points = getattr(dummy_env, "Pm", [])

        traj_fig = go.Figure()
        if pl and pr:
            band_x = [p[0] for p in pl] + [p[0] for p in pr][::-1]
            band_y = [p[1] for p in pl] + [p[1] for p in pr][::-1]
            traj_fig.add_trace(
                go.Scatter(
                    x=band_x,
                    y=band_y,
                    fill="toself",
                    fillcolor="rgba(44,160,44,0.15)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="Tolerance Tube",
                    hoverinfo="skip",
                )
            )

        if ref_points is not None and len(ref_points):
            ref_x = [p[0] for p in ref_points]
            ref_y = [p[1] for p in ref_points]
            traj_fig.add_trace(
                go.Scatter(
                    x=ref_x,
                    y=ref_y,
                    mode="lines",
                    name="Reference",
                    line=dict(dash="dash", color="blue", width=1),
                )
            )

        traj_fig.add_trace(
            go.Scatter(
                x=traj_df["x"],
                y=traj_df["y"],
                mode="lines",
                name="Actual",
                line=dict(color="#e45756", width=2.5),
            )
        )

        traj_fig.update_layout(
            height=550,
            title="Real-time Trajectory Visualization",
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            legend=dict(orientation="h", y=1.1),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(traj_fig, use_container_width=True)
    except Exception as e:  # pragma: no cover - UI 防御
        st.warning(f"可视化渲染挂起 (数据同步中...): {e}")


def _load_effective_config(model_path: Path) -> Tuple[dict, Path]:
    candidate = model_path.parent.parent / "config.yaml"
    if candidate.exists():
        return load_config(str(candidate))[0], candidate
    fallback = CONFIG_DIR / "default.yaml"
    st.warning("未找到模型同目录下的 config.yaml，已回退到默认配置。")
    return load_config(str(fallback))[0], fallback


def _build_env(config: dict, device: torch.device) -> Env:
    env_cfg = config["environment"]
    kcm_cfg = config["kinematic_constraints"]
    path_cfg = config["path"]
    reward_weights = config.get("reward_weights", {})
    Pm = _build_path(path_cfg)
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
        Pm=Pm,
        max_steps=env_cfg["max_steps"],
        lookahead_points=env_cfg.get("lookahead_points", 5),
        reward_weights=reward_weights,
    )


def _build_agent(config: dict, env: Env, device: torch.device):
    exp_cfg = config.get("experiment", {})
    kcm_cfg = config["kinematic_constraints"]
    ppo_cfg = config["ppo"]
    obs_space = getattr(env, "observation_space", None)
    act_space = getattr(env, "action_space", None)
    disable_kcm = exp_cfg.get("enable_kcm") is False or exp_cfg.get("mode") == "ablation_no_kcm"

    if disable_kcm:
        agent = NNCAgent(
            state_dim=None,
            hidden_dim=ppo_cfg["hidden_dim"],
            action_dim=None,
            actor_lr=ppo_cfg["actor_lr"],
            critic_lr=ppo_cfg["critic_lr"],
            lmbda=ppo_cfg["lmbda"],
            epochs=ppo_cfg["epochs"],
            eps=ppo_cfg["eps"],
            gamma=ppo_cfg["gamma"],
            device=device,
            max_vel=kcm_cfg["MAX_VEL"],
            max_ang_vel=kcm_cfg["MAX_ANG_VEL"],
            observation_space=obs_space,
            action_space=act_space,
        )
    else:
        agent = PPOContinuous(
            state_dim=None,
            hidden_dim=ppo_cfg["hidden_dim"],
            action_dim=None,
            actor_lr=ppo_cfg["actor_lr"],
            critic_lr=ppo_cfg["critic_lr"],
            lmbda=ppo_cfg["lmbda"],
            epochs=ppo_cfg["epochs"],
            eps=ppo_cfg["eps"],
            gamma=ppo_cfg["gamma"],
            device=device,
            observation_space=obs_space,
            action_space=act_space,
        )
    return agent


def _load_checkpoint(agent, model_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if "actor" in checkpoint:
        agent.actor.load_state_dict(checkpoint["actor"])
    if "critic" in checkpoint:
        agent.critic.load_state_dict(checkpoint["critic"])
    if hasattr(agent, "actor"):
        agent.actor.eval()
    if hasattr(agent, "critic"):
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


def _render_kcm_tuner(config_path: Path) -> Tuple[Dict[str, float], bool]:
    defaults: Dict[str, float] = {}
    try:
        mtime = config_path.stat().st_mtime
        defaults = _load_kcm_defaults(str(config_path), mtime)
    except FileNotFoundError:
        st.warning(f"配置文件不存在: {config_path}")
    except Exception as exc:  # pragma: no cover - UI防御
        st.warning(f"读取配置失败: {exc}")

    kcm_values: Dict[str, float] = {}
    is_custom = False

    with st.expander("Step 3 · 运动学参数微调", expanded=False):
        st.caption("覆盖 YAML 中的运动学约束，发射时通过 CLI 透传到 main.py。")
        for cfg_key, arg_name, label in KCM_FIELDS:
            base_value = float(defaults.get(cfg_key, 0.0))
            input_value = st.number_input(
                label,
                min_value=0.0,
                value=base_value,
                step=0.1,
                format="%.4f",
                key=f"kcm_{config_path.name}_{arg_name}",
            )
            kcm_values[arg_name] = float(input_value)
            if not is_custom and abs(float(input_value) - base_value) > 1e-9:
                is_custom = True

    return kcm_values, is_custom


def _clean_boundary(boundary: List) -> List[Tuple[float, float]]:
    cleaned: List[Tuple[float, float]] = []
    for item in boundary:
        if not item or len(item) < 2:
            continue
        x, y = item
        if x is None or y is None:
            continue
        cleaned.append((float(x), float(y)))
    return cleaned


def _make_path_fig(trace: Dict[str, List[float]], env: Env):
    pos = np.column_stack([trace["position_x"], trace["position_y"]])
    ref = np.column_stack([trace["reference_x"], trace["reference_y"]])
    pl = _clean_boundary(env.cache.get("Pl", [])) if hasattr(env, "cache") else []
    pr = _clean_boundary(env.cache.get("Pr", [])) if hasattr(env, "cache") else []

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ref[:, 0], y=ref[:, 1], name="Reference", mode="lines", line=dict(color="#4c78a8", dash="dash")))
    if pl and pr and len(pl) == len(pr):
        band_x = [p[0] for p in pl] + [p[0] for p in pr][::-1]
        band_y = [p[1] for p in pl] + [p[1] for p in pr][::-1]
        fig.add_trace(
            go.Scatter(
                x=band_x,
                y=band_y,
                fill="toself",
                fillcolor="rgba(44,160,44,0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Tolerance Tube",
                showlegend=True,
            )
        )
    fig.add_trace(go.Scatter(x=pos[:, 0], y=pos[:, 1], name="Actual", mode="lines", line=dict(color="#e45756", width=3)))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
    return fig


def _make_motion_fig(trace: Dict[str, List[float]], kcm_cfg: dict):
    t = trace["timestamp"]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=["Velocity", "Acceleration", "Jerk"])
    fig.add_trace(go.Scatter(x=t, y=trace["velocity"], name="Velocity", line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=[kcm_cfg["MAX_VEL"]] * len(t), name="Velocity Limit", line=dict(color="red", dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=trace["acceleration"], name="Acceleration", line=dict(color="#2ca02c")), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=[kcm_cfg["MAX_ACC"]] * len(t), name="Acc Limit", line=dict(color="red", dash="dash")), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=trace["jerk"], name="Jerk", line=dict(color="#d62728")), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=[kcm_cfg["MAX_JERK"]] * len(t), name="Jerk Limit", line=dict(color="red", dash="dash")), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=[-kcm_cfg["MAX_JERK"]] * len(t), name="Jerk Limit -", line=dict(color="red", dash="dot")), row=3, col=1)
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=20), legend=dict(orientation="h"))
    return fig


def render_sidebar_process_manager() -> None:
    st.sidebar.subheader("进行中任务 · Running Tasks")
    processes = st.session_state.get("running_processes", [])
    if not processes:
        st.sidebar.caption("暂无运行中的训练。")
        return

    for idx, proc in enumerate(list(processes)):
        pid = proc.get("pid", "?")
        exp_name = proc.get("experiment_name", "Unknown")
        with st.sidebar.expander(f"{exp_name} · PID {pid}", expanded=False):
            st.caption(f"启动时间: {proc.get('start_time', '未知')}")
            st.caption(f"命令: {proc.get('cmd', '')}")
            if st.button("终止任务 (Kill)", key=f"kill_{pid}_{idx}", type="secondary"):
                kill_process(str(pid))


def render_training_ops() -> None:
    st.header("模式 A · Training Ops")
    st.markdown("面向任务的实验向导 + 异步训练发射 + 被动监控。")

    col1, col2 = st.columns(2)
    trajectory_label = col1.selectbox("Step 1 · 选择训练场景 (Trajectory Selection)", list(CONFIG_OPTIONS.keys()))
    disable_kcm = col2.checkbox("Disable KCM (禁用运动学约束)", value=False)
    disable_smooth = col2.checkbox("Disable Smoothness Reward (禁用平滑奖励)", value=False)

    config_path = CONFIG_OPTIONS[trajectory_label]
    kcm_overrides, is_custom_kcm = _render_kcm_tuner(config_path)

    auto_name = generate_experiment_name(trajectory_label, disable_kcm, disable_smooth, customized=is_custom_kcm)
    exp_name = ensure_experiment_name(auto_name)

    with st.expander("Step 4 · 高级选项"):
        resume_path = st.text_input("Resume Checkpoint (.pth)", value="")
        force_gpu = st.checkbox("Force GPU (CUDA_VISIBLE_DEVICES)", value=False)

    st.caption(f"配置映射: {trajectory_label} -> {config_path}")

    if st.button("🚀 Launch Training", type="primary"):
        if not config_path.exists():
            st.error(f"配置文件不存在: {config_path}")
        else:
            launch_training_process(
                config_path=config_path,
                experiment_name=exp_name,
                disable_kcm=disable_kcm,
                disable_smooth=disable_smooth,
                resume_path=resume_path.strip(),
                force_gpu=force_gpu,
                kcm_overrides=kcm_overrides,
            )

    running = st.session_state.get("running_processes", [])
    if running:
        latest = running[-1]
        st.info(
            f"最近启动: {latest.get('experiment_name')} (PID: {latest.get('pid')}) · "
            f"实验目录: {latest.get('experiment_dir')}\n\n命令: `{latest.get('cmd')}`"
        )
        default_exp = latest.get("experiment_name")
        default_run = Path(latest["experiment_dir"]) if latest.get("experiment_dir") else None
    else:
        default_exp, default_run = None, None

    render_training_monitor(default_exp, default_run)


def render_paper_mode() -> None:
    st.header("模式 B · Paper Mode")
    st.markdown("加载 best_model.pth，批量评估并生成论文级图表与 LaTeX。")

    available_models = sorted(SAVED_MODELS_DIR.rglob("best_model.pth"), reverse=True)
    model_options = [str(p.relative_to(BASE_DIR)) for p in available_models]
    model_choice = st.selectbox("选择 best_model.pth", model_options)
    manual_model = st.text_input("或手动填写模型路径", value="")
    target_model = Path(manual_model) if manual_model.strip() else (BASE_DIR / model_choice if model_choice else None)

    runs = st.number_input("Batch Evaluation 次数", min_value=1, max_value=50, value=20, step=1)
    device_choice = st.selectbox("设备", ["auto", "cpu", "cuda"])
    device = torch.device("cuda" if (device_choice == "cuda" or (device_choice == "auto" and torch.cuda.is_available())) else "cpu")

    if st.button("开始批量评估", type="primary") and target_model:
        if not target_model.exists():
            st.error(f"模型不存在: {target_model}")
            return
        try:
            with st.spinner("评估中..."):
                config, cfg_path = _load_effective_config(target_model)
                env = _build_env(config, device)
                agent = _build_agent(config, env, device)
                _load_checkpoint(agent, target_model, device)

                metrics_list: List[dict] = []
                sample_trace: Optional[Dict[str, List[float]]] = None
                for _ in range(int(runs)):
                    metrics, trace = _rollout_episode(env, agent)
                    metrics_list.append(metrics)
                    if sample_trace is None:
                        sample_trace = trace

                summary = _summarize_metrics(metrics_list)
                latex_text = _latex_table(summary)
        except Exception as exc:  # pragma: no cover - UI error path
            st.error(f"模型文件损坏或不匹配: {exc}")
            return

        st.subheader("统计汇总")
        st.write(pd.DataFrame(summary).T.rename(columns={"mean": "Mean", "std": "Std"}))
        st.text_area("LaTeX 表格", latex_text, height=140)
        st.caption(f"配置来源: {cfg_path}")

        if sample_trace:
            st.subheader("论文级图表预览")
            path_fig = _make_path_fig(sample_trace, env)
            motion_fig = _make_motion_fig(sample_trace, config["kinematic_constraints"])
            st.plotly_chart(path_fig, use_container_width=True)
            st.plotly_chart(motion_fig, use_container_width=True)

            def _buffer_pdf(fig_obj) -> bytes:
                buf = io.BytesIO()
                fig_obj.savefig(buf, format="pdf", dpi=300, bbox_inches="tight")
                buf.seek(0)
                return buf.getvalue()

            import io  # local import to keep top-level tidy

            # Matplotlib版本用于PDF下载
            mpl_fig1, ax1 = plt.subplots(figsize=(6, 6))
            ax1.plot(sample_trace["reference_x"], sample_trace["reference_y"], "--", label="Reference", color="#4c78a8")
            ax1.plot(sample_trace["position_x"], sample_trace["position_y"], label="Actual", color="#e45756")
            ax1.set_aspect("equal", "box")
            ax1.set_title("Fig 1 · Reference vs Actual")
            ax1.legend()

            mpl_fig2, axes = plt.subplots(3, 1, figsize=(6.5, 7.5), sharex=True)
            t = sample_trace["timestamp"]
            kcm_cfg = config["kinematic_constraints"]
            axes[0].plot(t, sample_trace["velocity"], color="#1f77b4"); axes[0].axhline(kcm_cfg["MAX_VEL"], color="red", linestyle="--")
            axes[1].plot(t, sample_trace["acceleration"], color="#2ca02c"); axes[1].axhline(kcm_cfg["MAX_ACC"], color="red", linestyle="--")
            axes[2].plot(t, sample_trace["jerk"], color="#d62728"); axes[2].axhline(kcm_cfg["MAX_JERK"], color="red", linestyle="--"); axes[2].axhline(-kcm_cfg["MAX_JERK"], color="red", linestyle="--")
            axes[2].set_xlabel("Time (s)")
            axes[0].set_ylabel("Vel"); axes[1].set_ylabel("Acc"); axes[2].set_ylabel("Jerk")
            axes[0].set_title("Fig 2 · Dynamics")
            mpl_fig1.tight_layout(); mpl_fig2.tight_layout()

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.pyplot(mpl_fig1)
                st.download_button("Download Fig1 (PDF)", _buffer_pdf(mpl_fig1), file_name="fig1_path.pdf")
            with col_dl2:
                st.pyplot(mpl_fig2)
                st.download_button("Download Fig2 (PDF)", _buffer_pdf(mpl_fig2), file_name="fig2_dynamics.pdf")


def main() -> None:
    init_session_state()
    mode = st.sidebar.radio("选择模式", ["Training Ops", "Paper Mode"], index=0)
    render_sidebar_process_manager()
    if mode == "Training Ops":
        render_training_ops()
    else:
        render_paper_mode()


if __name__ == "__main__":
    main()
