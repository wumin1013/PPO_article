import sys
from pathlib import Path
from typing import List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from simulation_engine import (
    SimulationConfig,
    run_simulation,
    build_default_config,
)

# Paper figure utilities
import io
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

st.set_page_config(page_title="Trajectory Planner Dashboard", layout="wide")
st.title("CNC Trajectory Visualization")

DEFAULTS = build_default_config()


def _load_csv(uploaded, label: str) -> pd.DataFrame:
    if uploaded is None:
        return pd.DataFrame()
    df = pd.read_csv(uploaded)
    df["method"] = label
    if "reward_components" in df.columns:
        try:
            df["reward_components"] = df["reward_components"].apply(json.loads)
        except Exception:
            pass
    return df


def _set_academic_style(font: str = "Times New Roman") -> None:
    mpl.rcParams.update({
        "font.family": font,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    sns.set_style("whitegrid")


def _fig8(df_j, df_n, df_t):
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    metrics = ["velocity", "acceleration", "jerk"]
    titles = ["Velocity", "Acceleration", "Jerk"]
    style_map = {
        "J-NNC": {"linestyle": "-", "color": "#1f77b4"},
        "NNC": {"linestyle": "-.", "color": "#2ca02c"},
        "Traditional": {"linestyle": "--", "color": "#d62728"},
    }
    for label, df in [
        ("J-NNC", df_j),
        ("NNC", df_n),
        ("Traditional", df_t),
    ]:
        style = style_map.get(label, {})
        for ax, metric, title in zip(axes, metrics, titles):
            ax.plot(df["timestamp"], df[metric], label=label, **style, linewidth=1.8)
            ax.set_ylabel(title)
            ax.grid(True, linestyle=":", alpha=0.6)
    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title("S-shape Comparison (Fig.8)")
    axes[0].legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    return fig


def _fig9(df: pd.DataFrame):
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
    fig.colorbar(sc, ax=ax, label="Velocity")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Butterfly Path Velocity Heatmap (Fig.9)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig


def _fig11(df: pd.DataFrame):
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()
    ax1.plot(df["timestamp"], df["contour_error"], color="#1f77b4", label="Contour Error")
    ax2.plot(df["timestamp"], df["kcm_intervention"], color="#ff7f0e", linestyle="--", label="KCM Intervention")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Contour Error")
    ax2.set_ylabel("KCM Intervention")
    ax1.set_title("KCM Mechanism Analysis (Fig.11)")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig


def _table_stats(df: pd.DataFrame) -> dict:
    return {
        "Time": float(df["timestamp"].max() if not df.empty else 0.0),
        "Max Error": float(df["contour_error"].abs().max() if not df.empty else 0.0),
        "Mean Error": float(df["contour_error"].abs().mean() if not df.empty else 0.0),
        "Max Jerk": float(df["jerk"].abs().max() if not df.empty else 0.0),
    }


def _buffer_pdf(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def _chart_a(result):
    fig = go.Figure()
    if result.pm:
        pm = np.array(result.pm)
        fig.add_trace(go.Scatter(x=pm[:, 0], y=pm[:, 1], mode="lines+markers", name="Reference", line=dict(dash="dash", color="#444")))
    if result.pl:
        pl = np.array(result.pl)
        fig.add_trace(go.Scatter(x=pl[:, 0], y=pl[:, 1], mode="lines", name="Left Boundary", line=dict(color="green", width=2)))
    if result.pr:
        pr = np.array(result.pr)
        fig.add_trace(go.Scatter(x=pr[:, 0], y=pr[:, 1], mode="lines", name="Right Boundary", line=dict(color="blue", width=2)))
    if result.trajectory:
        traj = np.array(result.trajectory)
        fig.add_trace(go.Scatter(x=traj[:, 0], y=traj[:, 1], mode="lines", name="Actual Trajectory", line=dict(color="red", width=3)))
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    st.subheader("图表 A · 2D轨迹")
    st.plotly_chart(fig, use_container_width=True)


def _chart_b(result):
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Velocity", "Acceleration", "Jerk"))
    t = result.time
    fig.add_trace(go.Scatter(x=t, y=result.velocity, name="Velocity", line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=result.acceleration, name="Acceleration", line=dict(color="#2ca02c")), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=result.jerk, name="Jerk", line=dict(color="#d62728")), row=1, col=3)
    fig.add_trace(go.Scatter(x=t, y=[result.jerk_limit] * len(t), name="Jerk Limit", line=dict(color="#d62728", dash="dash"), showlegend=True), row=1, col=3)
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
    st.subheader("图表 B · 运动学曲线")
    st.plotly_chart(fig, use_container_width=True)


def _chart_c(result):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.progress, y=result.contour_error, mode="lines", line=dict(color="#9467bd")))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="Path Progress", yaxis_title="Contour Error")
    st.subheader("图表 C · 轮廓误差")
    st.plotly_chart(fig, use_container_width=True)


def _chart_d(result):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.time, y=result.kcm_intervention, mode="lines", line=dict(color="#ff7f0e")))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="Time (s)", yaxis_title="KCM Intervention")
    st.subheader("图表 D · KCM干预度")
    st.plotly_chart(fig, use_container_width=True)


def _render_metrics(result):
    m = result.metrics
    cols = st.columns(4)
    cols[0].metric("加工时间 (s)", f"{m['total_time']:.2f}")
    cols[1].metric("最大轮廓误差", f"{m['max_error']:.4f}")
    cols[2].metric("平均轮廓误差", f"{m['mean_error']:.4f}")
    cols[3].metric("最大捷度", f"{m['max_jerk']:.3f}")


with st.sidebar:
    st.header("配置")
    model_choice = st.selectbox("模型选择", ["J-NNC (Ours)", "NNC (Baseline)", "Traditional"], index=0)
    path_choice = st.selectbox("路径选择", ["S-Shape", "Butterfly"], index=0)
    disable_jerk = st.checkbox("禁用 Jerk 奖励", value=False)
    disable_kcm = st.checkbox("禁用 KCM", value=False)
    max_velocity = st.slider("Max Velocity", 0.2, 5.0, DEFAULTS.max_velocity, 0.1)
    max_jerk = st.slider("Max Jerk", 0.5, 20.0, DEFAULTS.max_jerk, 0.5)
    epsilon = st.slider("Epsilon (轮廓容限)", 0.1, 1.5, DEFAULTS.epsilon, 0.05)
    run_button = st.button("开始仿真", type="primary")

info_box = st.empty()

if run_button:
    with st.spinner("正在运行仿真..."):
        sim_cfg = SimulationConfig(
            model_name=model_choice,
            path_name=path_choice,
            disable_jerk_reward=disable_jerk,
            disable_kcm=disable_kcm,
            max_velocity=max_velocity,
            max_jerk=max_jerk,
            epsilon=epsilon,
        )
        result = run_simulation(sim_cfg)

    info_msgs: List[str] = []
    if disable_kcm:
        info_msgs.append("KCM已禁用：约束不再限制动作，捷度可能过大")
    if disable_jerk:
        info_msgs.append("Jerk奖励已关闭：速度会更激进以展示消融")
    if info_msgs:
        info_box.warning(" | ".join(info_msgs))
    else:
        info_box.empty()

    _render_metrics(result)
    _chart_a(result)
    _chart_b(result)
    _chart_c(result)
    _chart_d(result)
else:
    st.info("在左侧选择模型与参数后点击“开始仿真”")


# ------------------ 论文图表生成区 ------------------
st.markdown("---")
st.header("论文图表一键生成")
with st.expander("上传CSV并生成 Fig8 / Fig9 / Fig11 / Table", expanded=False):
    st.markdown(
        """
        **CSV 准备方法**
        - 运行推理：`python PPO_project/main.py --mode test --config PPO_project/configs/default.yaml --model 路径/到/模型.pth`
        - 生成的文件：默认写入 `PPO_project/logs/experiment_results.csv`
        - 若需三条 S 形对比曲线，请分别用 J-NNC / NNC / Traditional 三次推理，各自复制或重命名为 jncc_s.csv / nnc_s.csv / traditional_s.csv
        - 蝶形热力图可用蝶形路径跑一次，另存 `jncc_butterfly.csv`（或复用 jncc_s.csv）
        - Fig11 可复用 J-NNC 的 S 形 CSV，如需单独实验可另存
        """
    )
    font_choice = st.text_input("字体(默认 Times New Roman)", value="Times New Roman")
    col_u1, col_u2, col_u3 = st.columns(3)
    with col_u1:
        jncc_file = st.file_uploader("J-NNC · S形 CSV", type=["csv"], key="jncc")
        fig11_file = st.file_uploader("Fig11 CSV (可复用J-NNC)", type=["csv"], key="fig11")
    with col_u2:
        nnc_file = st.file_uploader("NNC · S形 CSV", type=["csv"], key="nnc")
        heatmap_file = st.file_uploader("热力图 CSV (蝶形)", type=["csv"], key="heatmap")
    with col_u3:
        trad_file = st.file_uploader("Traditional · S形 CSV", type=["csv"], key="trad")
    run_fig8 = st.button("生成 Fig8 (S形对比)")
    run_fig9 = st.button("生成 Fig9 (速度热力图)")
    run_fig11 = st.button("生成 Fig11 (误差 vs KCM)")
    run_table = st.button("生成 Table 2 统计")

    _set_academic_style(font_choice)

    if run_fig8:
        if not (jncc_file and nnc_file and trad_file):
            st.warning("需要同时提供 J-NNC / NNC / Traditional 的 S形 CSV")
        else:
            df_j = _load_csv(jncc_file, "J-NNC")
            df_n = _load_csv(nnc_file, "NNC")
            df_t = _load_csv(trad_file, "Traditional")
            fig = _fig8(df_j, df_n, df_t)
            st.pyplot(fig)
            st.download_button(
                "下载 Fig8 (PDF)", _buffer_pdf(fig), file_name="fig8_s_shape.pdf", mime="application/pdf"
            )

    if run_fig9:
        src = heatmap_file or jncc_file
        if not src:
            st.warning("请提供用于热力图的 CSV（通常蝶形路径），未提供则默认用 J-NNC CSV")
        else:
            df_h = _load_csv(src, "Heatmap")
            fig = _fig9(df_h)
            st.pyplot(fig)
            st.download_button(
                "下载 Fig9 (PDF)", _buffer_pdf(fig), file_name="fig9_heatmap.pdf", mime="application/pdf"
            )

    if run_fig11:
        src = fig11_file or jncc_file
        if not src:
            st.warning("请提供 Fig11 CSV，未提供则默认用 J-NNC CSV")
        else:
            df_f = _load_csv(src, "J-NNC")
            fig = _fig11(df_f)
            st.pyplot(fig)
            st.download_button(
                "下载 Fig11 (PDF)", _buffer_pdf(fig), file_name="fig11_kcm.pdf", mime="application/pdf"
            )

    if run_table:
        rows = []
        if jncc_file:
            rows.append(("S-shape", "J-NNC", _table_stats(_load_csv(jncc_file, "J-NNC"))))
        if nnc_file:
            rows.append(("S-shape", "NNC", _table_stats(_load_csv(nnc_file, "NNC"))))
        if trad_file:
            rows.append(("S-shape", "Traditional", _table_stats(_load_csv(trad_file, "Traditional"))))
        if heatmap_file:
            rows.append(("Butterfly", "J-NNC", _table_stats(_load_csv(heatmap_file, "Heatmap"))))
        if not rows:
            st.warning("请至少提供一份 CSV 以计算统计")
        else:
            table_data = []
            for path_name, label, stats in rows:
                table_data.append(
                    {
                        "Path": path_name,
                        "Method": label,
                        "Time": stats["Time"],
                        "Max Error": stats["Max Error"],
                        "Mean Error": stats["Mean Error"],
                        "Max Jerk": stats["Max Jerk"],
                    }
                )
            st.dataframe(pd.DataFrame(table_data))
