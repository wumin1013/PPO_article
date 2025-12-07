"""
主训练入口 - 基于模块化重构的PPO轨迹跟踪训练脚本
"""

from __future__ import annotations

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm

# 添加父目录到路径，便于导入原始模块
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 导入重构后的模块（使用相对导入）
from src.algorithms.baselines import NNCAgent, create_baseline_agent
from src.algorithms.ppo import PPOContinuous
from src.utils.checkpoint import CheckpointManager, load_for_resume
from src.utils.logger import CSVLogger, DataLogger, ExperimentManager
from src.utils.path_generator import get_path_by_name

# 导入原始模块（必须从父目录）
from PPO最终版 import Env, PaperMetrics, configure_chinese_font, visualize_final_path
import rl_utils  # noqa: F401


def resolve_config_path(config_path: str) -> str:
    """将配置路径转为绝对路径，保证任何调用地点一致。"""
    if os.path.isabs(config_path):
        return config_path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, config_path)


def load_config(config_path: str) -> Tuple[dict, str]:
    """加载YAML配置文件，返回配置及绝对路径。"""
    resolved_path = resolve_config_path(config_path)
    with open(resolved_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config, resolved_path


def _init_experiment(
    config_path: str,
    path_config: dict,
    experiment_mode: str,
    experiment_config: dict,
    experiment_dir: Path | str | None = None,
) -> tuple[ExperimentManager, str]:
    """创建实验管理器及日志标签。"""
    experiment_category = experiment_config.get("category") or experiment_mode
    manager = ExperimentManager(str(experiment_category), config_path, experiment_dir=experiment_dir)
    path_label = path_config.get("type", "path")
    log_tag = f"{experiment_mode}_{path_label}"
    return manager, log_tag


def _init_loggers(manager: ExperimentManager, log_tag: str) -> tuple[CSVLogger, CSVLogger, CSVLogger, CSVLogger]:
    """构建所需的原子化CSV日志记录器。"""
    step_logger = manager.create_logger(
        f"step_metrics_{log_tag}.csv",
        ["episode_idx", "env_step", "reward", "contour_error", "jerk", "kcm_intervention"],
    )
    episode_logger = manager.create_logger(
        f"episode_metrics_{log_tag}.csv",
        ["episode_idx", "total_reward", "actor_loss", "critic_loss", "epsilon"],
    )
    paper_logger = manager.create_logger(
        f"paper_metrics_{log_tag}.csv",
        [
            "episode_idx",
            "rmse_error",
            "mean_jerk",
            "roughness_proxy",
            "mean_velocity",
            "max_error",
            "mean_kcm_intervention",
            "steps",
            "progress",
        ],
    )
    training_logger = manager.create_logger(
        f"training_{log_tag}.csv", ["episode_idx", "reward", "actor_loss", "critic_loss", "wall_time"]
    )
    return step_logger, episode_logger, paper_logger, training_logger


def _build_path(path_config: dict) -> list[np.ndarray]:
    """根据配置生成参考路径。"""
    if path_config["type"] == "waypoints":
        return [np.array(wp) for wp in path_config["waypoints"]]

    path_type = path_config["type"]
    scale = path_config.get("scale", 10.0)
    num_points = path_config.get("num_points", 200)
    kwargs = {}
    if path_type in path_config:
        kwargs.update(path_config[path_type])
    Pm = get_path_by_name(path_type, scale=scale, num_points=num_points, **kwargs)
    print(f"使用参数化路径 {path_type}, 采样点数: {len(Pm)}")
    return Pm


def _find_model_checkpoint(category: str, mode_suffix: str) -> Path | None:
    """查找最新实验下的优选模型（best_model优先，其次旧版final）。"""
    base_dir = Path(__file__).resolve().parent / "saved_models" / category
    if not base_dir.exists():
        return None
    experiment_dirs = sorted([p for p in base_dir.iterdir() if p.is_dir()], reverse=True)
    for exp_dir in experiment_dirs:
        checkpoints_dir = exp_dir / "checkpoints"
        best_candidate = checkpoints_dir / "best_model.pth"
        if best_candidate.exists():
            return best_candidate
        legacy = checkpoints_dir / f"tracking_model{mode_suffix}_final.pth"
        if legacy.exists():
            return legacy
    return None


def train(config_path: str = "configs/default.yaml", resume_path: str | None = None) -> None:
    """训练入口，支持断点续训。"""
    configure_chinese_font()

    config, resolved_config_path = load_config(config_path)
    print(f"加载配置: {resolved_config_path}")
    print(yaml.dump(config, allow_unicode=True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    env_config = config["environment"]
    kcm_config = config["kinematic_constraints"]
    path_config = config["path"]
    experiment_config = config.get("experiment", {"mode": "train", "enable_kcm": True})
    experiment_mode = experiment_config.get("mode", "train")

    Pm = _build_path(path_config)
    env = Env(
        device=device,
        epsilon=env_config["epsilon"],
        interpolation_period=env_config["interpolation_period"],
        MAX_VEL=kcm_config["MAX_VEL"],
        MAX_ACC=kcm_config["MAX_ACC"],
        MAX_JERK=kcm_config["MAX_JERK"],
        MAX_ANG_VEL=kcm_config["MAX_ANG_VEL"],
        MAX_ANG_ACC=kcm_config["MAX_ANG_ACC"],
        MAX_ANG_JERK=kcm_config["MAX_ANG_JERK"],
        Pm=Pm,
        max_steps=env_config["max_steps"],
        lookahead_points=env_config.get("lookahead_points", 5),
    )

    obs_space = getattr(env, "observation_space", None)
    act_space = getattr(env, "action_space", None)
    print(f"环境创建成功: 状态维度{env.observation_dim}, 动作维度={env.action_space_dim}")

    resume_state = None
    resume_experiment_dir: Path | None = None

    if experiment_mode in ["baseline_nnc", "baseline_s_curve"]:
        baseline_type = experiment_mode.split("_")[1]
        config["state_dim"] = obs_space.shape[0] if obs_space is not None else env.observation_dim
        config["action_dim"] = act_space.shape[0] if act_space is not None else env.action_space_dim
        config["observation_space"] = obs_space
        config["action_space"] = act_space
        agent = create_baseline_agent(baseline_type, config, device)
        print(f"创建基线算法智能体 {baseline_type.upper()}")

    elif experiment_mode == "ablation_no_kcm":
        ppo_config = config["ppo"]
        agent = NNCAgent(
            state_dim=None,
            hidden_dim=ppo_config["hidden_dim"],
            action_dim=None,
            actor_lr=ppo_config["actor_lr"],
            critic_lr=ppo_config["critic_lr"],
            lmbda=ppo_config["lmbda"],
            epochs=ppo_config["epochs"],
            eps=ppo_config["eps"],
            gamma=ppo_config["gamma"],
            device=device,
            max_vel=kcm_config["MAX_VEL"],
            max_ang_vel=kcm_config["MAX_ANG_VEL"],
            observation_space=obs_space,
            action_space=act_space,
        )
        print("消融实验模式: 禁用KCM模块")

    else:
        ppo_config = config["ppo"]
        agent = PPOContinuous(
            state_dim=None,
            hidden_dim=ppo_config["hidden_dim"],
            action_dim=None,
            actor_lr=ppo_config["actor_lr"],
            critic_lr=ppo_config["critic_lr"],
            lmbda=ppo_config["lmbda"],
            epochs=ppo_config["epochs"],
            eps=ppo_config["eps"],
            gamma=ppo_config["gamma"],
            device=device,
            observation_space=obs_space,
            action_space=act_space,
        )

        if experiment_mode == "ablation_no_reward":
            config["reward_weights"]["w_j"] = 0.0
            config["reward_weights"]["w_c"] = 0.0
            print("消融实验模式: 禁用惩罚权重")
        else:
            print(f"PPO智能体创建成功 模式: {experiment_mode}")

    training_config = config["training"]
    num_episodes = training_config["num_episodes"]
    smoothing_factor = training_config["smoothing_factor"]
    save_interval = training_config["save_interval"]
    log_interval = training_config["log_interval"]
    checkpoint_interval_steps = training_config.get("checkpoint_interval_steps", 2048)

    smoothed_rewards: list[float] = []
    wall_time_start = time.perf_counter()
    start_episode = 0
    global_step = 0
    last_checkpoint_step = 0
    best_eval_reward = float("-inf")

    if resume_path:
        resume_file = Path(resume_path)
        if resume_file.exists():
            resume_state = load_for_resume(resume_file, agent, device)
            start_episode = resume_state.episode + 1
            global_step = resume_state.global_step
            last_checkpoint_step = resume_state.global_step
            best_eval_reward = resume_state.best_eval_reward
            resume_experiment_dir = resume_state.experiment_dir
            if resume_state.last_smoothed_reward is not None:
                smoothed_rewards.append(resume_state.last_smoothed_reward)
            if resume_state.obs_normalizer_stats:
                setattr(env, "normalization_params", resume_state.obs_normalizer_stats)
            print(
                f"断点续训: 从 {resume_file} 恢复，已完成 episode={resume_state.episode}, "
                f"global_step={resume_state.global_step}, best_eval_reward={best_eval_reward:.2f}"
            )
        else:
            print(f"警告: 未找到续训文件 {resume_file}，将从头开始。")

    manager, log_tag = _init_experiment(
        resolved_config_path, path_config, experiment_mode, experiment_config, experiment_dir=resume_experiment_dir
    )
    checkpoint_manager = CheckpointManager(manager.models_dir)
    step_logger, episode_logger, paper_logger, training_logger = _init_loggers(manager, log_tag)

    if start_episode >= num_episodes:
        print(f"续训起始回合({start_episode})已达到总回合数({num_episodes})，不再继续。")
        return

    print(f"\n开始训练 共{num_episodes}个回合\n")

    with tqdm(total=num_episodes, initial=start_episode, desc="训练进度") as pbar:
        for episode in range(start_episode, num_episodes):
            state = env.reset()
            paper_metrics = PaperMetrics()

            transition_dict = {
                "states": [],
                "actions": [],
                "next_states": [],
                "rewards": [],
                "dones": [],
            }

            episode_reward = 0.0
            done = False
            info: dict = {}

            while not done:
                action = agent.take_action(state)
                next_state, reward, done, info = env.step(action)
                global_step += 1

                transition_dict["states"].append(state)
                transition_dict["actions"].append(action)
                transition_dict["next_states"].append(next_state)
                transition_dict["rewards"].append(reward)
                transition_dict["dones"].append(done)

                paper_metrics.update(
                    contour_error=info["contour_error"],
                    jerk=info["jerk"],
                    velocity=env.velocity,
                    kcm_intervention=info["kcm_intervention"],
                )

                step_logger.log_step(
                    episode_idx=episode,
                    env_step=info["step"],
                    reward=reward,
                    contour_error=info["contour_error"],
                    jerk=info["jerk"],
                    kcm_intervention=info["kcm_intervention"],
                )

                if hasattr(agent, "actor") and global_step - last_checkpoint_step >= checkpoint_interval_steps:
                    checkpoint_manager.save_latest(
                        agent=agent,
                        episode_idx=episode,
                        global_step=global_step,
                        best_eval_reward=best_eval_reward,
                        config=config,
                        last_smoothed_reward=smoothed_rewards[-1] if smoothed_rewards else None,
                        obs_normalizer_stats=getattr(env, "normalization_params", None),
                        experiment_dir=manager.experiment_dir,
                    )
                    last_checkpoint_step = global_step

                episode_reward += reward
                state = next_state

            final_progress = info.get("progress", 0.0)

            if hasattr(agent, "update") and len(transition_dict["states"]) > 10:
                avg_actor_loss, avg_critic_loss = agent.update(transition_dict)
            else:
                avg_actor_loss, avg_critic_loss = 0.0, 0.0

            if not smoothed_rewards:
                smoothed_rewards.append(episode_reward)
            else:
                new_smoothed = smoothing_factor * episode_reward + (1 - smoothing_factor) * smoothed_rewards[-1]
                smoothed_rewards.append(new_smoothed)

            metrics = paper_metrics.compute()

            episode_logger.log_step(
                episode_idx=episode,
                total_reward=episode_reward,
                actor_loss=avg_actor_loss,
                critic_loss=avg_critic_loss,
                epsilon=env_config["epsilon"],
            )

            paper_logger.log_step(
                episode_idx=episode,
                rmse_error=metrics["rmse_error"],
                mean_jerk=metrics["mean_jerk"],
                roughness_proxy=metrics["roughness_proxy"],
                mean_velocity=metrics["mean_velocity"],
                max_error=metrics["max_error"],
                mean_kcm_intervention=metrics["mean_kcm_intervention"],
                steps=metrics["steps"],
                progress=final_progress,
            )

            training_logger.log_step(
                episode_idx=episode,
                reward=episode_reward,
                actor_loss=avg_actor_loss,
                critic_loss=avg_critic_loss,
                wall_time=time.perf_counter() - wall_time_start,
            )

            eval_reward = smoothed_rewards[-1] if smoothed_rewards else episode_reward
            if hasattr(agent, "actor") and eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                best_path = checkpoint_manager.save_best(
                    agent=agent,
                    eval_reward=eval_reward,
                    episode_idx=episode,
                    global_step=global_step,
                    config=config,
                )
                print(f"发现更优模型: eval_reward={eval_reward:.2f}, 保存到 {best_path}")

            if (episode + 1) % log_interval == 0:
                print(f"\n{'=' * 80}")
                print(f"Episode {episode + 1} - 论文指标摘要:")
                print(f"{'=' * 80}")
                print(f"  RMSE Error:              {metrics['rmse_error']:.6f}")
                print(f"  Mean Jerk:               {metrics['mean_jerk']:.6f}")
                print(f"  Roughness Proxy:         {metrics['roughness_proxy']:.6f}")
                print(f"  Mean Velocity:           {metrics['mean_velocity']:.4f}")
                print(f"  Max Error:               {metrics['max_error']:.6f}")
                print(f"  Mean KCM Intervention:   {metrics['mean_kcm_intervention']:.6f}")
                print(f"  Steps:                   {metrics['steps']}")
                print(f"  Progress:                {final_progress:.4f}")
                print(f"  Total Reward:            {episode_reward:.2f}")
                print(f"{'=' * 80}\n")

            if (episode + 1) % save_interval == 0 and hasattr(agent, "actor"):
                mode_suffix = f"_{experiment_mode}" if experiment_mode != "train" else ""
                model_path = manager.models_dir / f"tracking_model{mode_suffix}_ep{episode+1}.pth"
                torch.save(
                    {
                        "actor": agent.actor.state_dict(),
                        "critic": agent.critic.state_dict(),
                        "config": config,
                    },
                    model_path,
                )

            pbar.set_postfix(
                {
                    "Reward": f"{episode_reward:.1f}",
                    "Smoothed": f"{smoothed_rewards[-1]:.1f}",
                    "Actor Loss": f"{avg_actor_loss:.2f}",
                    "Critic Loss": f"{avg_critic_loss:.2f}",
                }
            )
            pbar.update(1)

    print("\n" + "=" * 80)
    print(f"训练完成！实验目录 {manager.experiment_dir}")
    print(f"日志目录: {manager.logs_dir}")
    print("=" * 80 + "\n")

    if hasattr(agent, "actor"):
        mode_suffix = f"_{experiment_mode}" if experiment_mode != "train" else ""
        final_model_path = manager.models_dir / f"tracking_model{mode_suffix}_final.pth"
        torch.save(
            {
                "actor": agent.actor.state_dict(),
                "critic": agent.critic.state_dict(),
                "config": config,
            },
            final_model_path,
        )
        checkpoint_manager.save_latest(
            agent=agent,
            episode_idx=num_episodes - 1,
            global_step=global_step,
            best_eval_reward=best_eval_reward,
            config=config,
            last_smoothed_reward=smoothed_rewards[-1] if smoothed_rewards else None,
            obs_normalizer_stats=getattr(env, "normalization_params", None),
            experiment_dir=manager.experiment_dir,
        )
        print(f"\n最终模型已保存: {final_model_path}")

    print(f"\n可视化最终轨迹(ε={env_config['epsilon']:.3f})")
    visualize_final_path(env)


def test(config_path: str = "configs/default.yaml", model_path: str | None = None) -> None:
    """测试入口 - 加载已训练模型进行推理。"""
    configure_chinese_font()

    config, resolved_config_path = load_config(config_path)
    print(f"加载配置: {resolved_config_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    env_config = config["environment"]
    kcm_config = config["kinematic_constraints"]
    path_config = config["path"]
    experiment_config = config.get("experiment", {})
    experiment_mode = experiment_config.get("mode", "test")

    manager, log_tag = _init_experiment(resolved_config_path, path_config, experiment_mode, experiment_config)

    Pm = _build_path(path_config)
    env = Env(
        device=device,
        epsilon=env_config["epsilon"],
        interpolation_period=env_config["interpolation_period"],
        MAX_VEL=kcm_config["MAX_VEL"],
        MAX_ACC=kcm_config["MAX_ACC"],
        MAX_JERK=kcm_config["MAX_JERK"],
        MAX_ANG_VEL=kcm_config["MAX_ANG_VEL"],
        MAX_ANG_ACC=kcm_config["MAX_ANG_ACC"],
        MAX_ANG_JERK=kcm_config["MAX_ANG_JERK"],
        Pm=Pm,
        max_steps=env_config["max_steps"],
        lookahead_points=env_config.get("lookahead_points", 5),
    )

    obs_space = getattr(env, "observation_space", None)
    act_space = getattr(env, "action_space", None)
    ppo_config = config["ppo"]
    agent = PPOContinuous(
        state_dim=None,
        hidden_dim=ppo_config["hidden_dim"],
        action_dim=None,
        actor_lr=ppo_config["actor_lr"],
        critic_lr=ppo_config["critic_lr"],
        lmbda=ppo_config["lmbda"],
        epochs=ppo_config["epochs"],
        eps=ppo_config["eps"],
        gamma=ppo_config["gamma"],
        device=device,
        observation_space=obs_space,
        action_space=act_space,
    )

    mode_suffix = f"_{experiment_mode}" if experiment_mode != "train" else ""
    if model_path is None:
        category = experiment_config.get("category") or experiment_mode
        latest = _find_model_checkpoint(str(category), mode_suffix)
        if latest is None:
            script_dir = Path(__file__).resolve().parent
            legacy = script_dir / f"tracking_model{mode_suffix}_final.pth"
            model_path = legacy
        else:
            model_path = latest
    model_path = Path(model_path)

    if not model_path.exists():
        print(f"错误: 模型文件不存在 {model_path}")
        return

    checkpoint = torch.load(model_path, map_location=device)
    agent.actor.load_state_dict(checkpoint["actor"])
    agent.critic.load_state_dict(checkpoint["critic"])
    print(f"成功加载模型: {model_path}")

    agent.actor.eval()
    agent.critic.eval()

    state = env.reset()
    paper_metrics = PaperMetrics()
    done = False
    total_reward = 0.0
    dt = env.interpolation_period

    print("\n开始测试.")
    path_label = path_config.get("type", "path")
    log_filename = f"experiment_results_{log_tag}.csv"

    with DataLogger(log_dir=manager.logs_dir, filename=log_filename) as data_logger, torch.no_grad():
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, info = env.step(action)

            ref_point = DataLogger.project_to_path(
                position=env.current_position,
                path_points=env.Pm,
                segment_index=info.get("segment_idx", getattr(env, "current_segment_idx", 0)),
            )

            data_logger.log_step(
                dt=dt,
                position=env.current_position,
                reference_point=ref_point,
                velocity=env.velocity,
                acceleration=env.acceleration,
                jerk=env.jerk,
                contour_error=info.get("contour_error", 0.0),
                kcm_intervention=info.get("kcm_intervention", 0.0),
                reward_components=getattr(env, "last_reward_components", {}),
            )

            paper_metrics.update(
                contour_error=info["contour_error"],
                jerk=info["jerk"],
                velocity=env.velocity,
                kcm_intervention=info["kcm_intervention"],
            )

            total_reward += reward
            state = next_state

    metrics = paper_metrics.compute()
    print("\n" + "=" * 80)
    print("测试结果 - 论文指标:")
    print("=" * 80)
    print(f"  RMSE Error:              {metrics['rmse_error']:.6f}")
    print(f"  Mean Jerk:               {metrics['mean_jerk']:.6f}")
    print(f"  Roughness Proxy:         {metrics['roughness_proxy']:.6f}")
    print(f"  Mean Velocity:           {metrics['mean_velocity']:.4f}")
    print(f"  Max Error:               {metrics['max_error']:.6f}")
    print(f"  Mean KCM Intervention:   {metrics['mean_kcm_intervention']:.6f}")
    print(f"  Steps:                   {metrics['steps']}")
    print(f"  Progress:                {info['progress']:.4f}")
    print(f"  Total Reward:            {total_reward:.2f}")
    print("=" * 80 + "\n")

    visualize_final_path(env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练/测试基于PPO的轨迹跟踪智能体")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test", "ablation_no_kcm", "ablation_no_reward", "baseline_nnc", "baseline_s_curve"],
        help="运行模式",
    )
    parser.add_argument("--model", type=str, default=None, help="测试模式下的模型路径")
    parser.add_argument("--resume", type=str, default=None, help="latest_checkpoint.pth 路径，用于断点续训")

    args = parser.parse_args()

    if args.mode != "train":
        config, _ = load_config(args.config)
        config.setdefault("experiment", {})
        config["experiment"]["mode"] = args.mode

        script_dir = os.path.dirname(os.path.abspath(__file__))
        temp_config_path = os.path.join(script_dir, "temp_config.yaml")
        with open(temp_config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True)

        if args.mode == "test":
            test(temp_config_path, args.model)
        else:
            train(temp_config_path, args.resume)

        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
    else:
        train(args.config, args.resume)
