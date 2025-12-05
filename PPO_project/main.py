"""
主训练入口
基于模块化重构的PPO轨迹跟踪训练脚本
"""

import os
import sys

# 添加父目录到路径，以便导入原始模块
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import yaml
import torch
import numpy as np
from tqdm import tqdm
import csv
import argparse
from pathlib import Path

# 导入重构后的模块（使用相对导入）
from src.algorithms.ppo import PPOContinuous
from src.algorithms.kcm import KinematicConstraintModule
from src.algorithms.baselines import NNCAgent, SCurvePlanner, create_baseline_agent
from src.utils.path_generator import get_path_by_name
from src.utils.logger import DataLogger

# 导入原始模块（必须从父目录）
from PPO最终版 import Env, PaperMetrics, visualize_final_path, configure_chinese_font
import rl_utils


def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    # 如果是相对路径，转换为相对于脚本所在目录的绝对路径
    if not os.path.isabs(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_path)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(log_dir: str = 'logs', tag: str = ''):
    """设置日志目录和文件"""
    # 确保日志目录在脚本所在目录下
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, log_dir)
    os.makedirs(log_dir, exist_ok=True)

    suffix = f"_{tag}" if tag else ""
    
    # 创建CSV日志文件
    step_log = open(f'{log_dir}/step_log{suffix}.csv', 'w', newline='', encoding='utf-8')
    episode_log = open(f'{log_dir}/episode_log{suffix}.csv', 'w', newline='', encoding='utf-8')
    paper_metrics_log = open(f'{log_dir}/paper_metrics{suffix}.csv', 'w', newline='', encoding='utf-8')
    
    # 写入表头
    step_writer = csv.writer(step_log)
    step_writer.writerow(['episode', 'step', 'reward', 'contour_error', 'jerk', 'kcm_intervention'])
    
    episode_writer = csv.writer(episode_log)
    episode_writer.writerow(['episode', 'total_reward', 'actor_loss', 'critic_loss', 'epsilon'])
    
    paper_metrics_writer = csv.writer(paper_metrics_log)
    paper_metrics_writer.writerow([
        'episode', 'rmse_error', 'mean_jerk', 'roughness_proxy', 
        'mean_velocity', 'max_error', 'mean_kcm_intervention', 'steps', 'progress'
    ])
    
    return step_log, step_writer, episode_log, episode_writer, paper_metrics_log, paper_metrics_writer


def train(config_path: str = 'configs/default.yaml'):
    """
    主训练函数
    
    Args:
        config_path: 配置文件路径
    """
    # 配置中文字体
    configure_chinese_font()
    
    # 加载配置
    config = load_config(config_path)
    print(f"加载配置: {config_path}")
    print(yaml.dump(config, allow_unicode=True))
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建环境
    env_config = config['environment']
    kcm_config = config['kinematic_constraints']
    path_config = config['path']
    experiment_config = config.get('experiment', {'mode': 'train', 'enable_kcm': True})
    
    # 生成参考路径
    if path_config['type'] == 'waypoints':
        # 使用预定义的路径点
        Pm = [np.array(wp) for wp in path_config['waypoints']]
    else:
        # 使用参数化路径生成器
        path_type = path_config['type']
        scale = path_config.get('scale', 10.0)
        num_points = path_config.get('num_points', 200)
        
        # 根据路径类型传递额外参数
        kwargs = {}
        if path_type in path_config:
            kwargs.update(path_config[path_type])
        
        Pm = get_path_by_name(path_type, scale=scale, num_points=num_points, **kwargs)
        print(f"使用参数化路径: {path_type}, 采样点数: {len(Pm)}")
    
    env = Env(
        device=device,
        epsilon=env_config['epsilon'],
        interpolation_period=env_config['interpolation_period'],
        MAX_VEL=kcm_config['MAX_VEL'],
        MAX_ACC=kcm_config['MAX_ACC'],
        MAX_JERK=kcm_config['MAX_JERK'],
        MAX_ANG_VEL=kcm_config['MAX_ANG_VEL'],
        MAX_ANG_ACC=kcm_config['MAX_ANG_ACC'],
        MAX_ANG_JERK=kcm_config['MAX_ANG_JERK'],
        Pm=Pm,
        max_steps=env_config['max_steps']
    )
    
    print(f"环境创建成功: 状态维度={env.observation_dim}, 动作维度={env.action_space_dim}")
    
    # 根据实验模式创建智能体
    experiment_mode = experiment_config.get('mode', 'train')
    
    if experiment_mode in ['baseline_nnc', 'baseline_s_curve']:
        # 创建基线算法智能体
        baseline_type = experiment_mode.split('_')[1]  # 提取 'nnc' 或 's_curve'
        config['state_dim'] = env.observation_dim
        config['action_dim'] = env.action_space_dim
        agent = create_baseline_agent(baseline_type, config, device)
        print(f"创建基线算法智能体: {baseline_type.upper()}")
        
    elif experiment_mode == 'ablation_no_kcm':
        # 消融实验：禁用KCM
        ppo_config = config['ppo']
        agent = NNCAgent(
            state_dim=env.observation_dim,
            hidden_dim=ppo_config['hidden_dim'],
            action_dim=env.action_space_dim,
            actor_lr=ppo_config['actor_lr'],
            critic_lr=ppo_config['critic_lr'],
            lmbda=ppo_config['lmbda'],
            epochs=ppo_config['epochs'],
            eps=ppo_config['eps'],
            gamma=ppo_config['gamma'],
            device=device,
            max_vel=kcm_config['MAX_VEL'],
            max_ang_vel=kcm_config['MAX_ANG_VEL']
        )
        print(f"消融实验模式: 禁用KCM模块")
        
    else:
        # 正常训练或消融实验（调整奖励权重）
        ppo_config = config['ppo']
        agent = PPOContinuous(
            state_dim=env.observation_dim,
            hidden_dim=ppo_config['hidden_dim'],
            action_dim=env.action_space_dim,
            actor_lr=ppo_config['actor_lr'],
            critic_lr=ppo_config['critic_lr'],
            lmbda=ppo_config['lmbda'],
            epochs=ppo_config['epochs'],
            eps=ppo_config['eps'],
            gamma=ppo_config['gamma'],
            device=device
        )
        
        if experiment_mode == 'ablation_no_reward':
            # 消融实验：禁用捷度奖励
            config['reward_weights']['w_j'] = 0.0
            config['reward_weights']['w_c'] = 0.0
            print(f"消融实验模式: 禁用捷度奖励权重")
        else:
            print(f"PPO智能体创建成功 - 模式: {experiment_mode}")
    
    # 设置日志（保存到PPO_project目录），按实验/路径区分文件名
    path_label = path_config.get('type', 'path')
    experiment_label = experiment_mode
    log_tag = f"{experiment_label}_{path_label}"
    step_log, step_writer, episode_log, episode_writer, paper_metrics_log, paper_metrics_writer = setup_logging('logs', tag=log_tag)
    
    # 训练配置
    training_config = config['training']
    num_episodes = training_config['num_episodes']
    smoothing_factor = training_config['smoothing_factor']
    save_interval = training_config['save_interval']
    log_interval = training_config['log_interval']
    
    # 训练循环
    reward_history = []
    smoothed_rewards = []
    loss_history = []
    
    print(f"\n开始训练 - 共{num_episodes}个回合\n")
    
    with tqdm(total=num_episodes, desc="训练进度") as pbar:
        for episode in range(num_episodes):
            # 重置环境和指标
            state = env.reset()
            paper_metrics = PaperMetrics()
            
            transition_dict = {
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': []
            }
            
            episode_reward = 0
            done = False
            
            # Episode循环
            while not done:
                # 选择动作
                action = agent.take_action(state)
                
                # 执行动作
                next_state, reward, done, info = env.step(action)
                
                # 记录转移
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                
                # 更新论文指标
                paper_metrics.update(
                    contour_error=info['contour_error'],
                    jerk=info['jerk'],
                    velocity=env.velocity,
                    kcm_intervention=info['kcm_intervention']
                )
                
                # 记录步日志
                step_writer.writerow([
                    episode,
                    info['step'],
                    reward,
                    info['contour_error'],
                    info['jerk'],
                    info['kcm_intervention']
                ])
                
                episode_reward += reward
                state = next_state
            
            final_progress = info['progress']
            
            # 更新策略（仅对可训练的智能体）
            if hasattr(agent, 'update') and len(transition_dict['states']) > 10:
                avg_actor_loss, avg_critic_loss = agent.update(transition_dict)
                loss_history.append((avg_actor_loss, avg_critic_loss))
            else:
                avg_actor_loss, avg_critic_loss = 0.0, 0.0
            
            reward_history.append(episode_reward)
            
            # 平滑奖励
            if not smoothed_rewards:
                smoothed_rewards.append(episode_reward)
            else:
                new_smoothed = smoothing_factor * episode_reward + (1 - smoothing_factor) * smoothed_rewards[-1]
                smoothed_rewards.append(new_smoothed)
            
            # 计算论文指标
            metrics = paper_metrics.compute()
            
            # 记录回合日志
            episode_writer.writerow([
                episode,
                episode_reward,
                avg_actor_loss,
                avg_critic_loss,
                env_config["epsilon"]
            ])
            
            # 记录论文指标
            paper_metrics_writer.writerow([
                episode,
                metrics['rmse_error'],
                metrics['mean_jerk'],
                metrics['roughness_proxy'],
                metrics['mean_velocity'],
                metrics['max_error'],
                metrics['mean_kcm_intervention'],
                metrics['steps'],
                final_progress
            ])
            
            # 定期打印详细指标
            if (episode + 1) % log_interval == 0:
                print(f"\n{'='*80}")
                print(f"Episode {episode + 1} - 论文指标摘要:")
                print(f"{'='*80}")
                print(f"  RMSE Error:              {metrics['rmse_error']:.6f}")
                print(f"  Mean Jerk:               {metrics['mean_jerk']:.6f}")
                print(f"  Roughness Proxy:         {metrics['roughness_proxy']:.6f}")
                print(f"  Mean Velocity:           {metrics['mean_velocity']:.4f}")
                print(f"  Max Error:               {metrics['max_error']:.6f}")
                print(f"  Mean KCM Intervention:   {metrics['mean_kcm_intervention']:.6f}")
                print(f"  Steps:                   {metrics['steps']}")
                print(f"  Progress:                {final_progress:.4f}")
                print(f"  Total Reward:            {episode_reward:.2f}")
                print(f"{'='*80}\n")
            
            # 定期保存模型（仅对神经网络智能体）
            if (episode + 1) % save_interval == 0 and hasattr(agent, 'actor'):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                mode_suffix = f"_{experiment_mode}" if experiment_mode != 'train' else ""
                model_path = os.path.join(script_dir, f'tracking_model{mode_suffix}_ep{episode+1}.pth')
                torch.save({
                    'actor': agent.actor.state_dict(),
                    'critic': agent.critic.state_dict(),
                    'config': config
                }, model_path)
            
            # 更新进度条
            pbar.set_postfix({
                'Reward': f'{episode_reward:.1f}',
                'Smoothed': f'{smoothed_rewards[-1]:.1f}',
                'Actor Loss': f'{avg_actor_loss:.2f}',
                'Critic Loss': f'{avg_critic_loss:.2f}'
            })
            pbar.update(1)
    
    # 关闭日志文件
    step_log.close()
    episode_log.close()
    paper_metrics_log.close()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("\n" + "="*80)
    print(f"训练完成！论文指标已保存到 {os.path.join(script_dir, 'logs', 'paper_metrics.csv')}")
    print("="*80 + "\n")
    
    # 保存最终模型（仅对神经网络智能体）
    if hasattr(agent, 'actor'):
        mode_suffix = f"_{experiment_mode}" if experiment_mode != 'train' else ""
        final_model_path = os.path.join(script_dir, f'tracking_model{mode_suffix}_final.pth')
        torch.save({
            'actor': agent.actor.state_dict(),
            'critic': agent.critic.state_dict(),
            'config': config
        }, final_model_path)
        print(f"\n最终模型已保存: {final_model_path}")
    
    # 可视化最终轨迹
    print(f"\n可视化最终轨迹 (ε={env_config['epsilon']:.3f})")
    visualize_final_path(env)


def test(config_path: str = 'configs/default.yaml', model_path: str = None):
    """
    测试函数 - 加载已训练模型进行推理
    
    Args:
        config_path: 配置文件路径
        model_path: 模型文件路径
    """
    # 配置中文字体
    configure_chinese_font()
    
    # 加载配置
    config = load_config(config_path)
    print(f"加载配置: {config_path}")
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建环境
    env_config = config['environment']
    kcm_config = config['kinematic_constraints']
    path_config = config['path']
    experiment_config = config.get('experiment', {})
    experiment_mode = experiment_config.get('mode', 'test')
    
    # 生成参考路径
    if path_config['type'] == 'waypoints':
        Pm = [np.array(wp) for wp in path_config['waypoints']]
    else:
        path_type = path_config['type']
        scale = path_config.get('scale', 10.0)
        num_points = path_config.get('num_points', 200)
        kwargs = {}
        if path_type in path_config:
            kwargs.update(path_config[path_type])
        Pm = get_path_by_name(path_type, scale=scale, num_points=num_points, **kwargs)
    
    env = Env(
        device=device,
        epsilon=env_config['epsilon'],
        interpolation_period=env_config['interpolation_period'],
        MAX_VEL=kcm_config['MAX_VEL'],
        MAX_ACC=kcm_config['MAX_ACC'],
        MAX_JERK=kcm_config['MAX_JERK'],
        MAX_ANG_VEL=kcm_config['MAX_ANG_VEL'],
        MAX_ANG_ACC=kcm_config['MAX_ANG_ACC'],
        MAX_ANG_JERK=kcm_config['MAX_ANG_JERK'],
        Pm=Pm,
        max_steps=env_config['max_steps']
    )
    
    # 创建PPO智能体
    ppo_config = config['ppo']
    agent = PPOContinuous(
        state_dim=env.observation_dim,
        hidden_dim=ppo_config['hidden_dim'],
        action_dim=env.action_space_dim,
        actor_lr=ppo_config['actor_lr'],
        critic_lr=ppo_config['critic_lr'],
        lmbda=ppo_config['lmbda'],
        epochs=ppo_config['epochs'],
        eps=ppo_config['eps'],
        gamma=ppo_config['gamma'],
        device=device
    )
    
    # 加载模型
    if model_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'tracking_model_final.pth')
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return
    
    checkpoint = torch.load(model_path, map_location=device)
    agent.actor.load_state_dict(checkpoint['actor'])
    agent.critic.load_state_dict(checkpoint['critic'])
    print(f"成功加载模型: {model_path}")
    
    # 测试运行
    agent.actor.eval()
    agent.critic.eval()
    
    state = env.reset()
    paper_metrics = PaperMetrics()
    done = False
    total_reward = 0.0
    dt = env.interpolation_period
    
    print("\n开始测试...")
    path_label = path_config.get('type', 'path')
    log_filename = f"experiment_results_{experiment_mode}_{path_label}.csv"

    with DataLogger(filename=log_filename) as data_logger, torch.no_grad():
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, info = env.step(action)

            ref_point = DataLogger.project_to_path(
                position=env.current_position,
                path_points=env.Pm,
                segment_index=info.get('segment_idx', getattr(env, 'current_segment_idx', 0))
            )

            data_logger.log_step(
                dt=dt,
                position=env.current_position,
                reference_point=ref_point,
                velocity=env.velocity,
                acceleration=env.acceleration,
                jerk=env.jerk,
                contour_error=info.get('contour_error', 0.0),
                kcm_intervention=info.get('kcm_intervention', 0.0),
                reward_components=getattr(env, 'last_reward_components', {}),
            )
            
            paper_metrics.update(
                contour_error=info['contour_error'],
                jerk=info['jerk'],
                velocity=env.velocity,
                kcm_intervention=info['kcm_intervention']
            )
            
            total_reward += reward
            state = next_state
    
    # 计算并显示指标
    metrics = paper_metrics.compute()
    print("\n" + "="*80)
    print("测试结果 - 论文指标:")
    print("="*80)
    print(f"  RMSE Error:              {metrics['rmse_error']:.6f}")
    print(f"  Mean Jerk:               {metrics['mean_jerk']:.6f}")
    print(f"  Roughness Proxy:         {metrics['roughness_proxy']:.6f}")
    print(f"  Mean Velocity:           {metrics['mean_velocity']:.4f}")
    print(f"  Max Error:               {metrics['max_error']:.6f}")
    print(f"  Mean KCM Intervention:   {metrics['mean_kcm_intervention']:.6f}")
    print(f"  Steps:                   {metrics['steps']}")
    print(f"  Progress:                {info['progress']:.4f}")
    print(f"  Total Reward:            {total_reward:.2f}")
    print("="*80 + "\n")
    
    # 可视化轨迹
    visualize_final_path(env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练/测试基于PPO的轨迹跟踪智能体')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='配置文件路径')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'test', 'ablation_no_kcm', 'ablation_no_reward', 
                               'baseline_nnc', 'baseline_s_curve'],
                       help='运行模式')
    parser.add_argument('--model', type=str, default=None,
                       help='测试模式下的模型路径')
    
    args = parser.parse_args()
    
    # 如果通过命令行指定模式，则更新配置
    if args.mode != 'train':
        # 加载配置并修改实验模式
        config = load_config(args.config)
        config.setdefault('experiment', {})
        config['experiment']['mode'] = args.mode
        
        # 保存临时配置
        script_dir = os.path.dirname(os.path.abspath(__file__))
        temp_config_path = os.path.join(script_dir, 'temp_config.yaml')
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True)
        
        if args.mode == 'test':
            test(temp_config_path, args.model)
        else:
            train(temp_config_path)
        
        # 删除临时配置
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
    else:
        train(args.config)
