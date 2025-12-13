"""
PPO算法实现模块
包含策略网络、价值网络和PPO训练算法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from src.utils import rl_utils
from typing import Optional


class ValueNet(nn.Module):
    """价值网络 - 估计状态价值"""
    
    def __init__(self, state_dim: int, hidden_dim: int):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc3 = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.fc4 = nn.Linear(hidden_dim//4, 1)
        
        # 层归一化
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim//2)
        self.ln3 = nn.LayerNorm(hidden_dim//4)
        
        self.dropout = nn.Dropout(0.1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = self.ln1(x)
        x = self.dropout(x)
        
        x = F.elu(self.fc2(x))
        x = self.ln2(x)
        x = self.dropout(x)
        
        x = F.elu(self.fc3(x))
        x = self.ln3(x)
        
        return self.fc4(x)


class PolicyNetContinuous(nn.Module):
    """策略网络 - 输出连续动作的均值和标准差"""
    
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        # 共享基础层
        self.shared_fc1 = nn.Linear(state_dim, hidden_dim)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        
        # 角速度控制分支
        self.angle_fc = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.angle_mu = nn.Linear(hidden_dim//4, 1)
        self.angle_std = nn.Linear(hidden_dim//4, 1)
        
        # 线速度控制分支
        self.speed_fc = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.speed_mu = nn.Linear(hidden_dim//4, 1)
        self.speed_std = nn.Linear(hidden_dim//4, 1)
        
        # 层归一化
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim//2)
        
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.angle_mu.weight)
        nn.init.xavier_uniform_(self.speed_mu.weight)
        nn.init.constant_(self.angle_std.bias, 0.3)
        nn.init.constant_(self.speed_std.bias, 0.3)

    def forward(self, x):
        x = F.elu(self.shared_fc1(x))
        x = self.ln1(x)
        x = F.elu(self.shared_fc2(x))
        x = self.ln2(x)
        
        # 角速度分支
        angle_feat = F.selu(self.angle_fc(x))
        angle_mu = torch.tanh(self.angle_mu(angle_feat))
        angle_std = F.softplus(self.angle_std(angle_feat)) + 1e-3
        
        # 线速度分支
        speed_feat = F.selu(self.speed_fc(x))
        speed_mu = torch.sigmoid(self.speed_mu(speed_feat))
        speed_std = F.softplus(self.speed_std(speed_feat)) + 1e-3
        
        mu = torch.cat([angle_mu, speed_mu], dim=1)
        std = torch.cat([angle_std, speed_std], dim=1)
        
        return mu, std


class PPOContinuous:
    """PPO算法 - 处理连续动作空间"""
    
    def __init__(self, 
                 state_dim: Optional[int],
                 hidden_dim: int,
                 action_dim: Optional[int],
                 actor_lr: float,
                 critic_lr: float,
                 lmbda: float,
                 epochs: int,
                 eps: float,
                 gamma: float,
                 device: torch.device,
                 ent_coef: float = 0.0,
                 observation_space=None,
                 action_space=None):
        """
        初始化PPO算法
        
        Args:
            state_dim: 状态空间维度（可为None，优先从observation_space推断）
            hidden_dim: 隐藏层维度
            action_dim: 动作空间维度（可为None，优先从action_space推断）
            actor_lr: Actor学习率
            critic_lr: Critic学习率
            lmbda: GAE参数
            epochs: 每次更新的epoch数
            eps: PPO clip参数
            gamma: 折扣因子
            ent_coef: 熵权重系数
            device: 计算设备
            observation_space: 可选的观测空间，用于自动推断输入维度
            action_space: 可选的动作空间，用于自动推断输出维度
        """
        resolved_state_dim = state_dim
        if observation_space is not None and hasattr(observation_space, "shape"):
            resolved_state_dim = int(np.prod(observation_space.shape))
        resolved_action_dim = action_dim
        if action_space is not None and hasattr(action_space, "shape"):
            resolved_action_dim = int(np.prod(action_space.shape))
        if resolved_state_dim is None:
            raise ValueError("state_dim must be provided or inferable from observation_space")
        if resolved_action_dim is None:
            raise ValueError("action_dim must be provided or inferable from action_space")

        self.state_dim = resolved_state_dim
        self.action_dim = resolved_action_dim
        self.actor = PolicyNetContinuous(resolved_state_dim, hidden_dim, resolved_action_dim).to(device)
        self.critic = ValueNet(resolved_state_dim, hidden_dim).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.max_grad_norm = 0.5
        self.ent_coef = ent_coef
        
        # 学习率调度器
        self.actor_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.actor_optimizer, mode='min', factor=0.5, patience=10)
        self.critic_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.critic_optimizer, mode='min', factor=0.5, patience=5)
    
    def take_action(self, state):
        """根据当前状态选择动作"""
        if not hasattr(self, 'state_tensor'):
            self.state_tensor = torch.empty((1, len(state)), dtype=torch.float, device=self.device)
        
        self.state_tensor[0] = torch.tensor(state, dtype=torch.float)
        mu, sigma = self.actor(self.state_tensor)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        
        return action.squeeze().cpu().numpy().tolist()
    
    def update(self, transition_dict):
        """更新策略和价值网络"""
        # 转换为Tensor
        states = torch.tensor(np.array(transition_dict['states']), 
                             dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), 
                              dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], 
                              dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), 
                                  dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], 
                            dtype=torch.float).view(-1, 1).to(self.device)
        
        # 计算TD目标和优势函数
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        
        # 优势函数标准化
        if advantage.std() > 1e-6:
            normalized_advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            advantage = torch.clamp(normalized_advantage, min=-10.0, max=10.0)
        
        # 计算旧策略的log概率
        mu, std = self.actor(states)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        old_log_probs = action_dists.log_prob(actions)

        # 多轮更新
        for _ in range(self.epochs):
            mu, std = self.actor(states)
            std = torch.clamp(std, min=1e-4, max=1.0)
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(actions)
            
            # 熵正则化（系数由配置控制，默认压制探索）
            entropy_bonus = action_dists.entropy().sum(dim=1).mean() * self.ent_coef
            
            # 计算概率比
            log_ratio = log_probs.sum(dim=1, keepdim=True) - old_log_probs.sum(dim=1, keepdim=True)
            log_ratio = torch.clamp(log_ratio, min=-5, max=5)
            ratio = torch.exp(log_ratio)
            
            # PPO裁剪目标
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            
            # Actor损失
            policy_loss = -torch.min(surr1, surr2).mean()
            actor_loss = policy_loss - entropy_bonus
            actor_loss = torch.clamp(actor_loss, min=-10.0, max=10.0)
            
            # Critic损失
            critic_loss = F.smooth_l1_loss(self.critic(states), td_target.detach())
            
            # 反向传播
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.max_grad_norm)
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            
        # 更新学习率
        self.actor_scheduler.step(abs(actor_loss.item()))
        self.critic_scheduler.step(critic_loss.item())
        
        return actor_loss.item(), critic_loss.item()
