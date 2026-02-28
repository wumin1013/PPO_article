# Phase 31：固定前瞻 + 聚焦跟踪学习（v4.0 简化执行稿）

版本日期：2026-02-28  
依赖：31A（策略有效性修复，已完成）  
替代：31_Phase_Lookahead_as_Observation_v3.2.md

---

## 1. 问题回顾与方案演变

### v3.2 方案的问题

v3.2 试图让策略学习控制前瞻距离（3维动作空间），但存在以下缺陷：
- **信用分配链过长**：动作→前瞻距离→参考方向→heading_error→reward
- **前瞻奖励与主任务冲突**：`r_lookahead` 把策略拉向预设目标而非最优行为
- **策略负担过重**：同时学跟踪+区域感知+前瞻控制

### v4.0 简化方案

核心洞察：**前瞻距离不需要学**。前瞻信息作为观测（让策略"看到"前方弯道）已经足够。策略唯一需要学的是：看到弯道信号后**如何减速和转向**。

---

## 2. 方案定义

### 2.1 状态空间（18维，不变）
- 8维核心特征 + 4维拐角感知 + 6维多尺度前瞻观测（3个尺度 × 2维）
- 前瞻观测提供"前方弯道信号"，策略自主决定如何响应

### 2.2 动作空间（2维）
- `[θ_u, v_u]`：角速度归一化 + 线速度归一化
- 移除 `u_lookahead` — 策略不控制前瞻距离

### 2.3 前瞻距离（自动，非策略控制）
- 基于 cornerness 自动插值：
  - 直线区：`straight_dist = 1.2`
  - 拐角区：`corner_dist = 2.8`
  - `base_dist = straight_dist + region_weight × (corner_dist - straight_dist)`
- `mix_gain = 0.0`：完全使用 base_dist，无策略调节

### 2.4 奖励函数（聚焦主任务）
- `r_progress`：进度正奖励（w_s=20）
- `r_track`：轮廓误差惩罚（cornerness 动态调度）
- `r_dir`：航向误差惩罚
- `r_smooth`：cornerness 驱动的平滑惩罚
- **移除 `r_lookahead`**：不对前瞻距离施加额外约束

---

## 3. 主配置

使用：`PPO_project/configs/train_square_v31_simplified.yaml`

关键差异（vs v3.2）：

| 参数 | v3.2 | v4.0（简化） |
|------|------|-------------|
| `policy_action` | true | **false** |
| `mix_gain` | 0.85 | **0.0** |
| `lookahead_reward.enabled` | true | **false** |
| 动作维度 | 3 | **2** |
| `actor_lr` | 2e-5 | **3e-4**（31A） |
| `ent_coef` | 0.002 | **0.005**（31A） |

---

## 4. 验收标准

### MUST（硬门槛）
- `success_rate >= 0.95`（完成率）
- `max_abs_e_n <= 0.30 × epsilon`
- `steps <= 1.2 × baseline`

### OBSERVE（涌现行为）
- 拐角区角速度峰值下降（vs 无前瞻基线）
- 直线段高速 + 拐角段自动减速
- 前瞻观测利用：策略行为在有/无前瞻特征时有显著差异

---

## 5. 执行步骤

### Step 1：冒烟验证（~5分钟）
```bash
python main.py --config configs/train_square_v31_simplified.yaml
# 观察：环境创建成功、动作维度=2、无报错
```
手动中断，确认配置无误。

### Step 2：短训验证（200 episode，~2小时）
修改 `num_episodes: 200`，运行后检查：
- reward 趋势是否呈上升曲线
- progress 是否逐步接近 1.0
- RMSE error 是否逐步下降

### Step 3：主训（800 episode，~10小时）
使用完整配置运行，收集最终指标。

### Step 4：消融对比
- A：无前瞻基线（`lookahead_obs_enabled: false`）
- B：简化前瞻（本方案）
- 对比 corner_peak_omega、progress、RMSE

---

## 6. 论文可用表述

我们将多尺度前瞻几何特征引入状态空间，使策略在决策时能感知未来路径的弯道结构。前瞻距离基于实时曲率估计自动调节（直线段短前瞻、拐角段长前瞻），策略的学习目标聚焦于"如何响应前方弯道信号"，而非"看多远"。该设计简化了动作空间维度，降低了信用分配的难度，同时保留了端到端学习的灵活性。

---

## 7. 交付物
- 主配置：`configs/train_square_v31_simplified.yaml`
- Phase31文档：本文件
- 训练输出：`saved_models/phase31_simplified/`
