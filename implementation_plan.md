# P1-P3 轨迹优化实施计划（更新版）

## 概述

基于 P0（残差控制 + 进度主导奖励）已验收通过，逐步实现：
1. **P1**：入弯圆弧化（LOS前瞻 + Deadzone仅入弯生效）
2. **P2**：出弯快速回线（回线奖励 + 对称性验收）
3. **P3**：速度优化（速度正激励 + 直线高效进给）

---

## 关键设计决策

> [!IMPORTANT]
> **Deadzone 仅入弯生效**：解决之前"出弯太晚"问题

| 阶段 | 设计要点 |
|-----|---------|
| P1 | `deadzone_enter_ratio=0.15`（入弯），出弯和直线=0 |
| P2 | `exit_symmetry_ratio ≥ 0.7`（入弯/出弯轨迹对称性验收） |
| P3 | `w_v=3.0~5.0`（速度正激励） |

---

## 阶段依赖

```
P0 ✅ → P1 → P2 → P3
```

---

## P1：LOS 前瞻与入弯 Deadzone

**文档**：[P1_v4.1_LOS前瞻与deadzone_ZeroActionSmoke.md](file:///c:/Users/wumin/Nutstore/1/DDPG的轨迹平滑/基于强化学习的轨迹平滑/P1_v4.1_LOS前瞻与deadzone_ZeroActionSmoke.md)

### 核心修改
- `cnc_env.py`：LOS 参考方向（L0=2.0, Lmin=1.0, Lmax=6.0）
- `reward.py`：入弯 Deadzone（仅 corner_phase="enter" 时生效）

### 验收条件
| 指标 | 阈值 |
|-----|------|
| `corner_peak_ang_acc` | 下降 ≥ 30% vs P0 |
| `corner_mean_speed` | ≥ P0的80% |
| `exit_recovery_steps` | ≤ P0 × 1.2（不恶化） |

### 自动化执行
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File PPO_project/tools/run_p1_pipeline.ps1
```

---

## P2：出弯回线奖励与对称性

**文档**：[P2_v4.1_出弯回线与速度软约束.md](file:///c:/Users/wumin/Nutstore/1/DDPG的轨迹平滑/基于强化学习的轨迹平滑/P2_v4.1_出弯回线与速度软约束.md)

### 核心修改
- `reward.py`：出弯回线奖励（exit window 期间 `r_recover = w_rec * delta_e`）

### 验收条件
| 指标 | 阈值 |
|-----|------|
| `exit_recovery_steps_mean` | 下降 ≥ 30% vs P1 |
| `exit_symmetry_ratio` | ≥ 0.7 |
| `corner_mean_speed` | ≥ P1的90% |

### 自动化执行
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File PPO_project/tools/run_p2_pipeline.ps1
```

---

## P3：速度激励与直线高效进给

**文档**：[P3_v4.1_速度激励与直线高效进给.md](file:///c:/Users/wumin/Nutstore/1/DDPG的轨迹平滑/基于强化学习的轨迹平滑/P3_v4.1_速度激励与直线高效进给.md)

### 核心修改
- `reward.py`：速度正激励 `r_speed = w_v * (v_exec / v_max)`
- `cnc_env.py`：直线段识别与速度目标上调

### 验收条件
| 指标 | 阈值 |
|-----|------|
| 直线段平均速度 | ≥ 0.85 × v_max |
| 全程平均速度 | 较P2提升 ≥ 20% |

### 自动化执行
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File PPO_project/tools/run_p3_pipeline.ps1
```

---

## 预计时间

| 阶段 | 训练时间 | 验收 |
|------|---------|------|
| P1 | 4-6 小时 | 30 分钟 |
| P2 | 4-6 小时 | 30 分钟 |
| P3 | 4-6 小时 | 30 分钟 |
| **总计** | **约 12-18 小时** | |
