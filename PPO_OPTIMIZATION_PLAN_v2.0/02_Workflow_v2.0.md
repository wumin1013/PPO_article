# PPO_project 总工作流（v2.0 极简路线）
版本日期：2026-01-17

---

## 0) v1.9 → v2.0 工作流变化

### 保留的 Phases（无需重做）
- **Phase 20**: Cleanup & Solidification ✅
- **Phase 21**: StateSpace Redesign (12维) ✅
- **Phase 22**: P0 Baseline Retrain ✅
- **Phase 23**: P0_L2 Archive ✅

### 废弃的 Phases
- **Phase 30 (v1.9 B2a)**: corridor/dir_pref 规则路线 ❌

### 新增的 Phases
- **Phase 30**: Minimal Reward（极简奖励实验）
- **Phase 31**: Curvature Observation（曲率状态扩展，可选）
- **Phase 32**: Multi-Path Validation（多轨迹验证）
- **Phase 33**: Ablation Study（消融实验）
- **Phase 40**: Paper Artifacts（论文产物）

---

## 1) 总执行顺序

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     已完成（v1.9 遗产，继续使用）                         │
├─────────────────────────────────────────────────────────────────────────┤
│  Phase 20: Cleanup           → DONE                                     │
│  Phase 21: StateSpace 12D    → DONE                                     │
│  Phase 22: P0 Baseline       → DONE                                     │
│  Phase 23: P0_L2 Archive     → DONE                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     v2.0 新起点：极简奖励实验                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Phase 30: Minimal Reward                                               │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │ • 实现极简奖励函数（progress + boundary + time）                   │ │
│  │ • 删除 corridor/dir_pref/r_track/r_dir 等复杂项                   │ │
│  │ • 在 square 路径上训练，验证策略是否涌现平滑行为                   │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                           ┌────────┴────────┐
                           │ 涌现平滑行为？   │
                           └────────┬────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              [是：继续]      [部分：调优]     [否：Phase 31]
                    │               │               │
                    ▼               │               ▼
┌──────────────────────────────────┐│  ┌────────────────────────────────┐
│ Phase 32: Multi-Path Validation  ││  │ Phase 31: Curvature Observation│
│ • line / s_shape / sharp_angle   ││  │ • 状态加入 kappa, dkappa_ds    │
│ • 验证泛化能力                   ││  │ • 让策略"看到"曲率信息         │
└──────────────────────────────────┘│  │ • 重新在 square 上训练         │
                    │               │  └────────────────────────────────┘
                    │◄──────────────┘               │
                    │                               │
                    ▼                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 33: Ablation Study                                                │
│ • 对比实验：minimal vs P0_L2 baseline                                   │
│ • 消融：time_penalty 权重敏感性                                         │
│ • 消融：boundary penalty 阈值敏感性                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 40: Paper Artifacts                                               │
│ • 轨迹对比图（Fig 3-5）                                                 │
│ • 指标汇总表（Tab 1-2）                                                 │
│ • 消融分析表（Tab 3）                                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2) 依赖关系

| 阶段 | 前置依赖 | 产出物 |
|------|----------|--------|
| Phase 30 | Phase 23 完成（P0_L2 基线可用） | minimal reward 模型 |
| Phase 31 | Phase 30 未涌现（可选） | curvature-aware 模型 |
| Phase 32 | Phase 30 或 31 涌现 | 多路径验证结果 |
| Phase 33 | Phase 32 完成 | 消融实验表格 |
| Phase 40 | Phase 33 完成 | 论文图表 |

---

## 3) 快速验证循环

每个实验遵循以下循环：

```
1. Hypothesis   → 本次改什么？为什么？
2. Config       → 修改 yaml 配置（一次只改一项）
3. Train        → 训练 100-200 episodes（快验）
4. Evaluate     → 确定性 rollout + 指标计算
5. Decision     → PASS/FAIL/需调整
6. Archive      → Run Bundle 固化
```

---

## 4) 决策树

### 4.1 Phase 30 决策

```
Phase 30 训练完成后：
│
├─ success_rate >= 0.95 AND max_abs_e_n <= ε/2?
│   ├─ 是 → 检查涌现指标
│   │       ├─ corner_peak_omega < 0.9 × MAX_ANG_VEL?
│   │       │   ├─ 是 → ✅ 涌现成功，进入 Phase 32
│   │       │   └─ 否 → 🔄 尝试增加 time_penalty
│   │       │
│   │       └─ inside_ratio > 0.5? (观察用)
│   │           ├─ 是 → 记录为"自然涌现内切"
│   │           └─ 否 → 记录为"未涌现内切但平滑"
│   │
│   └─ 否 → ❌ 基础失败
│           ├─ success_rate 低 → 检查 boundary penalty 是否过严
│           └─ 越带 → 检查 boundary penalty 是否过宽
```

### 4.2 Phase 31 决策（仅当 Phase 30 失败时）

```
Phase 31 训练完成后：
│
├─ corner_peak_omega 相比 Phase 30 显著下降?
│   ├─ 是 → ✅ 曲率状态有效，进入 Phase 32
│   └─ 否 → 🔄 考虑加入轻量 curvature penalty
```

---

## 5) Stop Rules

| 触发条件 | 操作 |
|----------|------|
| success_rate < 0.8 | 停止，检查 boundary penalty |
| steps > 2.0× baseline | 停止，检查是否"绕路" |
| 连续 3 次实验无改善 | 暂停，重新评估假设 |
| corner_peak_omega 无变化 | 考虑进入 Phase 31 |

---

## 6) 交付物清单

### 6.1 每个 Phase 的交付物

| Phase | 交付物 |
|-------|--------|
| 30 | `configs/train_square_minimal.yaml`, `artifacts/minimal/`, 涌现分析报告 |
| 31 | `configs/train_square_curvature.yaml`, `artifacts/curvature/`, 对比报告 |
| 32 | 多路径 summary 表, 泛化性分析 |
| 33 | `ablation_table.csv`, 敏感性分析图 |
| 40 | `paper_assets/figures/`, `paper_assets/tables/` |

### 6.2 最终论文产物

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
│   └── tab_ablation.csv
└── data/
    └── reproducibility_manifest.json
```

---

## 7) 时间估算

| Phase | 预计时间 | 说明 |
|-------|----------|------|
| Phase 30 | 2-4 小时 | 代码修改 + 1-2 轮训练 |
| Phase 31 | 2-4 小时 | 可选，仅当 30 失败 |
| Phase 32 | 4-6 小时 | 3 种路径验证 |
| Phase 33 | 4-6 小时 | 消融实验 |
| Phase 40 | 2-4 小时 | 图表生成 |

**最快路径**：如果 Phase 30 直接涌现成功，跳过 Phase 31，预计 12-20 小时完成。

---

## 8) 与 v1.9 的对比

| 维度 | v1.9 | v2.0 |
|------|------|------|
| Phase 数量 | 10+ | 5 |
| 奖励函数复杂度 | 15+ 权重 | 4 项 |
| 调参难度 | 高 | 低 |
| 科研叙事 | 规则工程 | 端到端学习 |
| 预计总时间 | 数周 | 1-2 天 |
