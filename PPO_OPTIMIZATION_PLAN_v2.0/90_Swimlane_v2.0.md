# PPO 项目执行泳道图（v2.0 极简路线）
版本日期：2026-01-17

---

## 1) 执行总览

```
═══════════════════════════════════════════════════════════════════════════════
                            v2.0 极简路线执行图
═══════════════════════════════════════════════════════════════════════════════

                     ┌─────────────────────────────────────┐
                     │        已完成（v1.9 遗产）           │
                     │                                     │
                     │  Phase 20 ✓  →  Phase 21 ✓         │
                     │  (Cleanup)      (12D State)         │
                     │       │              │              │
                     │       ▼              ▼              │
                     │  Phase 22 ✓  →  Phase 23 ✓         │
                     │  (P0 Train)     (P0_L2 Archive)     │
                     │                                     │
                     └──────────────────┬──────────────────┘
                                        │
                                        │ v2.0 起点
                                        ▼
                     ┌─────────────────────────────────────┐
                     │         Phase 30: Minimal Reward    │
                     │                                     │
                     │   • 极简奖励：progress + boundary   │
                     │   • 无 corridor / dir_pref         │
                     │   • 在 square 上训练               │
                     │                                     │
                     └──────────────────┬──────────────────┘
                                        │
                                ┌───────┴───────┐
                                │ 涌现平滑？    │
                                └───────┬───────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
                    ▼                   ▼                   ▼
              ┌─────────┐         ┌─────────┐         ┌─────────┐
              │  ✅ 是   │         │ 🔄 部分  │         │  ❌ 否  │
              └────┬────┘         └────┬────┘         └────┬────┘
                   │                   │                   │
                   │                   │                   ▼
                   │                   │         ┌─────────────────┐
                   │                   │         │ Phase 31:       │
                   │                   │         │ Curvature State │
                   │                   │         │ (14D 状态)      │
                   │                   │         └────────┬────────┘
                   │                   │                  │
                   │     ◄─────────────┴──────────────────┘
                   │
                   ▼
         ┌─────────────────────────────────────┐
         │      Phase 32: Multi-Path Validation │
         │                                      │
         │   • 在 line / s_shape 上测试泛化    │
         │   • 使用训练好的模型（不重训）       │
         │                                      │
         └──────────────────┬───────────────────┘
                            │
                            ▼
         ┌─────────────────────────────────────┐
         │      Phase 33: Ablation Study       │
         │                                      │
         │   • No KCM / Weak Time / Strong Time │
         │   • 量化各组件贡献                   │
         │                                      │
         └──────────────────┬───────────────────┘
                            │
                            ▼
         ┌─────────────────────────────────────┐
         │      Phase 40: Paper Artifacts      │
         │                                      │
         │   • 一键生成 figures + tables       │
         │   • reproducibility manifest        │
         │                                      │
         └─────────────────────────────────────┘
                            │
                            ▼
                     ┌─────────────┐
                     │  🎉 完成！   │
                     └─────────────┘

═══════════════════════════════════════════════════════════════════════════════
```

---

## 2) 各阶段检查清单

### Phase 30: Minimal Reward

| 检查项 | 状态 | 说明 |
|--------|------|------|
| `reward.py` 添加 minimal 模式 | ⬜ | |
| `train_square_minimal.yaml` 创建 | ⬜ | |
| 训练完成（200 episodes） | ⬜ | |
| success_rate ≥ 0.95 | ⬜ | |
| max_abs_e_n ≤ 0.75 | ⬜ | |
| corner_peak_omega 评估 | ⬜ | |

### Phase 31: Curvature State（可选）

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 曲率计算函数实现 | ⬜ | |
| 状态维度 12→14 | ⬜ | |
| 训练完成 | ⬜ | |
| 与 Phase 30 对比 | ⬜ | |

### Phase 32: Multi-Path Validation

| 检查项 | 状态 | 说明 |
|--------|------|------|
| eval_line.yaml 创建 | ⬜ | |
| eval_s_shape.yaml 创建 | ⬜ | |
| line 测试通过 | ⬜ | |
| s_shape 测试通过 | ⬜ | |
| 汇总表生成 | ⬜ | |

### Phase 33: Ablation Study

| 检查项 | 状态 | 说明 |
|--------|------|------|
| ablation_no_kcm.yaml | ⬜ | |
| ablation_weak_time.yaml | ⬜ | |
| ablation_strong_time.yaml | ⬜ | |
| 全部消融实验完成 | ⬜ | |
| 消融表生成 | ⬜ | |

### Phase 40: Paper Artifacts

| 检查项 | 状态 | 说明 |
|--------|------|------|
| fig_trajectory_*.pdf 生成 | ⬜ | |
| fig_velocity_profile.pdf 生成 | ⬜ | |
| tab_main_results.csv 生成 | ⬜ | |
| tab_ablation.csv 生成 | ⬜ | |
| manifest.json 生成 | ⬜ | |
| 论文仓库集成 | ⬜ | |

---

## 3) 时间估算

| Phase | 最快 | 典型 | 最慢 |
|-------|------|------|------|
| 30 | 1.5h | 3h | 6h |
| 31 | 跳过 | 2h | 4h |
| 32 | 1h | 1.5h | 3h |
| 33 | 3h | 5h | 8h |
| 40 | 1h | 2h | 3h |
| **总计** | **6.5h** | **13.5h** | **24h** |

**最快路径**（Phase 30 直接涌现）：~6.5 小时 ≈ 1 天

---

## 4) 决策点

### D1: Phase 30 后

```
corner_peak_omega < 0.9 × MAX_ANG_VEL?
├─ Yes → Phase 32 (涌现成功)
├─ Partial → 调整 time_penalty，重试 Phase 30
└─ No → Phase 31 (加入曲率状态)
```

### D2: Phase 31 后

```
corner_peak_omega 下降 ≥ 10%?
├─ Yes → Phase 32
└─ No → 加入轻量曲率惩罚，再试
```

### D3: Phase 32 后

```
所有路径 success_rate ≥ 0.90?
├─ Yes → Phase 33
└─ No → 分析失败路径，考虑多路径联合训练
```

---

## 5) 快速参考

### 关键命令

```powershell
# Phase 30: 训练
python main.py --config configs/train_square_minimal.yaml --mode train

# Phase 30: 评估
python tools/a1_pack_run.py --run_dir artifacts/minimal_v1 --baseline artifacts/P0_L2/P0_12d_gold_20260114_174552

# Phase 32: 多路径测试
python tools/rollout_trace.py --model artifacts/minimal_v1/checkpoint.pth --config configs/eval_line.yaml --out artifacts/minimal_v1/multipath/line

# Phase 33: 消融
python main.py --config configs/ablation_no_kcm.yaml --mode train

# Phase 40: 生成产物
python scripts/generate_paper_assets.py --run_dir artifacts/minimal_v1 --multipath_dir artifacts/minimal_v1/multipath --ablation_dir artifacts/ablation_aggregate --out paper_assets
```

### 关键文件

| 文件 | 用途 |
|------|------|
| `configs/train_square_minimal.yaml` | Phase 30 主配置 |
| `src/environment/reward.py` | 奖励函数（需修改） |
| `artifacts/minimal_v1/` | 主模型产物 |
| `paper_assets/` | 论文产物 |

---

## 6) 与 v1.9 对比

| 维度 | v1.9 | v2.0 |
|------|------|------|
| Phases 数量 | 10+ | 5 |
| 奖励权重 | 15+ | 4 |
| 配置文件复杂度 | 138 行 | ~50 行 |
| 预计总时间 | 数周 | 1-2 天 |
| 科研叙事 | 规则工程 | 端到端学习 |
| 可调参数 | 多 | 少 |
| 可复现性 | 中 | 高 |
