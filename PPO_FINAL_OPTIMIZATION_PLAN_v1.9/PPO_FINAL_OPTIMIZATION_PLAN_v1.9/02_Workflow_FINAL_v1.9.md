# PPO_project 总工作流（FINAL v1.9.1）
版本日期：2026-01-08  
作者：PPO 项目首席系统架构师

> 目标：把"优化过程"固定成可复制的科研流水线：  
> **假设 → 最小改动 → 训练/评测 → 归档 → 出图**

---

## 1) 固定循环（每个实验只做这 5 步）

1. **Hypothesis**：这次要改善什么？为什么？
2. **最小改动**：一次只动一个旋钮
3. **Train + Eval**：产出 summary + trace
4. **归档**：Run Bundle 固化 + main_table 追加
5. **出图**：baseline vs 当前 run 对比

---

## 2) 执行顺序（v1.9.1 更新版）

### 2.1 总览图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     PHASE 20: Cleanup & Solidification                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ • 固化 _p8_speed_cap, _p4_stall 等验证过的机制                  │   │
│  │ • 移除 if enabled 开关，收敛为唯一主干                          │   │
│  │ • 提升 _scan_for_next_turn → self.turn_info                    │   │
│  │ • 简化 step() 流程，收敛 reward.py 参数为 context              │   │
│  │ • 清除 P.* 遗留代码（删除或隔离到 _legacy.py）                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ⚠️ Phase 20 是 Refactoring，不是 Rewrite                              │
│     - 不改变 Physics / Reward / Done 语义                              │
│     - 不改变状态维度（Phase 21 负责）                                   │
│     - 不引入新训练策略                                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
            ╔═════════════════════════════════════════════════════════════╗
            ║           🚧 GATE: Logic Equivalence Check 🚧               ║
            ╠═════════════════════════════════════════════════════════════╣
            ║  验证脚本: verify_logic_equivalence.py                      ║
            ║                                                             ║
            ║  通过条件（全部满足）：                                      ║
            ║  ✓ Trace 长度完全一致                                       ║
            ║  ✓ Position: |Δpos| < 1e-9                                 ║
            ║  ✓ Velocity: |Δv| < 1e-9                                   ║
            ║  ✓ Reward: |Δr| < 1e-9（含各分项）                          ║
            ║  ✓ Progress: |Δprogress| < 1e-9                            ║
            ║  ✓ Done/DoneReason: 严格一致                                ║
            ║  ✓ corner_phase / turn_sign: 严格一致                       ║
            ║                                                             ║
            ║  ❌ GATE FAILED → 禁止进入 Phase 21，必须修复差异后重验      ║
            ╚═════════════════════════════════════════════════════════════╝
                                    │
                                    ▼ (GATE PASSED)
┌─────────────────────────────────────────────────────────────────────────┐
│                     PHASE 21: StateSpace Redesign                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ • 状态空间精简：36维 → 12维                                      │   │
│  │ • 核心特征 8 维 + 拐角感知 4 维                                  │   │
│  │ • 依赖 Phase 20 提供的 self.turn_info 接口                      │   │
│  │ • 建立分段指标体系（corner_mask / non-corner）                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     PHASE 22: P0_Baseline Retrain                       │
│                     基于 12 维状态重新训练 P0                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     PHASE 23: P0_L2 Archive                             │
│                     固化新基线，归档模型与配置                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     PHASE 30: B2a Corner Smoothing                      │
│                     拐角平滑优化                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
      ┌──────────────┬──────────────┬──────────────┬──────────────┐
      │   40: B2b    │   50: B2c    │   60: C      │   70: D      │
      └──────────────┴──────────────┴──────────────┴──────────────┘
```

### 2.2 依赖关系说明

| 阶段 | 前置依赖 | 阻断后续 |
|------|----------|----------|
| **Phase 20** | 无 | Gate: Logic Equivalence Check |
| **Gate** | Phase 20 完成 | Phase 21 |
| **Phase 21** | Gate PASSED | Phase 22 |
| **Phase 22** | Phase 21 | Phase 23 |
| **Phase 23** | Phase 22 | Phase 30+ |

> [!IMPORTANT]
> **Phase 20 → Gate → Phase 21** 形成**硬依赖链**。  
> Gate 失败时，整个流水线阻塞，必须修复后重验。

---

## 3) Phase 20 门禁（Gate）详细规范

### 3.1 验证脚本

**文件**：`PPO_project/tools/verify_logic_equivalence.py`

**用法**：
```powershell
# 步骤 1：保存清理后代码
git stash

# 步骤 2：切换到清理前版本，生成 before trace
git checkout pre_phase20
python tools/verify_logic_equivalence.py --mode before --config configs/p0_l2_gold.yaml --out out/phase20_gate

# 步骤 3：切换到清理后版本，生成 after trace
git checkout phase20-cleanup
python tools/verify_logic_equivalence.py --mode after --config configs/p0_l2_gold.yaml --out out/phase20_gate

# 步骤 4：运行 diff 比较
python tools/verify_logic_equivalence.py --mode compare --out out/phase20_gate
```

### 3.2 验收标准表

| 指标 | 强制标准 | 容差（仅限解释后应用） |
|------|----------|------------------------|
| Trace 长度 | 完全一致 | 不允许差异 |
| Position (pos_x, pos_y) | 完全一致 | `< 1e-9` |
| Velocity | 完全一致 | `< 1e-9` |
| Reward (及各分项) | 完全一致 | `< 1e-9` |
| Progress | 完全一致 | `< 1e-9` |
| Contour Error | 完全一致 | `< 1e-9` |
| Done / Done Reason | 严格一致 | 不允许差异 |
| corner_phase | 严格一致 | 不允许差异 |
| turn_sign | 严格一致 | 不允许差异 |

### 3.3 Gate 判定逻辑

```python
if exact_match:
    result = "✅ GATE PASSED (exact match)"
elif all_within_tolerance:
    # 必须在报告中解释差异来源
    result = "✅ GATE PASSED (within tolerance, see diff_report.json)"
else:
    result = "❌ GATE FAILED"
    # 禁止进入 Phase 21
```

### 3.4 Gate 产出物

| 文件 | 说明 |
|------|------|
| `out/phase20_gate/trace_before.csv` | 清理前 trace |
| `out/phase20_gate/trace_after.csv` | 清理后 trace |
| `out/phase20_gate/diff_report.json` | Diff 详情 |
| `out/phase20_gate/summary.json` | PASS/FAIL 判定 |

---

## 4) Stop Rule

| 触发条件 | 操作 |
|----------|------|
| 连续 2 个 run FAIL | 暂停，检查状态空间/reward |
| 拐角改善但直线退化 | 隔离 reward（corner vs non-corner） |
| **Phase 20 Gate FAILED** | 立即停止，回滚到 `pre_phase20`，定位差异来源 |
| Trace 单步偏差 > 1e-6 | 视为严重问题，需逐行 diff 代码变更 |

---

## 5) 每次交付物

### 5.1 常规 Phase 交付物

- `config.yaml`
- Run Bundle（summary/trace/plots）
- `main_table.csv`（更新后）
- 对比图（overlay 或 v/e_n）

### 5.2 Phase 20 专属交付物

| 交付物 | 说明 | 归档位置 |
|--------|------|----------|
| `trace_before.csv` | 清理前 trace | `artifacts/phase20/` |
| `trace_after.csv` | 清理后 trace | `artifacts/phase20/` |
| `diff_report.json` | Diff 详情 | `artifacts/phase20/` |
| `summary.json` | PASS/FAIL 证明 | `artifacts/phase20/` |
| `flag_inventory.csv` | Flags 盘点表 | `artifacts/phase20/` |
| `cnc_env.py` (清理后) | 核心文件 | `src/environment/` |

---

## 6) Phase 20 关键强调

> [!WARNING]
> **Phase 20 是重构（Refactoring），不是重写（Rewrite）！**

### 6.1 Phase 20 做什么
- ✅ 固化已验证机制（移除 `if enabled` 开关）
- ✅ 提升 `_scan_for_next_turn` 为核心感知组件
- ✅ 简化代码结构和接口
- ✅ 清除无用的 P.* 遗留代码
- ✅ 为 Phase 21 的 12 维状态提取准备稳定接口

### 6.2 Phase 20 不做什么
- ❌ 不改变 Physics 语义
- ❌ 不改变 Reward 语义
- ❌ 不改变 Done/Success 判定逻辑
- ❌ 不改变状态空间维度（Phase 21 负责）
- ❌ 不引入新的训练策略或奖励项

### 6.3 版本未通过 Gate 的处置

```
1. 立即停止 Phase 20 后续工作
2. 使用 git diff 定位代码变更点
3. 对比 diff_report.json 确认差异字段
4. 修复导致差异的代码变更
5. 重新运行 verify_logic_equivalence.py
6. 循环直到 GATE PASSED
```

---

## 附录：版本历史

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| v1.9 | 2026-01-07 | 初始版本 |
| v1.9.1 | 2026-01-08 | 插入 Phase 20 + Gate: Logic Equivalence Check |
