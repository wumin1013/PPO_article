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
│     - 尽量保持 Physics / Reward / Done 语义稳定（若变化必须记录）        │
│     - 不改变状态维度（Phase 21 负责）                                   │
│     - 不引入新训练策略                                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
            ╔═════════════════════════════════════════════════════════════╗
            ║           🚧 GATE: Regression Verification 🚧               ║
            ╠═════════════════════════════════════════════════════════════╣
            ║  验证脚本: acceptance_suite.py                              ║
            ║                                                             ║
            ║  通过条件（全部满足）：                                      ║
            ║  ✓ P0 回归评估 PASSED（p0_eval）                            ║
            ║  ✓ P0_L2 回归评估 PASSED（p0_eval）                         ║
            ║                                                             ║
            ║  （可选诊断）Logic Equivalence Check：失败需记录差异来源     ║
            ║                                                             ║
            ║  ❌ GATE FAILED → 禁止进入 Phase 21，必须修复回归后重验      ║
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
| **Phase 20** | 无 | Gate: Regression Verification |
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

**文件**：`PPO_project/tools/acceptance_suite.py`

**用法**：
```powershell
conda activate PPO
cd PPO_project

# 固定评估集（多 seed），避免 Gate 过窄
$episode_set = "episode_sets/phase20_gate_seeds.txt"
$episodes = 10

# Gate（阻断）：在 baseline 路径族（square）上做多 seed 回归
python tools/acceptance_suite.py --phase p0_eval --config configs/p0_l2_gold.yaml --path_type square --episode_set $episode_set --episodes $episodes --deterministic --model artifacts/P0_gold_20251230_034122/checkpoint.pth --out out/phase20_gate/p0_eval
python tools/acceptance_suite.py --phase p0_eval --config configs/p0_l2_gold.yaml --path_type square --episode_set $episode_set --episodes $episodes --deterministic --model artifacts/P0_L2/P0_gold_20251230_034122/checkpoint.pth --out out/phase20_gate/p0_l2_eval

# 可选扩展：多路径诊断覆盖（不阻断；用于发现“跨路径族”退化）
$path_types_ext = @("line", "s_shape", "sharp_angle")
foreach ($path_type in $path_types_ext) {
  python tools/acceptance_suite.py --phase p0_eval --config configs/p0_l2_gold.yaml --path_type $path_type --episode_set $episode_set --episodes $episodes --deterministic --model artifacts/P0_gold_20251230_034122/checkpoint.pth --out out/phase20_gate/extended/p0_eval/$path_type
  python tools/acceptance_suite.py --phase p0_eval --config configs/p0_l2_gold.yaml --path_type $path_type --episode_set $episode_set --episodes $episodes --deterministic --model artifacts/P0_L2/P0_gold_20251230_034122/checkpoint.pth --out out/phase20_gate/extended/p0_l2_eval/$path_type
}
```

### 3.2 验收标准表

| 指标 | 强制标准 |
|------|----------|
| P0 回归评估 | `out/phase20_gate/p0_eval/summary.json` 中 `summary.passed == true` |
| P0_L2 回归评估 | `out/phase20_gate/p0_l2_eval/summary.json` 中 `summary.passed == true` |

### 3.3 Gate 判定逻辑

```python
if p0_eval_passed and p0_l2_eval_passed:
    result = "✅ GATE PASSED (regression verification)"
else:
    result = "❌ GATE FAILED"
    # 禁止进入 Phase 21
```

### 3.4 Gate 产出物

| 文件 | 说明 |
|------|------|
| `out/phase20_gate/p0_eval/summary.json` | P0 回归评估结果（Gate 证据） |
| `out/phase20_gate/p0_l2_eval/summary.json` | P0_L2 回归评估结果（Gate 证据） |
| `episode_sets/phase20_gate_seeds.txt` | 固定评估集（多 seed） |
| `out/phase20_gate/summary.json` | Gate 汇总（可选聚合文件） |

### 3.5 可选：Logic Equivalence Check（诊断用，非 Gate）

- 脚本：`PPO_project/tools/verify_logic_equivalence.py`
- 目的：量化 pre/post 的行为差异，用于解释“为什么不等价”
- 产物（建议归档）：`trace_before.csv`, `trace_after.csv`, `diff_report.json`

---

## 4) Stop Rule

| 触发条件 | 操作 |
|----------|------|
| 连续 2 个 run FAIL | 暂停，检查状态空间/reward |
| 拐角改善但直线退化 | 隔离 reward（corner vs non-corner） |
| **Phase 20 Gate FAILED** | 立即停止，定位回归失败来源（P0/P0_L2 哪个 failed） |
|（可选）Logic Equivalence Check 失败 | 不阻断，但必须归档差异并在 Phase 20 总结中说明 |

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
| `p0_eval/summary.json` | P0 回归评估结果（Gate 证据） | `artifacts/phase20_gate/` |
| `p0_l2_eval/summary.json` | P0_L2 回归评估结果（Gate 证据） | `artifacts/phase20_gate/` |
| `summary.json` | Gate 汇总（可选） | `artifacts/phase20_gate/` |
| `trace_before.csv` | Logic Equivalence 诊断 trace（可选） | `artifacts/phase20_gate/` |
| `trace_after.csv` | Logic Equivalence 诊断 trace（可选） | `artifacts/phase20_gate/` |
| `diff_report.json` | Logic Equivalence diff（可选） | `artifacts/phase20_gate/` |
| `flag_inventory.csv` | Flags 盘点表（可选） | `artifacts/phase20_gate/` |
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
- ❌ 不改变状态空间维度（Phase 21 负责）
- ❌ 不引入新的训练策略或奖励项
- （目标）尽量不改变 Physics / Reward / Done 行为；若变化必须记录并通过回归验证

### 6.3 版本未通过 Gate 的处置

```
1. 立即停止 Phase 20 后续工作
2. 读取 `out/phase20_gate/p0_eval/summary.json` 与 `out/phase20_gate/p0_l2_eval/summary.json`，确认哪个回归失败
3. 使用 git diff 定位代码变更点，修复导致回归失败的改动
4. 重新运行 acceptance_suite（直至两份 summary 都 passed）
5. （可选）运行 verify_logic_equivalence.py 量化差异来源并归档
```

---

## 附录：版本历史

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| v1.9 | 2026-01-07 | 初始版本 |
| v1.9.1 | 2026-01-08 | 插入 Phase 20 + Gate（最初为 Logic Equivalence Check） |
| v1.9.2 | 2026-01-12 | Gate 调整为 Regression Verification，Logic Equivalence 作为诊断 |
