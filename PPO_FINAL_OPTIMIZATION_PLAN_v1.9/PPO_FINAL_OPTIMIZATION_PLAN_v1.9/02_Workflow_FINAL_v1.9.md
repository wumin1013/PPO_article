# PPO_project 总工作流（FINAL v1.9）
版本日期：2026-01-07

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

## 2) 执行顺序（v1.9 精简版）

```
21_StateSpace（12维 + 指标口径）
    ↓
22_P0_Baseline（基于12维重训）
    ↓
23_P0_L2_Archive（固化新基线）
    ↓
30_B2a（拐角平滑）
    ↓
40_B2b / 50_B2c / 60_C / 70_D
```

---

## 3) Stop Rule
- 连续 2 个 run FAIL：暂停，检查状态空间/reward
- 拐角改善但直线退化：隔离 reward（corner vs non-corner）

---

## 4) 每次交付物
- config.yaml
- Run Bundle（summary/trace/plots）
- main_table.csv（更新后）
- 对比图（overlay 或 v/e_n）
