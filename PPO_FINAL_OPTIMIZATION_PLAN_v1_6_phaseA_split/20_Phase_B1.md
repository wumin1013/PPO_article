# Phase B1：7维状态 + 逃生舱 Checkpoint（FINAL v1.6）
版本日期：2025-12-30  
依赖：**A-1 必须 PASS**（你要有尺子）；A-2 强烈建议 PASS（否则多 seed 慢）；A-3 可在 B1 early 并行推进，但在 Phase D 前必须 PASS。

---

## B1 目标
- 主路径：7 维状态达到“不退化”验收并具备稳定趋势
- 逃生舱：7 维若明显训练失败，不允许卡死；切换 Plan B（12 维关键状态）推进 B2 系列

---

## B2 主路径（7维）执行要点
- 先少量 seed 看趋势，再扩展到 ≥5 seed（自动化）
- 评估同口径（deterministic + 固定 eval seed + 固定轨迹集）
- 重点监控：success_rate / max_abs_e_n / mean_v_ratio_straight

---

## B3 Checkpoint（逃生舱触发条件）
在一个固定训练预算后，若满足其一则触发 Plan B：
- `success_rate < 0.60`
- 或多 seed 长期不收敛、波动极大
- 或失败集中在同一 done_reason（说明信息不足/可控性差）

---

## B4 Plan B（12维关键状态）原则
- 不回到大状态：以 7 维为骨架，只加 5 个“高信噪比、物理意义明确、与可控性关键”的量
- 额外维度必须可解释、可归一化、跨轨迹稳定

---

## B5 验收（PASS/FAIL）
### 硬约束（必须）
- `max_abs_e_n <= half_epsilon`
- `success_rate >= 0.80`
- `stall_rate <= 0.05`
- `mean_progress_final >= 0.95`

### 不退化对比（相对 P0 baseline）
- `mean_v_ratio_straight >= baseline - 0.02`
- `rmse_e_n <= baseline * 1.05`
- `cycle_time_steps_mean <= baseline * 1.05`

### 通过判定
- 7 维 PASS：进入 B2a
- 7 维 FAIL → 触发 Plan B：Plan B PASS 后允许进入 B2a（但论文主线仍优先 7 维）

