# Phase A1：Run Bundle 归档标准（反复使用）（FINAL v1.7.2）
版本日期：2025-12-30  

定位：**反复复用的落盘契约**。从 Phase B 开始以及后续所有阶段，**每一次 run（训练/评估/消融），无论 PASS/FAIL，都必须满足本标准并形成 Run Bundle（Level-2）**。没有归档，就不存在“通过”，也不存在“可用于论文对比”。

## A-1（标准）：每个 run 都能固化为 Run Bundle（以后 PhaseB 的“落盘协议”）
A-1 的目标：**任何 run（成功或失败）都可以被“封存”**，并且可追溯其输入/输出/对比基线/判定依据。

### A-1 必须覆盖的两类 run
1) 训练成功 run：用于论文主表与改进结论  
2) 训练失败 run：用于调试与证明“失败原因被记录，而不是被吞掉”

### A-1 的最小输入输出契约
- 输入：config、checkpoint、评测输出（至少 smoke+eval）、baseline_ref（可选但推荐）
- 输出：Run Bundle（见 README 的规范）

---

## A-1 的关键扩展：seed_eval + episode_set（可选、默认不启用）
你已经明确想在生成 P0_gold 时就完成这两件事。为了不破坏结构，建议采用“可选参数 + manifest 追溯”的方案。

### seed_eval（可选）
- 作用：让评测具备可重复性
- 要求：不提供时行为不变；提供时写入 summary 与 manifest

### episode_set（可选）
- 作用：让评测从“重复同一轨迹”升级为“评测集合统计”
- 要求：不提供时行为不变；提供时输出 per_episode 与 aggregate

---
