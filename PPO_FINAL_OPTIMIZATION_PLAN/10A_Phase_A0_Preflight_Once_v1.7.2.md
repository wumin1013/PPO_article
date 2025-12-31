# Phase A0：Preflight / 基线入口（只做一次）（FINAL v1.7.2）
版本日期：2025-12-30  

定位：**一次性入口**。把当前训练得到的 P0_gold（或你认定的最好 checkpoint）转换/固化为论文可用的 **P0_L2（Level-2 实验层基线）**。完成一次后冻结；只有当环境/评测口径/目录结构发生实质变化才重做。

## A-0（入口）：把 P0_gold 升级为 P0_L2（Level-2 实验层基线）
### 输入
- P0_gold（Level-1）的 checkpoint 与其训练配置/日志（来源可以是你现有的 pipeline 归档目录）

### 输出（必须是 Run Bundle）
- 一个符合 A-1 标准的 Run Bundle，命名为 P0_L2（或等价标识）
- 该 Run Bundle 将被后续所有 PhaseB 作为 baseline_ref

### 必须记录到 manifest
- P0_gold 来源（路径或唯一 id）
- 本次评测集合（episode_set）与 seed_eval（若支持）
- PASS/FAIL verdict 与关键指标快照

### 验收（A-0 PASS 的定义）
- smoke PASS：环境数值稳定、无 NaN、能推进
- eval PASS：满足 P0 阈值
- 产物完整：manifest + config snapshot + model snapshot + eval 输出都存在

---
