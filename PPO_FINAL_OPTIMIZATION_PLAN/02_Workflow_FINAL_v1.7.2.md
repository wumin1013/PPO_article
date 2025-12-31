# PPO_project 工作流文档（FINAL v1.7.2）
版本日期：2025-12-30

目标：让你从“能跑”升级到“可复现、可审计、可论文复用”。  
工作流核心：**每一次训练/改动都产出同结构的 Run Bundle**，这样 PhaseB/消融/论文绘图不再靠手工整理。

---

## 0) 主线（你当前状态：P0_gold 训练中）

当 P0_gold（Level-1）完成后，主线的“最省心顺序”固定为：

## 执行顺序（你当前状态：P0_gold 训练中）
推荐最省心的顺序：
1) P0_gold 训练完成（Level-1）
2) A-0：固化为 P0_L2（Level-2 baseline）
3) Patch 05（Detox/Legacy 分层）
4) PhaseB（每次 run 都按 A-1 落盘）
5) A-3 聚合与绘图（可在 PhaseB 早期并行推进）

对应的文档拆分如下（便于 Codex 精准实现、便于你后续引用）：
- A0（一次性入口）：`10A_Phase_A0_Preflight_Once_v1.7.2.md`
- A1（反复使用的落盘契约）：`10B_Phase_A1_RunBundle_Archive_Standard_v1.7.2.md`
- A2（吞吐加速，低频）：`10C_Phase_A2_Throughput_v1.7.2.md`
- A3（聚合与论文主表，低频）：`10D_Phase_A3_Aggregation_v1.7.2.md`

一句话：**A0 只做一次；A1 每次 run 都用；A2/A3 低频并行，不得破坏 A1 的可追溯性。**

---

## 1) 你需要维护的三种目录（最少就够）
 你需要维护的三种目录（最少就够）
1) **训练输出目录**：训练过程中产生的 logs/checkpoints（允许是临时的，不直接用于论文）  
2) **评测输出目录**：每次 eval 的 summary 与可选 trace（允许是临时的）  
3) **Run Bundle（归档目录）**：按 A-1 固化后的“论文级工件”，只从这里取数、做图、做对比

原则：论文与对比只认 Run Bundle，其他都算“中间产物”。

---

## 2) Run Bundle 的最小结构（概念层描述）
一个 Run Bundle 至少要能回答四个问题：
- 这次 run 的输入是什么（config/seed/episode_set）？
- 这次 run 的输出是什么（checkpoint/eval summary/trace）？
- 这次 run 相对谁做对比（baseline_ref）？
- 这次 run 为什么判定 PASS 或 FAIL（verdict）？

具体字段规范见：`00_README_SEND_TO_CODEX_v1.7.2.md`

---

## 3) 关于 seed / episode_set（你文档里提到但容易“漂”的点）
- seed_eval：用于保证评测可重复（尤其是未来引入随机扰动/随机初始化时）
- episode_set：用于把评测从“重复同一轨迹”升级为“评测集合统计”，是论文可信度的关键支撑

两者都必须满足：**默认不启用时行为不变；启用后在 manifest 可追溯。**

---

## 4) 什么时候把东西当作“基线”？
- 只有满足 A-0/A-1 标准并封存为 Run Bundle 的，才称为基线（baseline_ref 的合法对象）。
- 当前 baseline_ref 固定为：`PPO_project/artifacts/P0_L2/P0_gold_20251230_034122`（见 `PPO_project/artifacts/P0_L2/BASELINE_REF.txt`）。
- 训练过程中某次看起来不错的 checkpoint，不进入 Run Bundle 就不算基线（否则你会在论文阶段被自己的文件管理拖死）。

---

## 5) 最小化你的认知负担：只记两条规矩

### 规矩 0：全阶段统一归档（A1）

- 从 Phase B 开始（以及后续所有阶段），**每一次 run 都必须按 A1 的归档规范落盘为 Run Bundle（Level-2）**。  
  这样你才能做：自动汇总、可复现实验、论文稳定绘图、以及“退化一眼定位”。

1) “每次 run 必须能固化成 Run Bundle”  
2) “论文只从 Run Bundle 取数和绘图”

只要这两条成立，你后面 PhaseB 的试错成本会指数下降。
