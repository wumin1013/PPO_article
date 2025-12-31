# PPO_project 优化与验收系统（Codex 执行版 v1.7.2）
版本日期：2025-12-30  
面向：你要把当前 P0 系统推进到「可复现 + 可审计 + 可论文复用」的 PhaseB/PhaseC/PhaseD 全链路

本文件的定位：**给 Codex 的“实现规范（Spec）+ 验收清单（Checklist）”。**  
注意：本文刻意不提供任何逐行修改/补丁/命令行示例，避免把“实现细节”写死；但会给出**接口、输入输出、约束、失败模式、验收标准**，让 Codex 可以在不破坏现有结构的前提下完成实现。

---

## 0. 你当前的状态（上下文假设）
- 你已经完成 P0，且正在训练/生成 `P0_gold`（Level-1）。
- 你希望后续每一次 PhaseB run 都能自动落盘为“Run Bundle”，便于：
  - 回归验收（防退化）
  - 消融对比（论文表格/曲线）
  - 复现实验（RCIM Artifact 友好）

---

## 1. 必须遵守的“不变性约束”（Non-breaking Contract）
Codex 实现任何改动都必须满足下面三条，否则视为失败：

1) **默认行为不变**  
   - 现有脚本/配置在“不额外提供新参数/新字段”的情况下，输出结果与目录结构不变（允许新增额外文件，但不得破坏原有产物）。

2) **结构与命名不变**  
   - 不重命名、不迁移现有目录/文件/模块；只允许“新增组件”或“在原接口上做向后兼容的扩展”。

3) **验收接口兼容**  
   - 现有 `acceptance_suite`（或等价验收工具）的原始参数与行为必须保持可用；新增能力必须是可选的。

---

## 2. 核心术语与基线层级（Level-1 vs Level-2）
为了避免你之前纠结的“基线到底取哪个”问题，这里把概念钉死：

- **Level-1（P0_gold）**：训练得到的“最好权重/最好 checkpoint”，但它本身未必满足“论文级落盘结构”。  
- **Level-2（P0_L2）**：把 Level-1 通过统一的验收与封存标准（A-0/A-1）包装成**实验层基线**，从此以后：
  - 任何 PhaseB/消融/论文图表，都以 **P0_L2** 作为 baseline_ref。
  - Level-1 只作为“来源权重/来源实验”，不直接当论文基线（除非你明确写明例外）。

一句话：**P0_gold 先生成；P0_L2 才是以后所有对比的“尺子”。**

---

## 3. 数据流总览（你需要的“完整闭环”）
你想要的闭环，必须能把每次 run 的“因果链”追溯到：

- 输入：训练配置、随机性控制、评测集合定义
- 过程：训练/选择/评测的可重放信息
- 输出：统一结构的 Run Bundle（含 manifest）
- 对比：与 baseline_ref 的可计算差异（论文表格/曲线）

因此所有阶段都围绕一个核心产物：**Run Bundle**。

---

## 4. Run Bundle 标准（A-1：以后每个 PhaseB run 都按此落盘）
Run Bundle 是一个目录级工件（artifact），它至少包含 4 类信息：

### 4.1 必需内容（Must-have）
- **manifest.json**：整包“身份证”，记录版本、hash、命令、baseline_ref、评测集合、Git 信息等
- **config snapshot**：本次训练/评测最终使用的配置快照（不可只存路径引用）
- **model snapshot**：用于 eval 的 checkpoint（或其可追溯引用 + hash）
- **evaluation outputs**：至少两种视角
  - 快速健康检查（smoke）
  - 论文对比用的主评测（eval）

### 4.2 推荐内容（Should-have）
- rollout 轨迹（trace.csv 或等价）
- 论文绘图所需的图（overlay、v(t)、e_n(t)、corner-phase 标注等）
- 运行环境信息（Python/torch 版本、GPU、OS 等）

### 4.3 manifest.json 必须包含的字段（字段名可以调整，但语义必须覆盖）
- run_id（唯一）
- created_at（时间戳）
- git（commit、dirty、branch 可选）
- baseline_ref（指向某个 Run Bundle 的路径或其唯一 id）
- config_hash / model_hash（至少二选一；推荐都有）
- evaluation：
  - deterministic 是否启用
  - seed_eval（如果支持）
  - episode_set（如果支持）
  - episodes（评测次数/样本数的定义）
  - thresholds（验收阈值快照）
- verdict：
  - passed/failed
  - 关键失败原因（例如 reached_target=false, max_abs_e_n 超阈值等）

---

## 5. 评测接口扩展（A-1 的关键：seed_eval + episode_set）
你现在的文档已经提出“把 seed / episode_set 纳入 eval 接口与封存元数据”，但缺少可执行的精确定义。这里给出**最小可行定义（MVP）**，同时保持向后兼容。

### 5.1 seed_eval（可选）
目标：让同一模型在同一评测集合下得到**可重复**的统计结果（即使未来引入噪声/随机初始状态也不崩）。

规范：
- 若未提供 seed_eval：保持现状（沿用 config 中的 seed 或默认值）。
- 若提供 seed_eval：必须统一设定：
  - Python random / numpy / torch 的随机种子
  - 环境内部若支持 seed，也应同步（不支持则记录为“ignored”写进 manifest）

### 5.2 episode_set（可选）
目标：把 “episodes=N” 从“重复同一轨迹”升级为“对一个评测集合做统计”，为论文提供坚固的对比基础。

最小语义：
- episode_set 是一个“episode 描述列表”，每个 episode 描述至少能决定：
  - 路径族（line/square/s_shape…）与其参数（scale/num_points/closed 等）
  - 必要时的初始条件（起点姿态、初速度、干扰开关等；若当前环境固定起点，可为空）
- 若未提供 episode_set：默认为 “single”，即使用当前 config 的 path 定义作为唯一 episode（完全向后兼容）。
- 若提供 episode_set：验收工具必须逐 episode 运行并输出：
  - per_episode 指标（success、max_abs_e_n、steps 等）
  - aggregate 指标（均值/分位数/最差值），并以 aggregate 作为 PASS/FAIL 判据来源（可在阈值定义中选择 mean 或 worst-case）。

### 5.3 输出兼容性
- 现有 summary.json 字段不得删除；只能新增字段或新增子结构。
- per_episode 数据量可能较大，但应保持可解析（例如列表 + 聚合摘要）。

---

## 6. Phase A（拆成四份：A0 一次性 / A1 反复用 / A2-A3 低频并行）

**你这一步的目标**：把“P0_gold（权重）”升级成论文可用的 **P0_L2（Level-2 实验层基线）**，并把“尺子”（归档/评测口径）造出来，让后续 Phase B 每次 run 都能稳定落盘、稳定对比。

- **A0（只做一次）**：`10A_Phase_A0_Preflight_Once_v1.7.2.md`  
  把 P0_gold（或你认定的最好 checkpoint）固化为 P0_L2（Level-2）。完成后冻结。
- **A1（反复使用，最关键）**：`10B_Phase_A1_RunBundle_Archive_Standard_v1.7.2.md`  
  定义 Run Bundle（Level-2）落盘契约。从 Phase B 开始：**每一次 run（训练/评估/消融），无论 PASS/FAIL，都必须按 A1 归档**。
- **A2（低频，可并行）**：`10C_Phase_A2_Throughput_v1.7.2.md`  
  吞吐加速（多 seed / 多 episode / 更快做消融），但不改变默认语义与结构。
- **A3（低频，可并行）**：`10D_Phase_A3_Aggregation_v1.7.2.md`  
  自动化汇总与论文主表聚合（把“实验汇总”写进系统，而不是写在脑子里）。

### 推荐执行顺序（你当前状态：P0_gold 训练中）

## 执行顺序（你当前状态：P0_gold 训练中）
推荐最省心的顺序：
1) P0_gold 训练完成（Level-1）
2) A-0：固化为 P0_L2（Level-2 baseline）
3) Patch 05（Detox/Legacy 分层）
4) PhaseB（每次 run 都按 A-1 落盘）
5) A-3 聚合与绘图（可在 PhaseB 早期并行推进）

---

## 7. Phase B（优化主线）如何“写到工具里”，而不是写在脑子里
 Phase B（优化主线）如何“写到工具里”，而不是写在脑子里
Phase B 的关键原则：**任何会改变动力学/终止条件的东西，都必须可开关，并且在 Run Bundle 的 manifest 里可追溯。**

因此你需要把改动分成三类，并在文档与 manifest 中标注：

1) 硬影响（Hard-impact）  
   直接改变 action→state、终止条件、KCM 限制等。必须可开关，默认不改变旧行为。

2) 奖励塑形（Shaping）  
   不改动力学，只改学习驱动。也必须可开关。

3) 观测/日志（Observability）  
   只加信息与统计，不参与奖励（默认）。这类改动最安全，应优先推进，用来解释 B2 的“为什么更平滑”。

---

## 8. Phase B2 的论文指标：拐角平滑必须“可计算、可复现、可对比”
你论文要讲“拐角平滑”，审稿人会问：平滑是什么？怎么度量？有没有不退化？

建议（作为 Spec，不强制唯一实现）：
- 在 trace 中标注 corner_mask（来自几何扫描或 corner_phase 状态）
- 指标至少包含三组：
  1) 轨迹误差：max_abs_e_n / rmse_e_n（直线与拐角分段统计）
  2) 速度与时间：cycle_time_steps、v_profile（直线段不退化）
  3) 平滑性：|dω/dt|（角加速度/jerk 的代理）、ω 峰值、拐角段速度跌落幅度

验收逻辑：
- 必须满足“不退化”（error 与 time 不超过 baseline_ref 的容忍比例）
- 平滑性指标显著改善（给出阈值或统计显著性规则）
- KCM 违规为 0（或被明确记录为硬约束外异常）

---

## 9. 最小回归测试集合（每次改动都要过）
为防止“改一处、碎一地”，建议 Codex 为工具链准备最小回归：

- 旧接口回归：不提供新参数，acceptance_suite 输出与以前一致（字段可新增，但关键数值不应跳变）
- 新接口回归：提供 seed_eval 与 episode_set（哪怕 episode_set 只有 1 个 episode），manifest 记录完整，summary 聚合逻辑一致
- 失败样例回归：构造一个明显失败的 checkpoint，确保系统仍能落盘、记录失败原因，而不是崩溃

---

## 10. 交付清单（Codex 最终应交付什么）
- 更新后的文档（你现在读的这些 + 各 Phase 文档对齐）
- 工具链能力满足：
  - A-0：把 P0_gold 变成 P0_L2（Run Bundle）
  - A-1：任何 run 都能固化为 Run Bundle（含 baseline_ref、hash、eval 集合）
  - seed_eval / episode_set：接口存在、默认不启用、启用后可追溯
- 通过最小回归测试集合

---


## 归档硬约束（Phase B/C/D 全部适用）

- **每一次 run（训练 / 评估 / 消融），无论 PASS/FAIL，都必须按 A1 的归档规范落盘为一个 Run Bundle（Level-2）**。  
  未归档 = 不可复现 = 不可用于论文对比 = 这次 run 视为“没发生”。
- Run Bundle 必须显式包含：**config 快照、seed_eval/episode_set、summary（含 verdict）、关键 trace（代表性 episodes）、checkpoint 指针/引用、baseline_ref 指针、manifest（版本/依赖/数据指纹）**。
- 归档命名必须能一眼看出：**阶段（B1/B2a/...）、改动点、评测集合、seed 组**，用于后续自动汇总与论文绘图。
