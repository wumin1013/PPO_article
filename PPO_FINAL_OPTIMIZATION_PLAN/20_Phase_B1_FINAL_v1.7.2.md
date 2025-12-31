# Phase B1：7维状态 + 逃生舱 Checkpoint（FINAL v1.7.2）
版本日期：2025-12-30  
依赖：A-1 必须 PASS（你要先有“尺子”）；Patch 05 强烈建议 PASS（否则 PhaseB 难以解释）。

---

## 1) B1 的定位（别在这一阶段“发明新论文”）
B1 的目标不是拐角平滑，而是：  
- 在你选定的“主观测维度”（7 维）下，训练过程可控、趋势稳定、可复现  
- 同时准备好 Plan B（12 维关键状态）作为逃生舱，避免你被某次训练失败卡死一周

---

## 2) 输入与输出（B1 必须落盘）

### A1 归档要求（本阶段强制）

- **本阶段每一次 run（训练 / 评估 / 消融），无论 PASS/FAIL，都必须按 A1 归档规范生成 Run Bundle（Level-2）。**
- PASS 的前提：该 run 已形成完整 Run Bundle，且 summary.verdict=PASS。  
  （换句话说：没有归档，就不存在“通过”。）
- 失败 run 也必须归档：这是定位退化/漂移/偶发性的唯一可靠证据链。

### 输入
- baseline_ref：P0_L2（`PPO_project/artifacts/P0_L2/P0_gold_20251230_034122`，见 `PPO_project/artifacts/P0_L2/BASELINE_REF.txt`）
- 本次训练配置（包括开关组合：Hard-impact / Shaping / Observability）
- 评测集合定义（episode_set）与 seed_eval（若启用）

### 输出
- Run Bundle（按 A-1）
- 如果 7 维失败且触发 Plan B：仍然落盘，并在 verdict 记录触发原因与切换点

---

## 3) 7 维主路径（推荐执行节奏）
### 3.1 先看趋势，再扩 seed
- 先用少量 seed 快速验证“是否有明显退化/明显发散”
- 趋势稳定后再扩到 ≥5 seed（用于论文更可信的统计）

### 3.2 每个 seed 都必须可追溯
- 每个 seed 的训练与评测结果应能在 Run Bundle 中定位到（至少在 manifest 里记录 seed 列表与选择策略）
- 选择“best seed / best checkpoint”的规则必须写进 manifest（例如按 success_rate 优先，其次 max_abs_e_n）

---

## 4) 逃生舱（Plan B：12 维关键状态）
触发条件（建议）：
- 连续多个 seed 的 eval 明显失败，且失败形态一致（例如出弯慢漂导致越界）
- 或者训练曲线表现出系统性卡滞（stall_rate 上升、done_reason 集中）

要求：
- Plan B 不是“悄悄换实验设定”，必须在 Run Bundle manifest 中标注：使用了 Plan B 以及与 7 维的差异
- Plan B 的验收仍然以“不退化 + 可复现”为底线

---

## 5) B1 这一阶段最常见的坑（提前写进系统，而不是写在记忆里）
- 同一现象被多个模块共同影响（尤其 Hard-impact 与 Shaping 混在一起），导致“改了但不知道是谁起作用”
- 评测不稳定（缺少 seed_eval/episode_set 的可追溯定义），导致“看似提升其实是统计噪声”
- 失败 run 没有落盘，debug 只能靠回忆

对策：
- Patch 05 的三类拆分必须落实
- A-1 允许失败 run 也封存，并强制记录失败原因

---

## 6) B1 的 PASS/FAIL（建议写入 verdict）
PASS（进入 B2a）：
- 相对 baseline_ref：必须项不退化（可达性/误差/时间/硬约束）
- 多 seed 下趋势稳定（至少没有“只有一个 seed 偶然好看”的假象）

FAIL（仍可进入 Plan B 或回滚）：
- 必须项退化明显
- 或训练失败高度一致且可解释为“观测不足/Hard-impact 过强导致学不动”
