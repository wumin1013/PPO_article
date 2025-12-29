# PPO_project 工作流文档（FINAL v1.6）
版本日期：2025-12-30

你指出的担忧成立：把“尺子 + 机器 + 流水线”全部塞进一个 Phase A，会带来心理负担和实现风险。  
解决方式不是砍掉它们，而是把 Phase A **拆成三个可独立验收的子阶段（Sub-phases）**，并明确“最小可进入 B 阶段的闸门”。

---

## 0) 更新后的主线（推荐执行顺序）
**Phase A-1（先造尺子：指标/分段/验收口径） → Phase A-2（再造机器：加速） → Phase A-3（最后造流水线：多 seed 自动化） → Phase B1 → B2a → B2b → B2c → Phase C → Phase D**

> 关键变化：  
> - 原来 Phase C 的“指标计算与分段口径固化”被提前到 **Phase A-1**。  
> - Phase A 被拆为 A-1/A-2/A-3，每个都有独立 PASS/FAIL。


## 0.5) 已完成与当前起点（你现在在哪）
- **P0 已执行完成**：其补丁指令已归档到 `00_Archive/00_P0_DONE_v4.md`（无需重复执行）。
- 接下来不要立刻堆新 patch：先做一次“**规则栈瘦身/模块归类**”（见 `05_P4-P8_Detox_and_Baseline_Split.md`），把 P4/P6/P7/P8 中的 *安全约束* 与 *启发式规划* 拆开，否则后续 B1/B2 的调试会被耦合项拖死。

---

## 1) 闸门定义（避免“一口吃成胖子”）
### 进入 Phase B（B1）的最小闸门（必须满足）
- **A-1 已 PASS**：你已经有可信的指标与分段口径，能看清 B 阶段是不是变好还是变坏
- **A-2 建议 PASS**：否则 B 阶段多 seed 会拖慢节奏（但允许在 B1 early 阶段并行推进 A-2）

### 进入 Phase D（大规模实验）的闸门（必须满足）
- **A-3 已 PASS**：否则多 seed + 多 baseline 会把你拖进“人肉跑实验”的地狱

---

## 2) 三条纪律（降低返工风险）
1. **闸门式推进**：一次只推进一个阶段；FAIL 必须回到上一闸门定位。
2. **一次只改一种学习问题定义**：B 阶段只动“状态/奖励/软约束”；大规模目录重构放到 Phase C。
3. **统一证据链**：结论只认 `summary.json + plots/ + episodes`，禁止手工截图式结论。

---

## 3) 固定执行闭环（每个阶段都照抄）
1) smoke：短评估，确认数值稳定、指标能统计、产物能落盘  
2) train：固定训练预算与 seed 集合  
3) eval：deterministic + 固定 eval seed + 固定轨迹集  
4) compare：与 baseline（P0 或上一闸门）做同口径对比  
5) archive：保存 meta / summary / plots / episodes

---

## 4) 强制产物清单（每个阶段必须齐）
- `summary.json`：包含目标文档指标（A-1 起就尽量齐全）
- `meta.json`：seed、git commit、配置摘要、时间戳、硬件信息
- `plots/`：overlay / corner zoom / v_ratio 曲线（最小集合）
- `episodes.jsonl`：逐 episode 的 done_reason / max_abs_e_n / steps / success

