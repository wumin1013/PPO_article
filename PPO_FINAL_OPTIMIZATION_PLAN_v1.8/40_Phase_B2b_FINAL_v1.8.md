# Phase B2b：出弯回线（Recovery）（FINAL v1.8）
版本日期：2026-01-03  
依赖：B2a 已得到至少 1 个不退化且平滑改善的候选 run。

---

## 1) 目标（解决“慢漂/越界拉不回来”）
相对 baseline_ref（P0_L2）以及 B2a 最优候选：
- corner_end 后固定窗口内，误差收敛更快、带外步数更少
- 不允许通过“全程更慢”来伪改善
- 仍需满足必须项（可达性/贴线/效率/硬约束）

---

## 2) 交付物（每个 B2b run 必须产出）
- summary.json：新增 “corner_end_window” 的回线指标
  - e_n 进入带内所需步数
  - 带外步数占比（window 内）
  - e_n 峰值（window 内）
- plots：
  - e_n(t) 的局部放大（corner_end 附近）
  - v(t)（确保不是全程降速）
- main_table.csv：新增一行（含回线指标）

---

## 3) 修改流程（不写代码细节，只写步骤）
### Step 1：固定“回线窗口口径”
- 从 B1 的定义中选定 window（N 步或时间）
- 将该口径写入 manifest（避免换口径导致不可比）

### Step 2：设计最小改动（一次只改 1 项）
- 优先改 reward/判定逻辑中与 recovery 相关的一个旋钮
- 若问题来自“仍在 corner_mask”，先修 corner_mask 的退出判定（口径优先于奖励）

### Step 3：运行并归档
- 训练/评测 → A1 固化 → A3 聚合 → plotter 出图
- 输出 1 页摘要：回线指标 + 局部曲线

---

## 4) PASS/FAIL（建议）
PASS：
- 必须项不退化
- 回线指标改善（例如：window 内带外步数下降、回到带内更快）
- v(t) 不显示“全程更慢”的伪改善

FAIL：
- 必须项退化，或回线未改善

---

## 5) 论文映射
- Fig：可在 `fig:curve_butterfly` 的局部子图展示 recovery 段对比
- Table：`tab:results` 可增加 recovery 指标列（或作为补充材料）

---

## 6) 自动论文产物生成（B2b 完成后必做）
- 自动生成：`paper_assets/figures/recovery_zoom_*`
- 自动更新：`paper_assets/tables/tab_results.csv`（加入 recovery 列）

---

## 7) 一键清理（可选）
仅保留：P0_L2 基线包 + aggregation + B2a/B2b 各自最优 Run Bundle。

前提：已完成 A3 聚合并更新 `main_table.csv`。

```bash
python PPO_project/tools/cleanup_keep_best.py --phases B2a,B2b --apply
```
