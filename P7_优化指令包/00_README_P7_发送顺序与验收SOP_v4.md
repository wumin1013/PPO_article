# P7 执行顺序与验收SOP（v4：操作规范/流程化/可复现）

本 SOP 的目标不是“拍脑袋觉得行”，而是把 P7 的实施做成一套**可复现、可审计、可回滚**的流程：每一步都有明确输入、输出、命令、通过门槛、失败定位路径。

---

## 0. 你要实现的行为（验收定义）
**必须同时满足：**
1) **直线段快**：速度接近 MAX_VEL（考虑 acc/jerk 爬升），不越界、不抖动  
2) **弯前降速**：降速来源于**可控性边界 v_cap**（LOS+可达性），不是 reward 强行压慢  
3) **拐角平滑内切**：允差带内自选内切幅度，轨迹圆滑（航向连续、曲率连续或近似连续）  
4) **出弯回中**：离开拐角后在有限步内回到参考附近  
5) **可靠到终点**：open path success 触发稳定；stall 不误杀

---

## 1. 强制实现顺序（不要交换）
**P7.0 → P7.2 → P7.1 → P7.3**

理由（简化版）：
- P7.0 解决“尖角/不回中/速度上不去”的结构性根因，否则后面都学不出来。
- P7.2 把“内切更聪明→允许更快”写成硬机制，否则探索没有动力。
- P7.1 放开走廊自由度并实现出弯回中（含 ramp+滞回），否则策略机械。
- P7.3 做数值安全与收口（曲率连续、stall/success 稳）。

---

## 2. 通用操作规范（每一步都要执行）
### 2.1 工作区与版本控制（强烈建议）
- 每个 P7.x 单独开分支：`p7_0_* / p7_2_* / p7_1_* / p7_3_*`
- 每次只改一个 P7.x：禁止把四个阶段混在一个 PR/commit。
- 每个阶段完成后在提交信息里写明：通过了哪些 accept 脚本、关键指标是多少。

### 2.2 可复现性设置（建议写到 tools/utils_seed.py）
每个脚本都必须支持 `--seed`，并且设置：
- python random
- numpy random
- torch seed（如用 torch）
- torch deterministic（可选，但建议）

### 2.3 输出目录规范（统一）
所有验收脚本输出到：
- `PPO_project/out/p7_xxx/YYYYMMDD_HHMMSS_*`
其中至少包含：
- `summary.json`（机器可读）
- `summary.txt`（人可读）
- `trace.csv`（1 个代表回合的逐步数据）
- `path.png`（轨迹图：参考路径+允差带+实际轨迹）
- 关键曲线图：`v_ratio_exec.png` `v_ratio_cap.png` `e_n.png` `kappa.png` `dkappa.png`

### 2.4 “红线检查”——出现即判失败
- reward/state/info 中出现 NaN/Inf（必须 assert 并保存 trace）
- `v_exec > max_vel_cap + 1e-6`（硬上限被突破）
- `|omega_exec| > MAX_ANG_VEL + 1e-6`
- corner_phase 内频繁 on/off 抖动（toggle 次数明显 > 3/回合）

---

## 3. 基线快照（实施 P7 前必须做一次）
目的：你需要一个“改坏了马上知道”的对照组。

在 PPO_project 根目录执行（Windows 可用 python.cmd）：
1) P4 现有验收（已存在）：
- `python tools/accept_p4_0_speed_and_done.py --episodes 20 --seed 42 --outdir out/baseline_p4`

2) 常量动作 sanity（已存在）：
- `python tools/sanity_constant_action.py --case open_line --episodes 1 --seed 42 --theta 0.0 --vel-ratio 1.0 --print-episode-end`
- `python tools/sanity_constant_action.py --case open_square --episodes 1 --seed 42 --theta 0.0 --vel-ratio 0.6 --print-episode-end`

把 baseline 的 `summary.json/trace.csv/path.png` 留存，后面每一步都要对比。

---

## 4. 每个阶段的“执行流程模板”（照抄执行）
以 P7.0 为例（其他阶段同理）：

### Step A：实施前（确保环境干净）
- 清理旧实验（可选）：`saved_models/p7_*`、`out/p7_*`
- 固定 seed（如 42）作为主验收 seed

### Step B：只做该阶段的代码修改
- 限定修改文件范围（文档里有清单）
- 修改完先跑：
  - `python -m compileall .`
  - `python -m pytest -q`（如果 tests 可跑）

### Step C：跑该阶段的自验证脚本（必须新增）
- `python tools/accept_p7_0_*.py --seed 42 --outdir out/p7_0_accept`

### Step D：通过门槛判定（硬判）
- accept 脚本返回码 0 才算通过
- summary.json 里所有指标达到阈值

### Step E：人工复核（3 张图 + 5 个数）
- 图：path.png / v_ratio_cap.png / kappa.png
- 数：success_rate、stall_rate、oob_rate、mean(v_ratio_exec_last20%)、dkappa_p95

### Step F：归档
- 把 out 目录复制到 `out/_archive/p7_0/<commit_id>/`
- 在文档/README 里记录指标与日期

---

## 5. 总验收（最后再做）
当 P7.0~P7.3 都通过各自 accept 后，再跑总验收：
- `python tools/accept_p7_all.py --seed 42 --episodes 20 --outdir out/p7_all`
并把结果写进论文/报告的实验表。

