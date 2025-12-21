# Codex Runbook（Phase B v3）：可配置门禁 + Baseline 自适应 + 自动报告（工程化交付版）

> 目标：把“环境能闭环”升级为“策略能以最大效率（最小降速）平滑过弯并到终点”。  
> 风格：像 Phase A 一样硬核——**每一步都有自动化验收与（最小）人工复核**，避免训练玄学。

---

## 0) Phase B 入口条件（不满足就禁止进入）
进入 Phase B 前必须同时满足：

### 0.1 Phase A（含补丁）已通过
- Phase A DoD 全绿（check_physics / accept_p7_0 / accept_p8_1 / pytest）  
- Phase A 补丁门禁通过（Square 上 `v_ratio_exec` 非退化、speed_util 达标、v_cap 变化存在、v_exec 跟随 v_cap）  
- 用户已人工复核并明确回复 **APPROVED**（看过 Square 的 `e_n(t)` 与 `v_ratio_exec(t)`，认可闭环与速度剖面）

> 任何 Phase A 门禁失败：优先回 Phase A/补丁定位，不要在 Phase B 里“靠奖励硬掰”。

---

## 1) Phase B 的核心交付物（Definition of Done）
你最终要交付：

1) **训练可复现**：提供 config/seed/命令，能稳定复现评估结果。  
2) **到终点**：Square 与 S-shape 的评估 success rate 达标（见 6.2）。  
3) **效率提高**：带内 `mean_speed_util` 明显高于 Phase A 专家策略 baseline，且“弯前降速—出弯提速”清晰。  
4) **平滑过弯**：轨迹无明显尖角；`dkappa_exec`/`Δomega` 不出现持续大振荡。  
5) **产物齐全**：每个任务（line/square/s_shape）输出标准化 artifacts（见 6.1）。

---

## 2) Phase B 必须新增的“训练前物理可学性检查”（强烈建议）
> 你现在最大的风险不是奖励权重，而是：PPO 速度动作是否真的能改变 v_exec。  
> 这个检查能在 30 秒内排除“动作被覆盖/滤波压死/尺度错误”的坑。

### 2.1 新增脚本：`tools/accept_phaseB_action_effectiveness.py`（必须）
脚本行为（不训练）：
1) 用默认配置 reset 到同一初始状态（固定 seed）。
2) 在同一 state 下对比 3 个动作：
   - A: `v_ratio=0.2`
   - B: `v_ratio=0.5`
   - C: `v_ratio=0.8`
   `omega_ratio` 固定为 0（避免转向耦合）
3) 运行 1 步，记录 `v_ratio_exec`（或 v_exec）并输出 slope：

必须断言：
- `v_ratio_exec(C) > v_ratio_exec(B) > v_ratio_exec(A)`（严格单调）
- `v_ratio_exec(C) - v_ratio_exec(A) >= delta_min`（建议 delta_min=0.05，按你系统尺度可调）
- 无 NaN/Inf

> 如果这一步失败：说明训练必失败。优先修动作映射/滤波/裁剪，而不是调 reward。

---

## 3) Step 4：效率奖励（Time Efficiency）落地（带安全门控）
文件：`src/environment/reward.py`

### 3.1 必须保留的安全硬约束（不要为了快牺牲安全）
所有效率相关奖励都必须 gated（门控）：
- 仅在 `abs(e_n) <= half_epsilon` 且未出界/未撞墙/未终止 时生效
- 一旦出界/碰撞/终止：优先给强惩罚，并让效率项为 0（不要反向奖励冒险）

### 3.2 Progress 奖励（必须弯道不归零）
要求：
- progress 用**弧长进度增量**（Δs 或等价物），不是欧式距离
- 在弯道/角落 phase 中可以降低权重，但**禁止归零**

验收（自动）：
- 在 eval 轨迹中统计“弯道 steps”（dist_to_turn < 阈值 或 |turn_angle|>阈值），progress_reward 不应在大多数 steps 为 0

### 3.3 速度利用率奖励（强烈建议）
定义：
- `speed_util = v_ratio_exec / (v_ratio_cap + 1e-9)`（若你更信任 speed_target，可用 speed_target_ratio）
- `r_speed = w_speed * clip(speed_util, 0, 1)`（简单有界，避免炸）

注意：
- **不要**在带外奖励 speed_util，否则会鼓励贴边/越界高速
- `w_speed` 建议从小到大（例如 0.1 → 0.3 → 0.5）

验收（自动）：
- `mean_speed_util_in_band`（带内）必须显著高于 Phase A baseline（建议 +0.15 以上，见 6.2）

### 3.4 Shortcut（内切）奖励（可选，建议 Phase B 后半段再加）
只在 `is_isolated_corner == True` 启用（S 型必须禁用），并且权重必须远小于出界惩罚。
推荐形式：
- `inside = turn_sign * e_n / (half_epsilon + 1e-9)`  # inside>0 表示处于内侧半带
- `r_shortcut = w_shortcut * clip(inside, 0, 1) * clip(speed_util, 0, 1)`

验收（半自动+人工）：
- 轨迹在孤立拐角略有内切但不长期贴边；`max_abs_e_n` 仍≤ half_epsilon

---

## 4) Step 4.5：平滑过弯（把“尖角/抖动”压下去，但不掐死速度）
文件：`src/environment/reward.py`（或已有平滑模块）

### 4.1 推荐使用的平滑指标（按已有字段选）
- `dkappa_exec`（曲率变化率）惩罚：`r_smooth = -w_dkappa * abs(dkappa_exec)`
- 或 `Δomega` 惩罚：`r_smooth = -w_domega * abs(omega_exec - omega_prev)`
- 或 `Δaction` 惩罚（最通用）

权重建议：
- 从很小开始（例如 0.01~0.05），逐步加；  
- 一旦发现策略又回到“龟速不动”，说明你把平滑项加太大了。

验收（自动）：
- eval 的 `p95(|dkappa_exec|)` 不得高于 Phase A baseline（至少不恶化）
- 同时 `mean_speed_util_in_band` 不能因为平滑项而显著下降（避免“为了平滑而慢”）

---

## 5) Step 4.8：动力学模式（可选，但建议作为第二阶段改动）
> 若你在 Phase A 仍看到“角速度打满、轨迹蛇形纠偏”，建议引入 `tangent_relative`。  
> 但必须过回归门禁：不能破坏 Phase A 的专家闭环与速度剖面门禁。

### 5.1 启用 `tangent_relative` 的强制回归
启用后必须重新通过：
- Phase A DoD（含补丁门禁）
- 开局脉冲测试（reset 初始化 last_path_angle）

并且新增一条“对比门禁”：
- 在 Square eval 中，`rmse_e_n` 或 `p95(|e_n|)` 相比 heading_integrator 不得恶化（允许小幅变化，但不能爆）

---

## 6) Step 5：正式训练（分阶段 Curriculum + 明确门禁）
> 训练不是“一次跑到底”，而是过关制。

### 6.1 统一产物输出（每个阶段、每个任务都要生成）
要求新增/复用 eval 脚本：`tools/eval_policy.py`（若没有就新增）
每次评估必须输出到 `artifacts/phaseB/<stage>/<task>/`：
- `path.png`（参考轨迹 vs 实际轨迹）
- `e_n.png`
- `v_ratio_exec.png`
- `v_ratio_cap.png`
- `speed_util.png`
- `summary.json`（见 6.2）
- `trace.csv`

> 产物命名固定，便于你人工对比版本。

### 6.2 评估指标（summary.json 必须包含）
对每个 task（line/square/s_shape）至少输出：
- `success_rate`（N 次 rollout 到终点比例，建议 N=20）
- `mean_speed_util_in_band`
- `mean_progress_per_step`
- `max_abs_e_n`、`rmse_e_n`、`p95_abs_e_n`
- `p95_abs_dkappa_exec`（或 p95_abs_domega / p95_abs_delta_action）
- `stall_rate`
- `mean_episode_steps`

并输出与 Phase A baseline 的对比字段：
- `delta_mean_speed_util_vs_baseline`
- `delta_mean_progress_per_step_vs_baseline`

### 6.3 Curriculum 关卡（推荐）
关卡按“可学性”从易到难：

#### Stage 1：Line（直线高速）
目标门禁：
- success_rate >= 0.95
- mean_speed_util_in_band >= 0.7
- e_n 稳定，不出现漂移

#### Stage 2：Square（拐角提前减速+出弯提速）
目标门禁：
- success_rate >= 0.8
- mean_speed_util_in_band >= baseline + 0.15（或 >=0.55，取较严者）
- v_ratio_exec 曲线呈现结构：直线高、弯前低、出弯再高（可用自动检测：turn_window 内均值显著小于 straight_window）

#### Stage 3：Gentle S（轻 S，允许较大 epsilon 或较低 MAX_VEL）
目标门禁：
- success_rate >= 0.7
- p95_abs_dkappa_exec 不爆炸（不出现持续尖峰）
- 轨迹不穿墙

#### Stage 4：Hard S（目标任务）
目标门禁（你最终关心）：
- success_rate >= 0.6（先达标，再逐步提高）
- mean_speed_util_in_band 不低于 Square 阶段的 80%
- max_abs_e_n <= half_epsilon（允许极少数点逼近，但不能长期越界）

> 如果某关卡不过：禁止直接调下一关。先在本关卡调整 reward 权重/动作尺度/平滑项。

---

## 7) 人工验收（你只需要看 3 张图就能判断“像不像 CNC”）
每个 task，只看：
1) `path.png`：拐角是否圆滑（无尖角折线）、是否出弯贴边振荡  
2) `v_ratio_exec.png` + `v_ratio_cap.png`：是否“顶着 cap 跑”（直线高、弯前低）  
3) `e_n.png`：是否长期贴边（如果贴边换来的速度提升很小，那就是坏策略）

放行 Phase B 完成的标准：
- Square 和 Hard S 在门禁达标的前提下，肉眼认可“过弯平滑 + 速度合理”

---

## 8) 最常见失败模式（Phase B 版快速定位）
- **训练仍然龟速**：检查 2.1 动作有效性；检查效率奖励 gated 是否过严；检查平滑惩罚是否过大  
- **弯道冲出界**：检查 dist_to_turn/turn_angle 接线；检查 braking envelope 是否被其它规则覆盖；检查 speed_util 是否在带外被奖励  
- **出弯贴边抖动**：减小 shortcut 奖励；增加少量平滑项；检查 corner_phase 是否仍被 LOS 污染  
- **S 形崩**：先在 gentle S 降难度（更大 epsilon 或更低 MAX_VEL）建立可学，再渐进提高

---

## 9) 强制约束（必须原样执行）
- 在修改任何 reward/dynamics 之前，先通过 Phase A（含补丁）并获得用户 APPROVED。  
- 每次改动后必须跑：`check_physics_logic.py`、`accept_p7_0...`、`accept_p8_1...`、`pytest`。  
- 任何门禁失败：优先检查“动作有效性与动力学对齐”，不要靠调 reward 硬掰。

---

# 附录 A：把门禁阈值做成可配置常量（强制）
> 目的：避免“阈值散落在脚本里”导致难以复现与难以对比版本。  
> 要求：所有门禁阈值统一收敛到一个 YAML 配置里，eval/accept 脚本只读取该配置。

## A.1 新增配置文件（必须）
新增：`configs/phaseB_gates.yaml`

建议结构（字段名可按你项目风格调整，但必须集中在一个文件）：

- `gates.common`  
  - `num_eval_episodes: 20`  
  - `half_epsilon_factor: 0.5`  # half_epsilon = epsilon*factor  
  - `in_band_margin: 0.0`       # 可选：允许略小于 half_epsilon 的统计带内
- `gates.action_effectiveness`  
  - `delta_min_v_ratio_exec: 0.05`
- `gates.stage1_line`  
  - `success_rate_min: 0.95`  
  - `mean_speed_util_in_band_min: 0.70`
- `gates.stage2_square`  
  - `success_rate_min: 0.80`  
  - `delta_speed_util_vs_baseline_min: 0.15`  
  - `straight_dist_threshold: 2.0`  
  - `turn_dist_threshold: 0.5`  
  - `turn_speed_drop_ratio_min: 0.20`  # turn_window mean v_ratio_exec <= (1-0.2)*straight_window mean
  - `corr_v_exec_vs_v_cap_min: 0.30`
- `gates.stage3_gentle_s`  
  - `success_rate_min: 0.70`  
  - `p95_abs_dkappa_exec_max: <set_from_baseline_or_value>`  
- `gates.stage4_hard_s`  
  - `success_rate_min: 0.60`  
  - `mean_speed_util_in_band_min_ratio_vs_square: 0.80`  
  - `max_abs_e_n_max: "half_epsilon"`  # 允许使用表达式或关键字

> 注意：`p95_abs_dkappa_exec_max` 建议用 baseline 自动计算（见附录 B），避免硬编码不合理阈值。

## A.2 代码要求（必须）
- eval/accept 脚本必须支持：`--gates configs/phaseB_gates.yaml`
- 若不提供 `--gates`，默认使用 `configs/phaseB_gates.yaml`
- 所有断言报错必须打印“阈值来源”与“当前值”，例如：
  - `FAIL stage2_square.success_rate: got 0.55 < min 0.80 (from phaseB_gates.yaml)`

---

# 附录 B：Baseline 自动生成与阈值自适应（强烈建议）
> 目的：用 Phase A 专家策略作为 baseline，让 Phase B 的“效率提升/平滑不恶化”有客观参照。

## B.1 新增 Baseline 生成脚本（必须）
新增：`tools/make_phaseB_baseline.py`

功能：
- 使用 Phase A 专家策略（补丁后版本）在 line/square/s_shape 上各跑 N=20 回合
- 产出：`artifacts/baseline/<task>/summary.json` 与标准图表
- 汇总写入：`artifacts/baseline/baseline.json`

baseline.json 必须包含：
- 每个 task 的：`mean_speed_util_in_band`、`mean_progress_per_step`、`p95_abs_dkappa_exec`、`rmse_e_n`、`success_rate`

## B.2 门禁中的 baseline 引用规则（必须）
- `delta_speed_util_vs_baseline_min` 直接用：
  - `mean_speed_util_in_band - baseline.mean_speed_util_in_band >= threshold`
- 平滑上限（p95_abs_dkappa_exec_max）建议设为：
  - `baseline.p95_abs_dkappa_exec * smooth_multiplier`（例如 1.1 或 1.2）
- 脚本必须把 baseline 值写进报告，避免“只看通过/失败不知道差在哪”。

---

# 附录 C：自动生成报告 report.md（强制）
> 目的：你不应该手动打开一堆 png/csv 才知道发生了什么。  
> 要求：每次 eval 都自动生成一份可读报告，包含关键图 + 指标表 + 门禁结果。

## C.1 新增脚本：`tools/make_phaseB_report.py`（必须）
输入：
- `--run_dir artifacts/phaseB/<stage>/<task>/`
- `--baseline artifacts/baseline/baseline.json`（可选，但推荐）
- `--gates configs/phaseB_gates.yaml`

输出：
- `artifacts/phaseB/<stage>/<task>/report.md`

report.md 必须包含：
1) 本次运行元信息：git hash（若可取）、config 路径、seed、时间戳  
2) 指标表（至少）：success_rate、mean_speed_util_in_band、mean_progress_per_step、max_abs_e_n、rmse_e_n、p95_abs_e_n、p95_abs_dkappa_exec、stall_rate  
3) 与 baseline 的差值（若提供 baseline）  
4) 门禁结果清单（PASS/FAIL + 阈值 + 当前值 + 来源）  
5) 嵌入关键图：path.png、e_n.png、v_ratio_exec.png、v_ratio_cap.png、speed_util.png  
6) 失败时的建议定位（引用 Phase B 文档第 8 节的失败模式）

## C.2 eval_policy.py 的强制行为（必须）
- `tools/eval_policy.py` 在完成评估后必须自动调用 `make_phaseB_report.py` 生成 report.md  
- eval_policy.py 必须返回非 0 exit code 如果任一门禁失败（便于 CI/脚本化）

---

# 附录 D：最小化 CI / 一键验收命令（建议）
新增：`tools/run_phaseB_gates.sh`（或 python 等价脚本）

建议流程：
1) `python tools/make_phaseB_baseline.py --out artifacts/baseline --seed 0`
2) 依次跑 Stage1~Stage4 的 eval（用当前策略/或训练 checkpoint）
3) 自动生成 report
4) 任一门禁失败则退出码非 0

> 这能把“训练工程”变成可重复的流水线，而不是靠人盯 dashboard。

---

# 最终强制约束（v3 追加）
- 任何 Stage 门禁失败：必须在 report.md 里给出失败原因与下一步定位建议，不允许只说“训练不好”。  
- 进入更高 Stage 前，必须把当前 Stage 的 report.md 发给用户进行人工复核（只需要看报告里的 3 张关键图与门禁表）。
