# Codex Runbook（Phase A）：Step 1–3 环境闭环修复 + 专家策略验证（完成后必须停止等待人工复核）

> 目的：在任何正式训练（Step 5）之前，先把环境做成“物理闭环健康”的系统。  
> **Phase A 完成后必须停下**，把产物交给人类复核通过，才允许进入 Phase B。

---

## DoD（Phase A）
必须全部通过：
```bash
cd PPO_project
python tools/check_physics_logic.py
python tools/accept_p7_0_dynamics_and_scale.py
python tools/accept_p8_1_observation_and_corner_phase.py
python -m pytest -q
```

并且必须产出（Square 路径）：
- `artifacts/phaseA/square_v_ratio_exec.png`
- `artifacts/phaseA/square_e_n.png`
- `artifacts/phaseA/summary.json`
- `artifacts/phaseA/trace.csv`
- `artifacts/phaseA/path.png`

---

## Step 1：观测语义一致性（state 绑定，强制）
文件：`src/environment/cnc_env.py`（通常在 `apply_action()` 或 state 更新处）

- `state[3] = scan.dist_to_turn`
- `state[5] = scan.turn_angle`
- 禁止用旧的 next_angle 占用 state[5]（可放 info/debug）

验收：在 `tools/accept_p8_1_observation_and_corner_phase.py` 里做严格断言（误差 < 1e-6）。

---

## Step 2：归一化链路（只允许一个责任方）
二选一，且必须贯彻一致：

**方案 A（推荐）**：Env 负责归一化，训练端禁用 `StateNormalizer`  
- `Env.step()` 返回 normalized obs  
- `main.py` 新增 `training.use_obs_normalizer=false` 时不创建/不调用 `StateNormalizer`

**方案 B**：训练端归一化，Env 返回 raw obs  
- `Env.step()` 返回 raw state  
- 训练端启用 `StateNormalizer`

同时必须修复 lookahead 归一化：按三列尺度（s、d、kappa_rate）归一化，避免 clip 饱和。

验收：`accept_p8_1_...` 检查 lookahead 不全饱和、弯道附近 kappa_rate 非零。

---

## Step 3：corner_phase 隔离（走廊开关必须管住全局状态）
文件：`src/environment/cnc_env.py::_compute_corridor_status()`

- `enable_corridor == False`：
  - 禁止更新 `self.in_corner_phase`
  - 强制 `corner_phase=False`
- `enable_corridor == True`：
  - corner_phase 进入/退出只用 scan 的 `dist_to_turn/turn_angle`
  - 禁止用 `alpha/LOS` 决定 corner_phase（可仅 debug）

验收：直线+横向偏置情况下 corner_phase 不得误触发；v_cap 不受 alpha 变化影响。

---

## Step 3.1（必须）：dist_to_turn 必须是弧长（Arc Length），禁止欧式距离
文件：`src/environment/cnc_env.py::_compute_p4_pre_step_status()`

- `dist_to_turn` 必须来自 `_scan_for_next_turn(s_now)`（Arc Length）
- 禁止 `norm(corner_xy - pos_xy)` 作为 braking/cap 的距离来源

验收：`accept_p8_1_...` 做 Arc vs Euclid 分离测试：
- dist_arc 与 dist_euclid 至少相差 5%
- dist_arc 与独立弧长计算一致（误差 < 1e-3）

---

## Step 3.2（强制门禁）：专家策略验证（Square）
在 `tools/accept_p8_1_observation_and_corner_phase.py` 中实现/启用专家策略（PD + speed_target）并输出：

- `artifacts/phaseA/square_v_ratio_exec.png`：v_ratio_exec(t)
- `artifacts/phaseA/square_e_n.png`：e_n(t)（含 ±half_epsilon 参考线）
- `artifacts/phaseA/summary.json`：max_abs_e_n、rmse_e_n、mean_v_ratio_exec、是否到终点、是否 stall
- `artifacts/phaseA/trace.csv`：每步的 s_now、dist_to_turn、turn_angle、v_ratio_cap、v_ratio_exec、e_n、omega_exec（如有）、dkappa_exec（如有）

> 要点：这是“环境物理闭环证明”，不是训练效果。

---

## Step 3.3（新增，建议强制）边界应力测试：极小 epsilon + 极大 MAX_VEL（验证刹车包络/失速惩罚是否“硬核”）
> 目的：用“几乎不可能的允差 + 极端速度上限”去敲打系统，观察系统是否会**自动压速**或**触发 stall（失速）惩罚**。  
> 如果在这种极端条件下系统还能保持数值稳定、且逻辑能按预期自保，说明 **Braking Envelope / v_cap / stall** 机制不是摆设。

### 3.3.1 执行方式（不要直接改 default.yaml，避免污染回归）
1) 新增一个仅用于测试的配置文件：`configs/stress_tiny_eps_high_vel.yaml`（从 `configs/default.yaml` 拷贝即可）。  
2) 覆写以下字段（其余保持不变）：
   - `path.epsilon: 0.1`  
     - 说明：若你项目单位是 **mm**（default epsilon 常见为 1.5），则 0.1 是“极小允差”；  
     - 若你项目单位是 **m**，请改为 `0.0001`。
   - `kinematic_constraints.MAX_VEL: 100000.0`（或至少比默认值大 100×）
3) 运行专家策略验证脚本时，必须支持“指定 config 与输出目录”：  
   - 若脚本已有参数：直接用  
     ```bash
     python tools/accept_p8_1_observation_and_corner_phase.py --config configs/stress_tiny_eps_high_vel.yaml --out artifacts/phaseA_stress
     ```  
   - 若脚本目前不支持参数：请为该脚本新增 `--config` 与 `--out` 参数（默认行为不变），再运行上面命令。

### 3.3.2 预期现象（通过标准）
通过不要求“成功到终点”，但要求“系统硬核自保且数值稳定”：

必须满足：
- **数值稳定**：trace 中不得出现 NaN/Inf；位置/速度不爆炸。
- **自保机制生效**（二选一即可）：
  1) 专家策略会把 `v_ratio_exec` 自动压到很低（例如 `mean(v_ratio_exec) < 0.05`，且 `speed_target_ratio`/`v_ratio_cap` 同步偏低）；  
  2) 或者系统会在无法满足允差时**明确触发 stall/终止**（summary.json 标记为 stall/failed），而不是“默默发散画大弧线”。

判失败的典型反例：
- v_ratio_exec 仍很高但 e_n 长期越界（说明 braking envelope / v_cap 没兜住）
- 不 stall 也不压速，轨迹直接飞出（说明极限保护缺失）
- 出现 NaN/Inf（说明动力学/归一化/状态机仍有 bug）

### 3.3.3 产物要求
必须在 `artifacts/phaseA_stress/` 下生成：
- `square_v_ratio_exec.png`
- `square_e_n.png`
- `summary.json`
- `trace.csv`
- `path.png`

并在终端打印一句总结（必须包含）：
- `epsilon`、`MAX_VEL`、`mean_v_ratio_exec`、`max_abs_e_n`、`stall_triggered`

### 3.3.4 回归保护（必须）
边界应力测试跑完后，必须再用**默认配置**跑一遍 Phase A 的 DoD 命令，确保没有污染正常场景。

---

## Phase A 结束与停机要求（必须执行）
1) 运行本页 DoD 的全部命令，并确保全绿。  
2) 确保 `artifacts/phaseA/` 与（若执行了应力测试）`artifacts/phaseA_stress/` 下的文件存在且可打开。  
3) **停止一切后续修改**，向用户展示 Phase A 产物（尤其两张曲线图：`square_e_n.png` 与 `square_v_ratio_exec.png`）。  
4) 等待用户明确回复“通过/APPROVED”后，才允许进入 Phase B。
