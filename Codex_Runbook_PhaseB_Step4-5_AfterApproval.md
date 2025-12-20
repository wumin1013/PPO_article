# Codex Runbook（Phase B）：Step 4–5（需人工复核通过后执行）

> 进入 Phase B 的前提：用户已对 Phase A 的专家策略闭环结果（Square 的 e_n 与 v_ratio_exec 曲线）人工复核通过，并明确回复“通过/APPROVED”。

---

## Step 4（建议）：引入 tangent_relative 动力学模式（对齐 CNC 语义）
目的：让“沿切线进给 + 小角度修正”成为默认可学动力学；保留旧模式用于回归。

实现：
- `environment.dynamics_mode` ∈ {`heading_integrator`, `tangent_relative`}
- `tangent_relative`：
  - `path_angle = _get_path_direction(pos)`
  - `effective_angle = path_angle + omega_exec*dt`
  - 位移按 effective_angle，不累计 heading（heading 仅 debug）

### Step 4.1（必须）：状态连续性（reset 初始化 last_path_angle，防开局脉冲）
在 `src/environment/cnc_env.py::reset()`：
- `self.last_path_angle = _get_path_direction(self.current_position)`（tangent_relative）
- 清零所有差分历史量（prev_action、last_kappa_exec、exit_boost 等）

验收：在 `tools/accept_p8_1_observation_and_corner_phase.py` 加入“开局脉冲测试”：
- reset 后 step([0.0, 0.5]) 不得出现 dkappa_exec/omega_exec 的尖峰。

---

## Step 4.2（强烈建议）：时间效率的硬性拉动（Reward）
在 `src/environment/reward.py`：
- progress 奖励在弯道内不能归零（可减但不为 0）
- 速度利用率奖励 `v_exec/(v_cap+eps)`：仅在带内/未出界 steps 生效（必须 gated）
- 可选：孤立拐角（is_isolated_corner）内侧给予小额 shortcut 奖励（S 型必须禁用）

验收：`accept_p8_1_...` 输出：
- mean_speed_util、progress_per_step
- 弯道 steps 上 progress 与速度奖励不长期为 0

---

## Step 5：正式训练（仅在 Step 4 全部验收通过后）
训练前必须再跑一遍：
```bash
cd PPO_project
python tools/check_physics_logic.py
python tools/accept_p7_0_dynamics_and_scale.py
python tools/accept_p8_1_observation_and_corner_phase.py
python -m pytest -q
```

然后再执行训练入口（按项目既有脚本/命令），并至少提供：
- 训练曲线截图（reward、episode length、success rate）
- eval 下 Square/S-shape 的轨迹图（`path.png`）
- 如果 dashboard 支持：导出关键指标（mean_speed_util、max_abs_e_n、终点成功率）

> 若训练表现异常：优先回到 Phase A/Step 4 的验收脚本定位（先怀疑动力学模式与 reset 初始化，再怀疑 arc dist 与 corner_phase）。
