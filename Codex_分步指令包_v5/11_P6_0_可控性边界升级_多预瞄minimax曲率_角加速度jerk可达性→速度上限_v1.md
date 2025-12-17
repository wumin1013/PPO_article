# 11_P6.0 可控性边界升级：多预瞄 minimax 曲率 + 角加速度/角jerk 可达性 → 速度上限_v1
> 前置条件：P5.2 已通过（幅度自由 + LOS turn 指标已对齐）。  
> 本阶段目标：把“弯前降速”从 reward/经验规则升级为**真正的可控性边界（control-feasibility boundary）**：  
> 不仅满足 `|ω| <= MAX_ANG_VEL`，还要满足“在到达急弯前这段距离里，系统**来得及**把 ω ramp 到所需值”，因此引入 `|ω̇| <= MAX_ANG_ACC` 与 `|ω̈| <= MAX_ANG_JERK` 的可达性约束，映射成速度上限。  
> 同时用**多预瞄 minimax** 处理“前方曲率峰值”，让减速更提前、更平顺、更像顶级数控的前瞻刹车。

---

## 目标（Scope）
1) 多预瞄：对 lookahead 的多个点 i 计算 LOS 几何 `alpha_i, L_i, kappa_i`
2) 速度上限由三条边界取最小（每个点都算一次，再取全局 min）：
   - 角速度边界：`v <= ω_max / κ`
   - 角加速度可达性边界：`v <= sqrt( ω̇_max * s / κ )`
   - 角jerk 可达性边界：`v <= cbrt( 0.5 * ω̈_max * s^2 / κ )`
3) v_ratio_cap 使用上述边界的 minimax：对所有 lookahead 点取最小 cap
4) 输出 debug：证明“在急弯前 cap 会提前、平滑地下压速度”，并减少迟刹/冲出走廊

---

## 允许改动的文件
- `src/environment/cnc_env.py`
- 新增：`tools/p6_feasibility_cap_report.py`（必做）

## 禁止改动（本阶段不要碰）
- corridor reward 形状（P5.2 已锁定）
- PPO 算法主体
- done 判据/终点逻辑（除非发现 bug；否则别动）

---

## 任务 1：多预瞄 LOS κ_i（必须）
对 lookahead 点 i（建议 i=1..K，K=5~8）：
- 获取世界坐标 `P_i`
- `r_i = P_i - pos`
- `L_i = ||r_i||`
- `theta_los_i = atan2(r_i.y, r_i.x)`
- `alpha_i = wrap(theta_los_i - heading)`
- `kappa_i = |alpha_i| / (L_i + eps)`

同时定义 `s_i`：到该点的**沿路径距离**（优先用你缓存的 lookahead longitudinal；至少要与 L_i 同尺度）
- 若没有精确弧长，就用 `s_i ≈ max(L_i, lookahead_spacing*i)`，但必须在 debug 输出中标注你选用的定义。

---

## 任务 2：三条可控性边界 → v_cap_i（必须）
对每个 i：
1) 角速度边界：
- `v_cap_w = ω_max / (kappa_i + eps)`

2) 角加速度可达性边界（推导假设：用距离 s_i 作为 ramp 空间，时间 T≈s_i/v）：
- 需要在到达该点前把 ω ramp 到 `ω_req = v * κ`  
- 最大可实现 `Δω <= ω̇_max * T = ω̇_max * s_i / v`  
- 令 `ω_req <= Δω` 得：
- `v_cap_wdot = sqrt( ω̇_max * s_i / (kappa_i + eps) )`

3) 角jerk 可达性边界（假设从 ω̇=0 起步、jerk 限制下 Δω≈0.5*ω̈_max*T^2）：
- `Δω <= 0.5 * ω̈_max * (s_i / v)^2`
- `v * κ <= 0.5 * ω̈_max * s_i^2 / v^2`
- 得：
- `v_cap_wddot = cbrt( 0.5 * ω̈_max * s_i^2 / (kappa_i + eps) )`

最后：
- `v_cap_i = min(v_cap_w, v_cap_wdot, v_cap_wddot, MAX_VEL)`

全局：
- `v_cap = min_i(v_cap_i)`
- `v_ratio_cap = clip(v_cap / MAX_VEL, 0, 1)`

---

## 任务 3：与 KCM 的衔接（必须）
- 这一层 cap 仍然是“速度上限”——你在 P5.0 已保证 KCM 输入是物理速度，因此直接：
  - `max_vel_cap_phys = v_cap`
  - 或保持 ratio 的形式：`max_vel_cap_phys = v_ratio_cap * MAX_VEL`
- 要求：info 中同时输出三种边界的最小值，用于诊断是哪条边界在主导：
  - `v_cap_w_min, v_cap_wdot_min, v_cap_wddot_min, v_cap_final`

---

## 任务 4：新增 tools/p6_feasibility_cap_report.py（必做）
目的：用数据证明“弯前降速像真正的可控性边界”。

建议输出两类报告（都必须有）：
1) **时序报告（square 的一个 episode）**  
   - 打印/保存序列：`step, progress, v_ratio_exec, v_ratio_cap, v_cap_w_min, v_cap_wdot_min, v_cap_wddot_min, alpha_max_ahead, kappa_max_ahead`
   - 验收：在进入急弯前若干步，`v_ratio_cap` 已开始下降（不是到拐点才突然掉）

2) **越界改善对比（A/B）**  
   - A：只用 P5.1 的角速度边界（关掉 wdot/wddot）  
   - B：开启 wdot+wddot  
   - quick_eval：square E=20  
   - 输出：`oob_rate, success_rate, steps_mean, v_mean`  
   - 验收：B 的 oob_rate 明显下降或在相同 oob 下 v_mean 上升；且行为更平顺（角jerk 统计下降）

---

## 自验证/验收标准（你将这样验证）
1) **边界提前生效（硬指标）**  
   - 运行 `tools/p6_feasibility_cap_report.py` 时序报告  
   - **验收：** 急弯前 `v_ratio_cap` 提前下降，且不出现“一步掉到底”的毛刺（应更平滑）

2) **稳定性不下降（硬指标）**  
   - quick_eval（line/square/s_shape 各 E=20）  
   - **验收：** success_rate 不下降；square 的 oob_rate 不上升

3) **更像顶级数控的“又快又顺”（期望指标）**  
   - 统计：`angular_jerk_mean`、`kcm_intervention_mean`  
   - **验收：** 在保持 success 下，这两项下降或不升，同时 v_mean/steps_mean 改善

---

## 交付物（提交时必须包含）
1) 改动文件列表 + 三条边界公式说明（含你选用的 s_i 定义）  
2) `tools/p6_feasibility_cap_report.py`  
3) A/B 对比输出样例（至少 square E=20）  
4) quick_eval 输出样例（line/square/s_shape）
