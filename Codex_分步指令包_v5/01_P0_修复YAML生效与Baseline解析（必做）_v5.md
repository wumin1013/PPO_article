# P0：修复 YAML 生效 + baseline_s_curve 解析（必做）
> 本阶段只解决“参数不生效”和“baseline 跑不起来”。**不要动动作语义、奖励、状态空间、训练算法**。

> 统一协议提醒：本阶段也必须按 `00_README` 的“统一实验协议”固定 seed，并在日志里打印 **最终生效** 的 dt/MAX_*（来自 YAML）。

## 目标（Scope）
1) configs/*.yaml 里的 `environment.interpolation_period`、`kinematic_constraints.*` 等参数必须真正生效；  
2) `baseline_nnc` 和 `baseline_s_curve` 两种 baseline 模式必须都能运行。

## 允许改动的文件
- `src/environment/cnc_env.py`（只为移除 LEGACY 覆盖、增加日志打印）
- `main.py`（只为修 baseline 解析、增加日志打印/最小 smoke）
- （可选）新增 `tests/smoke_baselines.py` 或 `tools/print_effective_params.py`（用于验证）

## 禁止改动（本阶段不要碰）
- 动作语义/运动学更新逻辑（effective_angle 等）
- reward 设计、done 判据
- observation/state 维度
- PPO 网络/超参数逻辑

## 必须修复点 A：环境覆盖 YAML 的 dt/约束上限
在 `src/environment/cnc_env.py` 中若存在类似：
- `self.interpolation_period = LEGACY_INTERPOLATION_PERIOD`
- `self.MAX_VEL/MAX_ACC/... = LEGACY_KINEMATICS[...]`
则会导致 YAML 参数不生效。

### 要求
- 删除/禁用上述覆盖逻辑；
- YAML 作为唯一来源（若缺字段才回退默认值）；
- Env 初始化时打印“实际生效的 dt/MAX_*”（写到 logger/stdout 均可）。

- **同时打印训练侧的 gamma 与有效视界（前移检查）**：在训练脚本/agent 初始化时打印：
  - `dt = env.interpolation_period`
  - `gamma`（若启用训练，来自配置；若无则明确打印 `gamma=N/A`）
  - `H_steps ≈ 1/(1-gamma)`、`H_time ≈ dt/(1-gamma)`（若 gamma 可用）
  - 目的：避免 P0 改了 dt 但 gamma 没同步，导致后续 P1~P3 验证训练“莫名其妙变差”。

## 必须修复点 B：baseline_s_curve 解析 bug
如果有：`baseline_type = experiment_mode.split("_")[1]`，会把 `baseline_s_curve` 解析成 `"s"`。

### 要求
- 改成：`baseline_type = experiment_mode.replace("baseline_", "")`
  或 `split("_", maxsplit=1)[1]` + 正确处理 `s_curve`；
- 增加最小 smoke：两种 baseline 都可启动并走若干步。

## 自验证/验收标准（你将这样验证）
### 验证 1：YAML 参数真正生效
- 用任意 config（比如 `configs/train_line.yaml` 或 default）启动一次运行（train 或 test 都行）。
- **验收：日志里清晰打印**：dt（interpolation_period）和 MAX_VEL/MAX_ACC/...（或 kinematic_constraints）为 YAML 中配置值，而不是 legacy 常量。

### 验证 2：baseline 模式都能跑
分别运行两次（用相同 config 也行）：
- `experiment.mode = baseline_nnc`
- `experiment.mode = baseline_s_curve`
**验收：** 两次均能启动并走至少 100 step，不报“未知 baseline/解析错误”。

## 交付物（提交时必须包含）
1) 改动文件列表  
2) baseline 解析修复说明  
3) 日志样例（展示生效 dt/MAX_*）  
4) 两条运行命令（baseline_nnc 与 baseline_s_curve）