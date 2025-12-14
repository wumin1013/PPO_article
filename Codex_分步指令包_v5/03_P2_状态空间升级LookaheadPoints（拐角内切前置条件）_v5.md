# P2：状态空间升级（Lookahead Points 强制）
> 前置条件：P1 parity_check 已通过（或已达到可接受误差）。  
> 本阶段只做 observation/state 的升级，让 Agent “看见弯的形状”。**不要引入内切走廊奖励，不要改 P4 的速度目标**。

## 目标（Scope）
- observation 中必须包含未来 N 个预瞄点（N = `environment.lookahead_points`），并提供稳定的局部几何特征，至少 (s_i, d_i)。

## 允许改动的文件
- `src/environment/cnc_env.py`（observation 构造、lookahead 采样/归一化）
- （可选）`src/environment/utils.py` / 新增 helper（避免 cnc_env.py 过臃肿）

## 禁止改动（本阶段不要碰）
- reward/done 判据（包括 corridor、速度目标、停滞惩罚）
- PPO 算法/网络结构（除非 obs 维度改变需要调整输入层的自动适配）

## 任务 1：Lookahead Features 设计（强制）
要求每个预瞄点至少提供：
- `s_i`：沿路径切向的前向距离（归一化到 [0,1]）
- `d_i`：法向偏差（归一化到 [-1,1]，尺度用 half_epsilon）
### 归一化陷阱提醒（必须遵守）
- `d_i` 的尺度往往远大于 `half_epsilon`（尤其 lookahead 取物理距离时）。如果直接 `clip(d_i/half_epsilon)`，很容易长期饱和在 ±1，等于“远处全是同一个信号”，学习会变钝。
- 推荐实现（任选其一，默认用 A）：
  - A) `d_norm = tanh((d_i / half_epsilon) / k)`，建议 `k=2~4`（让中小偏差更可分辨）
  - B) `d_norm = clip(d_i / (half_epsilon * k), -1, 1)`，建议 `k=2~4`
  - C) 对远端点使用更保守缩放（例如随 `s_i` 增大逐渐增大 k），避免远端全饱和
- **自检要求**：打印一段统计（min/mean/max/饱和比例），证明 `|d_norm|>0.95` 的比例不是长期接近 100%。
可选但建议：曲率或曲率变化率（归一化）。

要求：
- 特征在数值尺度上稳定（不爆炸、不塌缩）；
- 明确使用的局部坐标系（推荐 path-tangent frame；若用 body frame 必须说明理由）。

### 坐标系与符号约定（强制统一，给 P3/P4 复用）
- 切向 `t_hat`：取“当前投影到路径上的切向方向”。
- 法向 `n_hat`：取切向左侧为正（左手侧/左法向）。
- 有符号法向误差 `d`（或 `e_n`）：**左正右负**。
- 以后所有关于 `turn_sign`（左转 +1 / 右转 -1）与“内侧/外侧”的定义都要与上述符号一致。


## 任务 2：采样策略（推荐按弧长）
推荐：按弧长采样（例如前方若干固定距离或固定比例）。  
若先按 index：必须解释如何保证不同路径点密度不造成尺度漂移，并在归一化中体现。

## 自验证/验收标准（你将这样验证）
### 验证 1：维度变化正确
- 将 `lookahead_points` 从 1 改到 8，启动 Env reset：
- **验收：** observation 维度随 N 增长，并且打印/断言通过。

### 验证 2：数值健康检查（必须）
写或运行一个小脚本（可放 tools/）：
- 对 3 条路径（line/square/s_shape）各 reset + step 100 次；
- **验收：** obs 无 NaN/Inf；`d_i` 大多在 [-1,1]；`s_i` 在 [0,1] 且随 i 增加。

### 验证 3：训练不崩（建议）
跑 50 episode：
- **验收：** 不出现 loss 爆炸/NaN；策略输出有意义（不全 0、不全 saturate）。

## 交付物（提交时必须包含）
1) observation 结构说明（每个字段含义、归一化范围）  
2) 验证脚本/命令 + 输出样例（健康检查）