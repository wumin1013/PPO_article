# 04_P2.5 修复 Pl/Pr 偏移线生成（无交叉 / 无自交 / 左右语义一致）_v6
> 目标：让 square / S / line 的 Pl/Pr 与 corridor 四边形稳定、可验收、可调试。  
> 原则：**复刻你原始代码的“平行线 + 延长线求交(miter)”思路**，但补上工程必需的保护（退化点清理、miter_limit、凹角处理、闭环退化 quad 修复）。

---

## Scope（只做这些）
1) Pl/Pr 偏移线生成鲁棒化（保持“原始方法等价”）  
2) corridor 四边形构造修复：**闭环不产生退化片段、不多 append**  
3) 新增一键自检脚本：检测自交/翻转/退化 quad，并输出可视化

---

## 允许改动的文件
- `src/utils/geometry.py`（或你项目里几何模块所在文件）：Pl/Pr 生成、交叉检测 helpers
- `src/environment/cnc_env.py`：仅限 `_create_polygons()` 或 corridor 构造相关函数
- 新增：`tools/offset_debug.py`（或同名脚本）

## 禁止改动（本阶段不要碰）
- PPO 训练逻辑、奖励、done、动作语义、状态维度（P2.5 只做几何与走廊构造的稳定性）

---

## Task 1：统一闭环表示，清理退化点（**必须**）
### 1.1 退化点清理（open/closed 都要做）
对输入中心线 `Pm` 进行预处理：
- 移除连续重复点：`||Pm[i]-Pm[i-1]|| < eps_len`
- 移除极短段导致的抖动点（可选但强烈建议）：若三点形成“极短段 + 近共线”，合并中点  
建议：`eps_len = 1e-6 ~ 1e-4`（按坐标量纲调）

### 1.2 闭环点表示（关键）
若 `closed=True` 且 `Pm[0] == Pm[-1]`（常见闭合写法）：
- 内部使用 `Pm_core = Pm[:-1]` 作为闭环顶点序列（长度 m）
- 后续所有计算（切向、法向、Pl/Pr、polygon）都按 `Pm_core` 长度 m 做
- **禁止**用 `Pl[-1],Pl[0],Pr[0],Pr[-1]` 额外 append 形成 closing quad（会退化/污染索引）

> 目的：彻底消灭闭环走廊的退化四边形与 segment_idx 抖动源。

---

## Task 2：按“原始 miter 思路”生成 Pl/Pr，但加工程护栏
你原始方法：两段各自平移到左/右，再用“延长线求交点”作为 join。  
本任务允许继续该思路，但必须加三道保险：**miter_limit / 凹角保护 / 平行退化处理**。

### 2.1 统一左右语义（强制约定）
给定段方向单位切向 `t=(tx,ty)`：
- 左法向：`nL = (-ty, tx)`
- 右法向：`nR = ( ty,-tx)`

所有地方必须一致，禁止“某处左某处右”或符号偷偷反掉。

### 2.2 join 的计算（推荐实现路径）
对每个顶点 `p_i`（open：中间点；closed：所有点）：
1) `t_prev = unit(p_i - p_{i-1})`，`t_next = unit(p_{i+1} - p_i)`（closed 用环绕索引）
2) `n_prev, n_next` 分别取左（或右）法向
3) miter 方向：`m = unit(n_prev + n_next)`  
   - 若 `||n_prev + n_next|| < eps`（180° 或近似反向），用 `m = n_prev`
4) miter 长度：`miter_len = d / dot(m, n_prev)`（d = corridor_half_width）

### 2.3 miter_limit（必须）
若 `abs(dot(m, n_prev)) < 1e-6` 或 `abs(miter_len) > MITER_LIMIT*d`：
- clamp：`miter_len = sign(miter_len) * MITER_LIMIT * d`
建议：`MITER_LIMIT = 4.0`（先用 4，必要时降到 2）

### 2.4 凹角保护（必须）
用 `cross = cross2(t_prev, t_next)` 判断转向（2D 叉乘标量）：
- 对“左边界”：若该点在左侧是凹角（常见判定：`cross < 0`），禁止使用 miter 交点（容易翻折）
  - 改用 bevel：`Pl_i = p_i + d * n_prev`（或 `d*n_next`，需保持一致）
- 对“右边界”：判定符号相反（常见：`cross > 0` 视为右侧凹角）

> 目标：避免 join 交点跑到错误一侧，导致局部翻折/quad 自交。

### 2.5 端点处理（open path）
open 的首尾点不做 miter（没有两段），直接：
- `Pl_0 = p_0 + d*n0`，`Pr_0 = p_0 + d*n0_right`
- `Pl_end = p_end + d*n_end`，`Pr_end = p_end + d*n_end_right`

---

## Task 3：走廊 polygon 构造修复（**闭环不多 append**）
在 `cnc_env.py` 的 `_create_polygons()`（或同等函数）里：

- open：`segments = n-1`  
  `quad_i = [Pl[i], Pl[i+1], Pr[i+1], Pr[i]]`
- closed：使用 `Pm_core` 长度 m（注意 Pl/Pr 也应是 m）
  对每个 i：`j = (i+1) % m`  
  `quad_i = [Pl[i], Pl[j], Pr[j], Pr[i]]`
  **不要再额外 append** `[Pl[-1],Pl[0],Pr[0],Pr[-1]]`

---

## Task 4：新增一键自检脚本 `tools/offset_debug.py`（必须可跑）
对路径：`line / square / S` 输出并判定：

### 4.1 必做检查
A) `Pl` 自交计数 = 0  
B) `Pr` 自交计数 = 0  
C) corridor quads 自交/退化计数 = 0（重点）  
D) 左右语义一致性：对每段切向 t，检查  
   - `dot(Pl[i]-Pm[i], nL(t)) > 0` 且 `dot(Pr[i]-Pm[i], nR(t)) > 0`  
   （允许少量数值误差）

### 4.2 输出可视化（必须）
保存一张图（png）叠加：
- Pm（黑）、Pl（绿）、Pr（蓝）
- 标注任何失败的段 index、失败类型（self-intersect / quad-intersect / flipped）

---

## 验收标准（全部满足才算过）
- line/square/S：A=0、B=0、C=0  
- square：闭环 quad 数量= m（不多不少），并且没有退化 quad  
- 任何路径：左右语义不翻转（D 全部通过）

---

## 交付物
1) 修复后的 `geometry.py` / `_create_polygons()` 变更  
2) `tools/offset_debug.py` + 示例输出图（3 条路径各 1 张）  
3) 一段 README：如何运行自检、如何解读失败
