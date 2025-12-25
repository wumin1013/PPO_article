# 指令文档 P1：平滑拐角（把“尖角”变成圆弧/样条风格）

**前置**：P0 已通过（训练能稳定完成全程）。  
**目标**：拐角处形成平滑轨迹（提前入弯、连续换向），并尽量少降速；仍然不依赖硬规则限速。

---

## Step 1 用 lookahead 方向替代“线段切向”（消灭参考方向不连续）
### 1.1 修改文件
- `PPO_project/src/environment/cnc_env.py`
- 修改/新增函数：用于计算 `theta_ref` 的“lookahead 视线方向”

### 1.2 实现要点（pure pursuit / LOS）
1. 先把当前位置投影到 progress path 上得到弧长 `s_now`
2. 取前方弧长 `s_target = s_now + L`
3. 用 `p_target = point_at_s(s_target)`（沿 Pm 插值或利用 `_scan_for_next_turn`/缓存）
4. `theta_ref = atan2(p_target - pos)`

L 的建议：
- 与速度相关：`L = clip(L0 + kL * v_exec, Lmin, Lmax)`
- 初始可以固定：`L = 2~6 * lookahead_spacing`

### 1.3 参考伪代码
```python
def _theta_ref_lookahead(self, pos, v_exec):
    s_now = self._project_onto_progress_path(pos)[2]
    L = np.clip(self._L0 + self._kL * v_exec, self._Lmin, self._Lmax)
    p_tgt = self._point_at_arc_length(s_now + L)   # 需要你实现或用已有 cache
    d = p_tgt - pos
    return math.atan2(d[1], d[0])
```

然后在 `calculate_new_position()` 里用 `theta_ref = _theta_ref_lookahead(...)`。

**验收（局部）**：在拐角前，`theta_ref` 应该逐渐转向下一个边，不再“瞬间跳 90°”。

---

## Step 2 在拐角相位加入“角加速度惩罚”（逼出提前转向的圆弧）
### 2.1 原理
尖角策略的本质：把 90° 的转向集中在很短的时间内完成。  
要让它变圆，就要让“把转向集中起来”变得昂贵。

`angular_jerk` 不够，因为保持恒定角加速度时 jerk 可以很小。  
所以需要惩罚 `|angular_acc|`（或 `|dκ/dt|`）。

### 2.2 修改位置
- `reward.py -> RewardCalculator.calculate_reward()`

### 2.3 推荐方式：仅在“接近拐角”时增强惩罚
用 `next_angle` 与 `distance_to_next_turn` 构造一个权重：
- `turn_strength = clip(|next_angle| / theta0, 0, 1)`
- `near_strength = exp(-distance_to_turn / d0)`
- `w_corner = turn_strength * near_strength`

然后：
- `r_angacc = - w_angacc * w_corner * (|angular_acc| / MAX_ANG_ACC)^2`

---

## Step 3 在直线段强化“严格贴线”（防止出弯带偏置）
同样利用 `next_angle` 判断当前是否直线段：
- 如果 `|next_angle| < theta_straight` 且 `distance_to_turn > d_straight`：
  - 把 `w_e`（横向误差惩罚）放大 2~5 倍

---

## Step 4 验收标准（P1 结束条件）
使用固定测试路径（正方形）评估 100 回合（只用 deterministic policy 或固定 seed）：

1. **拐角处轨迹不再出现“尖点”**：  
   - 视觉检查：轨迹在拐角附近呈圆弧过渡
   - 定量：拐角窗口内 `max(|angular_acc|)` 明显降低（相较 P0 降 30%+）

2. **速度保持**：  
   - 拐角窗口内平均速度不低于 P0 的 80%（允许适度降速）

3. **出弯回线**：  
   - 过弯后 N 步内（建议 N=50~150，取决于 dt）`|e_n|` 回到 `0.1*half_epsilon` 以内

---

# P1 的产出
P1 会让参考方向连续，并把“集中转向”变贵，从而自然产生圆弧式过弯。  
下一步 P2 会进一步把“出弯回线”写成显式学习目标。
