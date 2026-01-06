# Phase 30：拐角平滑（B2a）（FINAL v1.9）
版本日期：2026-01-07  
依赖：23_P0_L2_Archive 已通过（新基线固化完成）

---

## 0) 目标（一句话）
在允差带内实现**内切、圆弧平滑、切线连续**的拐角过渡，减少降速，且直线段不退化。

---

## 1) 核心目标（对应 01_Objectives）

| 目标 | 说明 | 可测指标 |
|------|------|---------|
| **切线连续（C1）** | 进弯/出弯方向连续，无尖角 | `corner_sharpness_index` ↓ |
| **内切倾向** | 轨迹落在拐角内侧半带 | `inside_ratio ≥ 0.6` |
| **效率优先** | 降速尽量少 | `v_drop_ratio` ↓ |
| **直线不退化** | 贴线与效率不变 | `steps ≤ 1.05× baseline` |

---

## 2) 执行步骤（迭代策略）

### Step 1：快检（10分钟，不通过禁止训练）
```powershell
# 检查 turn_sign 和 e_n 符号是否一致
python tools/b2a1_quick_check.py --config configs/train_square_b2a.yaml --steps 500
```
- `inside_ratio < 0.5` → 检查符号对齐
- `inside_ratio ≥ 0.6` → 继续

### Step 2：走廊/内切 reward 生效（核心）
在 `corner_phase=True` 时启用：

| 奖励项 | 作用 | 参数 |
|--------|------|------|
| `r_barrier` | 带外硬惩罚 | `barrier_weight: 2.0` |
| `r_center` | 内切目标偏好 | `center_weight: 2.0` |
| `r_heading` | 航向一致 | `heading_weight: 2.0` |
| `r_dir_pref` | 方向性偏好 | `dir_pref_weight: 4.0` |

### Step 3：带内平滑微调（可选）
仅在 Step 2 验收后进行：

| 参数 | 作用 | 建议值 |
|------|------|--------|
| `track_deadzone_ratio` | 带内"平坦区"占比 | 0.3 |
| `corner_w_tau_scale` | 拐角航向惩罚衰减 | 0.5 |
| `w_smooth` | 平滑惩罚强度 | 0.1 |
| `w_ang_acc` | 角加速度惩罚 | 0.05 |

---

## 3) 训练配方

**推荐顺序**（每次只改一个旋钮）：
1. 先开 `barrier + heading` → 验证不越带
2. 加 `center + dir_pref` → 验证 inside_ratio ↑
3. 最后加 `smoothness` → 验证 corner_sharpness ↓

**配置示例**：
```yaml
reward_weights:
  track_deadzone_ratio: 0.3
  corner_w_tau_scale: 0.5
  corridor:
    enabled: true
    barrier_weight: 2.0
    center_weight: 2.0
    heading_weight: 2.0
    dir_pref_weight: 4.0
    dir_pref_beta: 4.0
```

---

## 4) 验收标准

### PASS（必须全部满足）
| 指标 | 条件 |
|------|------|
| `success_rate` | ≥ 0.95 |
| `inside_ratio` | ≥ 0.6 |
| `max_abs_contour_error` | ≤ half_epsilon |
| `corner_sharpness_index` | 相对 baseline ↓ 20%+ |
| `steps` | ≤ baseline × 1.05 |

### FAIL
- 任何必须项退化
- 依靠降速获得"伪平滑"

---

## 5) 交付物
- `overlay_corner_zoom.png`（baseline vs 当前 run）
- `summary.json`（含 inside_ratio / corner_sharpness / v_drop）
- `main_table.csv` 新增一行

---

## 6) Stop Rule
- `inside_ratio` 连续 2 次 < 0.5 → 回到 Step 1 修符号
- `smoothness` 变好但 `v_drop` 变差 → 降低平滑权重
- `steps` 暴增（> 1.2× baseline）→ 检查 reward 权重平衡

---

## 7) 论文映射
- Fig：`fig:curve_square`（拐角对比）
- Table：`tab:results`（smoothness / v_drop 列）
