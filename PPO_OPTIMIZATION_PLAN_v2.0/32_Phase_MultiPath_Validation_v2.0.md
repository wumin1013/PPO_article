# Phase 32：多路径验证（Multi-Path Validation）
版本日期：2026-01-17  
依赖：Phase 30 或 31 涌现平滑行为

---

## 0) 目标（一句话）

**验证极简策略的泛化能力**：在 line、s_shape、sharp_angle 等不同轨迹上测试，确保策略不是"过拟合"到 square。

---

## 1) 测试轨迹族

| 轨迹类型 | 特点 | 挑战 |
|----------|------|------|
| `square` | 4 个 90° 直角拐角 | 训练分布（baseline） |
| `line` | 纯直线 | 无拐角，测试直线性能是否退化 |
| `s_shape` | 2 个反向弯道 | 复合弯道，测试连续转向 |
| `sharp_angle` | 1 个锐角（< 90°） | 极端拐角 |

---

## 2) 验证策略

### 2.1 测试方式

使用 **训练好的模型**（来自 Phase 30 或 31），在不同路径上做确定性 rollout。

**不做额外训练**，这是泛化性测试。

### 2.2 评估脚本

```powershell
# 假设使用 Phase 30 的模型
$model = "artifacts/minimal_v1/checkpoint.pth"
$out_base = "artifacts/minimal_v1/multipath"

# 各路径评估
foreach ($path_type in @("line", "s_shape", "square")) {
    $config = "configs/eval_$path_type.yaml"
    $out = "$out_base/$path_type"
    
    python tools/rollout_trace.py --model $model --config $config --out $out --episodes 10
    python tools/a3_aggregate_runs.py --run_dir $out
}
```

---

## 3) 配置文件

### 3.1 `configs/eval_line.yaml`

```yaml
# 继承 train_square_minimal.yaml 的大部分配置
# 只修改 path 部分

path:
  type: line
  closed: false
  scale: 40.0       # 40mm 直线
  num_points: 200

experiment:
  mode: eval        # 评估模式
  name: eval_line
```

### 3.2 `configs/eval_s_shape.yaml`

```yaml
path:
  type: s_shape
  closed: false
  scale: 10.0
  num_points: 200

experiment:
  mode: eval
  name: eval_s_shape
```

### 3.3 `configs/eval_square.yaml`

```yaml
path:
  type: square
  closed: true
  scale: 10.0
  num_points: 200

experiment:
  mode: eval
  name: eval_square
```

---

## 4) 验收标准

### 4.1 必须项（全路径）

| 指标 | 条件 |
|------|------|
| success_rate | ≥ 0.90（略放宽，因为是泛化测试） |
| max_abs_e_n | ≤ ε/2 |

### 4.2 期望（泛化性）

| 路径 | 期望行为 |
|------|----------|
| line | 高速贴线，无震荡 |
| s_shape | 两个弯道平滑过渡，无尖角 |
| square | 与训练时性能一致 |

### 4.3 对比表

| 指标 | square | line | s_shape | 说明 |
|------|--------|------|---------|------|
| success_rate | baseline | - | - | 训练分布 |
| mean_velocity | V_sq | ≥ V_sq | ≈ V_sq | 直线应更快 |
| corner_peak_omega | ω_sq | N/A | ≤ ω_sq | s_shape 弯道更缓 |
| steps | S_sq | ≤ 0.5×S_sq | ≈ S_sq | 直线应更短 |

---

## 5) 执行步骤

### Step 1：创建评估配置
复制 `train_square_minimal.yaml`，修改 path 部分。

### Step 2：运行评估
```powershell
$model = "artifacts/minimal_v1/checkpoint.pth"

# Line
python tools/rollout_trace.py --model $model --config configs/eval_line.yaml --out artifacts/minimal_v1/multipath/line --episodes 10

# S-shape
python tools/rollout_trace.py --model $model --config configs/eval_s_shape.yaml --out artifacts/minimal_v1/multipath/s_shape --episodes 10

# Square (sanity check)
python tools/rollout_trace.py --model $model --config configs/eval_square.yaml --out artifacts/minimal_v1/multipath/square --episodes 10
```

### Step 3：聚合指标
```powershell
python tools/a3_aggregate_runs.py --run_dir artifacts/minimal_v1/multipath --out artifacts/minimal_v1/multipath/aggregate
```

### Step 4：生成对比图
```powershell
python scripts/paper_plotter.py --multipath artifacts/minimal_v1/multipath --out paper_assets/figures
```

---

## 6) 结果处理

### 6.1 如果全部通过

```
✅ 泛化成功
→ 进入 Phase 33（消融实验）
→ 论文可声称：策略具有跨轨迹泛化能力
```

### 6.2 如果部分失败

```
🔄 分析失败模式
│
├─ line 失败：策略可能"学坏了"，需检查直线段奖励是否足够
├─ s_shape 失败：复合弯道处理能力不足，可考虑：
│   └─ 在多路径上联合训练（curriculum learning）
└─ square 失败：退化，需回滚检查
```

### 6.3 备选：多路径联合训练

如果泛化不足，可以考虑：

```yaml
# configs/train_multipath.yaml
path:
  type: random      # 每 episode 随机选择路径类型
  path_types: ["square", "line", "s_shape"]
  ...
```

但这会增加训练复杂度，优先尝试单路径泛化。

---

## 7) 交付物

| 文件 | 说明 |
|------|------|
| `configs/eval_*.yaml` | 各路径评估配置 |
| `artifacts/minimal_v1/multipath/` | 各路径评估结果 |
| `artifacts/minimal_v1/multipath/aggregate/summary.json` | 聚合指标 |
| `paper_assets/figures/fig_multipath.pdf` | 多路径对比图 |

---

## 8) 论文映射

### 8.1 图表

| 论文图表 | 来源 |
|----------|------|
| Fig. 3: 轨迹对比（square） | `multipath/square/overlay.png` |
| Fig. 4: 轨迹对比（line） | `multipath/line/overlay.png` |
| Fig. 5: 轨迹对比（s_shape） | `multipath/s_shape/overlay.png` |
| Tab. 1: 多路径性能汇总 | `aggregate/summary.json` |

### 8.2 叙事

> "为验证策略的泛化能力，我们在直线、S形曲线、直角方波等不同轨迹族上进行了测试。实验结果表明，仅在 square 路径上训练的策略，能够成功泛化到其他几何形状，成功率均超过 90%。这证明了端到端学习的策略具有良好的泛化性，而非简单记忆训练轨迹。"

---

## 9) 时间估算

| 步骤 | 时间 |
|------|------|
| 创建配置 | 15 分钟 |
| 运行评估 | 30 分钟 |
| 分析结果 | 30 分钟 |
| 生成图表 | 15 分钟 |
| **总计** | **1.5 小时** |
