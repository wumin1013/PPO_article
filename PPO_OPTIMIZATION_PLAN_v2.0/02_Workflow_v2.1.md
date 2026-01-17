# v2.1 执行工作流
版本日期：2026-01-17

---

## 0) 概述

本文档定义 v2.1（曲率感知路线）的完整执行流程。

---

## 1) Phase 执行顺序

```
Phase 30 (曲率感知基线)
    │
    ├─ 成功 → Phase 32 (多路径验证)
    │                │
    │                ├─ 成功 → Phase 33 (消融分析)
    │                │                │
    │                │                └─ Phase 40 (论文输出)
    │                │
    │                └─ 失败 → 调试并重试
    │
    └─ 失败 → 尝试轻量曲率惩罚
              │
              ├─ 成功 → Phase 32
              └─ 失败 → 调整论文叙事（见 99_Fallback）
```

---

## 2) Phase 30：曲率感知基线

### 2.1 前置条件
- P0_L2 基线可用
- Phase 21 状态空间已实现

### 2.2 执行步骤

```powershell
# Step 1: 代码修改
# 修改 cnc_env.py 添加曲率观测

# Step 2: 创建配置
# 创建 configs/train_square_curvature_v21.yaml

# Step 3: 训练
conda activate PPO
cd PPO_project
python main.py --config configs/train_square_curvature_v21.yaml --mode train

# Step 4: 评估
python tools/acceptance_suite.py `
    --phase p0_eval `
    --config configs/train_square_curvature_v21.yaml `
    --model saved_models/curvature_v21/*/checkpoints/best_model.pth `
    --episodes 50 `
    --out out/curvature_v21_eval `
    --deterministic

# Step 5: 涌现分析
python tools/b2a1_corner_evidence.py `
    --candidate artifacts/curvature_v21 `
    --baseline artifacts/P0_L2/P0_12d_gold_20260114_174552
```

### 2.3 验收标准

| 指标 | 条件 |
|------|------|
| success_rate | ≥ 0.95 |
| max_abs_e_n | ≤ 0.75 |
| corner_peak_omega | < 0.9 × MAX_ANG_VEL |

---

## 3) Phase 32：多路径验证

### 3.1 前置条件
- Phase 30 通过

### 3.2 测试路径

| 路径 | 配置 | 说明 |
|------|------|------|
| square | scale=10 | 基线（已验证） |
| s_shape | - | S形曲线 |
| sharp_angle | angle=45° | 锐角转弯 |
| line | - | 直线（消融） |

### 3.3 验收标准

所有路径 `success_rate ≥ 0.90`

---

## 4) Phase 33：消融分析

### 4.1 对比实验

| 配置 | 状态维度 | 预期 |
|------|----------|------|
| 无曲率状态 | 12 | 尖角（对照组） |
| 有曲率状态 | 14 | 平滑（实验组） |

### 4.2 输出

- 消融对比图表
- 统计显著性分析

---

## 5) Phase 40：论文输出

### 5.1 生成物

| 内容 | 格式 |
|------|------|
| 轨迹对比图 | PDF/PNG |
| 指标表格 | LaTeX |
| 训练曲线 | PNG |
| 代码存档 | ZIP |

### 5.2 论文章节映射

| 章节 | 对应 Phase |
|------|-----------|
| 方法 | Phase 30 设计 |
| 实验 | Phase 32 验证 |
| 分析 | Phase 33 消融 |

---

## 6) 时间估算

| Phase | 预估时间 |
|-------|----------|
| Phase 30 | 4-6 小时 |
| Phase 32 | 2-3 小时 |
| Phase 33 | 2-3 小时 |
| Phase 40 | 4-6 小时 |
| **总计** | **12-18 小时** |
