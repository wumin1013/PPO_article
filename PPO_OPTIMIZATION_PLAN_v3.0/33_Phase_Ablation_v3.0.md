# Phase 33：消融分析（v3.0）
版本日期：2026-01-18  
依赖：Phase 30 完成

---

## 目标

通过消融实验验证前瞻向量和管道奖励各自的贡献。

---

## 实验设计

| 实验 | 前瞻 Lookahead | 管道 Tube | 配置 |
|------|----------------|-----------|------|
| A (Baseline) | ❌ | ❌ | `ablation_none.yaml` |
| B (Lookahead Only) | ✅ | ❌ | `ablation_lookahead.yaml` |
| C (Tube Only) | ❌ | ✅ | `ablation_tube.yaml` |
| D (Full) | ✅ | ✅ | `train_square_v30.yaml` |

---

## 配置差异

### ablation_none.yaml
```yaml
reward_weights:
  lookahead:
    enabled: false
  tube:
    enabled: false
```

### ablation_lookahead.yaml
```yaml
reward_weights:
  lookahead:
    enabled: true
    distance: 4.5
  tube:
    enabled: false
```

### ablation_tube.yaml
```yaml
reward_weights:
  lookahead:
    enabled: false
  tube:
    enabled: true
    ratio: 0.5
```

---

## 评估指标

| 指标 | 预期结果 |
|------|----------|
| corner_peak_ω | D < B < C < A |
| inside_ratio | D > B > C > A |
| success_rate | D ≈ B > C ≈ A |

---

## 输出

- 消融对比表格（Table）
- 柱状图对比（Figure）
- 论文正文引用
