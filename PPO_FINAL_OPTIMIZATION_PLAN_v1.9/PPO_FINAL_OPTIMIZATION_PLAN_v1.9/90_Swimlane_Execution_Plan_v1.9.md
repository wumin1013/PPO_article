# PPO 项目执行泳道图（FINAL v1.9）
版本日期：2026-01-07

---

## 执行顺序

```
21_StateSpace → 22_P0_Baseline → 23_P0_L2 → 30_B2a → 40_B2b → 50_B2c → 60_C → 70_D
```

---

## 每个 Phase 的完成条件

| Phase | 完成条件 |
|-------|---------|
| **21** | observation_dim == 12 |
| **22** | success_rate ≥ 0.95 |
| **23** | 归档目录存在 |
| **30** | inside_ratio ≥ 0.6 + corner_sharpness ↓ |
| **40+** | 按各文档定义 |
