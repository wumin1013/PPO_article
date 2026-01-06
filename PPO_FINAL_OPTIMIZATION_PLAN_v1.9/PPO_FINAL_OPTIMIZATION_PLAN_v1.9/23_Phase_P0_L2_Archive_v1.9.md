# Phase 23：P0_L2 归档（新基线固化）（v1.9）
版本日期：2026-01-07  
依赖：22_P0_Baseline_Retrain 已通过验收

---

## 0) 目标
将 22 产出的模型固化为新的 **P0_L2 基线**，作为后续所有 Phase 的对比参照。

---

## 1) 执行步骤

### Step 1：确认 22 验收通过
检查 `artifacts/P0_12d/eval/summary.json`：
- `success_rate ≥ 0.95`
- `max_abs_contour_error ≤ half_epsilon`

### Step 2：归档到 P0_L2
```powershell
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$dest = "PPO_project/artifacts/P0_L2/P0_12d_gold_$timestamp"
New-Item -ItemType Directory -Force -Path $dest
Copy-Item -Recurse "PPO_project/saved_models/P0_12d/*" "$dest/"
Copy-Item -Recurse "PPO_project/artifacts/P0_12d/eval/*" "$dest/"
```

### Step 3：更新 baseline_ref
在后续 Phase 的配置文件中，将 `baseline_ref` 指向新归档路径。

---

## 2) 验收标准
- [ ] 归档目录存在且包含 `best_model.pth` 和 `summary.json`
- [ ] 可通过 `acceptance_suite.py` 对归档模型重新评估并得到一致结果

---

## 3) 交付物
- `artifacts/P0_L2/P0_12d_gold_<timestamp>/` 目录
- 更新后的 `baseline_ref` 路径记录
