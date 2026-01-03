# Phase C：论文图表自动生成（FINAL v1.8）
版本日期：2026-01-03  
依赖：至少存在 baseline_ref（P0_L2）+ 2 个以上 PhaseB Run Bundle。

> 目标：把“写论文”从手工搬运变成自动流水线：  
> 每新增一个 Run Bundle，自动更新论文图/表素材。

---

## 1) 交付目标（C 完成的标志）
- `paper_assets/figures/`：自动生成并按论文标签命名的图（png/pdf）
- `paper_assets/tables/`：tab_results.csv / tab_ablation.csv / tab_hyperparams.csv
- `paper_assets/README.md`：说明这些素材如何被 tex 引用（路径与命名规则）
- 任意 Run Bundle → 一键生成对比图 + 更新表格（不需要手工改图）

---

## 2) 修改流程（不写代码细节，只写步骤）
### Step 1：定义“论文素材规范”
- 图命名：与 tex label 对齐（例如 `fig_curve_s_overlay.png`）
- 表命名：与 tex label 对齐（例如 `tab_results.csv`）
- 指标列：与 Objectives/main_table 列名对齐

### Step 2：实现“自动出图”入口（调用即可）
- 输入：baseline bundle + candidate bundle（可批量）
- 输出：overlay、v(t)、e_n(t)、omega/domega（若有）、recovery_zoom（若有）

### Step 3：实现“自动出表”入口
- 从 main_table.csv 导出 `tab:results`
- 从 ablation_table.csv 导出 `tab:ablation`
- 从 bundle config/manifest 导出 `tab:hyperparams`

### Step 4：与 tex 对齐（不改论文内容，只改引用方式）
- 统一在 tex 中从 `paper_assets/` 读取图表
- 保证重新跑实验后，只需重新生成 paper_assets，tex 自动更新

---

## 3) 验收标准
- 给定一组 bundles：能一次性生成所有论文图/表素材
- paper_assets 中的文件名与 tex label 一一对应
- 无需手工复制粘贴数据即可更新论文表格

---

## 4) 论文映射（与你的 tex 完整对齐）
- Figures：`fig:paths`、`fig:curve_s`、`fig:curve_butterfly`、`fig:ablation_vis`
- Tables：`tab:results`、`tab:ablation`、`tab:hyperparams`

---
