# Phase D：RCIM 投稿与 Artifact（复现包）（FINAL v1.7.2）
版本日期：2025-12-30  
依赖：Phase C 已能从 Run Bundle 自动生成论文表格与图。

---

## 1) Phase D 的目标

### A1 归档要求（本阶段强制）

- **本阶段每一次 run（训练 / 评估 / 消融），无论 PASS/FAIL，都必须按 A1 归档规范生成 Run Bundle（Level-2）。**
- PASS 的前提：该 run 已形成完整 Run Bundle，且 summary.verdict=PASS。  
  （换句话说：没有归档，就不存在“通过”。）
- 失败 run 也必须归档：这是定位退化/漂移/偶发性的唯一可靠证据链。

把你的工程交付变成“审稿人/读者拿到就能复现”的科研工件：

- 一键复现：给定 Run Bundle（或给定 checkpoint+config），能跑出同样的 eval summary 与关键图
- 清晰说明：硬约束（KCM）与学习贡献（policy）边界清楚
- 复现资源可控：说明需要的硬件、耗时范围、随机性控制方式

---

## 2) Artifact 需要包含的最小内容
- 环境与依赖说明（Python/torch 版本等）
- 训练入口（可选，但至少要能跑 eval）
- 评测入口（必须）：支持 seed_eval 与 episode_set（如果你在 PhaseA 做了）
- Run Bundle 示例：
  - baseline：P0_L2
  - 最佳模型：PhaseB 的 best run
  - 可选：一个失败 run（展示系统可审计性）

---

## 3) 复现声明建议（写给审稿人看的）
- 你如何定义“episode_set”（评测集合）
- 你如何确保评测可重复（seed_eval 与 deterministic）
- 你如何保证硬约束满足（KCM/Shielding 与统计）

---

## 4) Phase D 的验收
- 任何第三方在干净环境中，仅凭 Artifact 文档与提供的 Run Bundle，能复现：
  - 主要 eval 指标
  - 至少一张关键轨迹对比图
- 复现失败时，错误信息可定位（缺依赖、缺文件、路径错误等明确报错）

