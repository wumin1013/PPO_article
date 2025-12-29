# Phase A（拆分版）：A-1 尺子 / A-2 机器 / A-3 流水线（FINAL v1.6）
版本日期：2025-12-30  
属性：优先保持行为等价；但允许做“必要的工程升级”，因为 B 阶段的判定依赖它。

---

# A-1：先造尺子（指标计算 + 分段口径 + 验收口径）
## A-1.0 目的
让你在进入 B1/B2 之前就拥有“看得清”的判定工具：  
- 能区分“直线更快了”还是“拐角切得更狠了”  
- 能稳定计算 straight/corner/exit 分段指标  
- 能输出一致的 summary.json 并给出清晰的 FAIL reasons

## A-1.1 逐步执行清单
1) 固化 `summary.json` schema（至少覆盖：精度/成功/效率/直线分段/拐角平滑/出弯专项）
2) 固化分段口径（masks）：
   - corner_mask（覆盖入弯—过弯—出弯窗口）
   - straight_mask（其余）
   - exit_active（拐角结束后短窗口）
3) 固化“可统计的平滑性指标”：
   - 允许用 proxy，但必须跨轨迹可比、采样周期一致
   - 输出峰值 + p95（更稳健）
4) 固化 baseline compare：
   - 对 P0 baseline 输出 reasons（字段、阈值、当前值、baseline 值）
5) 固化最小证据链自动生成：
   - episodes（逐 episode）
   - overlay / corner zoom / v_ratio 曲线

## A-1.2 产物清单
- 一次 P0 eval 产物：summary + meta + plots + episodes
- baseline compare 输出（PASS/FAIL reasons）

## A-1.3 验收（PASS/FAIL）
PASS（必须同时满足）：
- 同一模型重复 eval（deterministic + 固定 seed）指标稳定（允许小数值误差，但趋势不应漂移）
- summary.json 字段齐全（至少能得到 mean_v_ratio_straight 与一个平滑性指标）
- baseline compare 能输出可读 reasons
- 自动生成最小证据链（plots/episodes）

FAIL 常见信号：
- 同一模型重复 eval 指标飘：统计口径或随机性没收敛（先修口径再做 B）
- corner_mask 不可信：corner zoom 的标注与直觉不一致（先修 mask）

---

# A-2：再造机器（训练吞吐加速）
## A-2.0 目的
为 B 阶段与多 seed 试验“减速阻力”：  
- 吞吐提升 ≥ 2.0x（建议目标 3–6x）
- 不改变学习问题定义（行为等价优先）

## A-2.1 逐步执行清单（不限定实现方式）
1) 建立性能基线（同硬件/同配置/同训练步数）
2) 定位瓶颈（env.step、几何特征、同步、IO）
3) 优先做吞吐大头：
   - 多环境并行采样（方式不限）
   - 几何特征预计算/缓存
   - 热点批处理/向量化
   - 日志降频与 IO 控制
4) 回归验证：P0 eval 仍 PASS（硬约束）

## A-2.2 产物清单
- baseline vs after 的吞吐对比记录
- P0 回归 eval 产物（summary/meta/plots/episodes）

## A-2.3 验收（PASS/FAIL）
PASS：
- 吞吐提升 ≥ 2.0x
- P0 回归仍 PASS
- 没有引入新的不稳定性（NaN、随机飘）

FAIL 诊断：
- 吞吐没提升：热点仍在 env.step 或重复几何计算（回到瓶颈画像再做）
- 回归失败：怀疑语义被改或统计口径漂移（优先回滚定位差异）

---

# A-3：最后造流水线（多 seed 自动化 + 自动聚合出图）
## A-3.0 目的
把“跑实验”变成“提交任务”：  
- 一键跑 5–10 seeds（训练+评估）
- 自动聚合输出（均值/方差）与最小论文图（overlay/corner/v_ratio/平滑）

## A-3.1 逐步执行清单（不限定用 shell 还是 python）
1) 输入协议：方法名/配置/seed 列表/输出目录
2) 运行协议：
   - 每个 seed 独立产物目录（summary/meta/plots/episodes）
   - 失败要能定位（done_reason + 哪个 seed）
3) 聚合协议：
   - 自动汇总 summary（均值/方差/必要分位数）
   - 生成主表 CSV（最小字段见目标文档）
4) 出图协议：
   - 至少生成：overlay、corner zoom、v_ratio、平滑性曲线（或统计图）

## A-3.2 产物清单
- 多 seed 的 run 列表（每个都有 summary）
- 聚合 CSV（主表雏形）
- 最小图表集（论文雏形）

## A-3.3 验收（PASS/FAIL）
PASS：
- 一键跑 ≥5 seeds 并自动生成聚合 CSV + 图
- 失败 seed 能自动标出并给出原因（便于重跑/定位）
- 输出路径结构稳定（Phase D 直接复用）

FAIL 诊断：
- 聚合出错：往往是 schema 不稳定（回到 A-1 固化 schema）
- 图表缺失：绘图输入不齐（先补齐 episodes 或关键时间序列）

