# P2.5 `offset_debug.py` 使用说明

## 1) 一键自检（推荐）
- 在仓库根目录执行：
  - `python.cmd PPO_project\\tools\\offset_debug.py`

默认会读取 `PPO_project/configs/train_line.yaml`、`train_square.yaml`、`train_s_shape.yaml`，并在 `PPO_project/tools/offset_debug_out/` 输出三张图：
- `offset_debug_line.png`
- `offset_debug_square.png`
- `offset_debug_s_shape.png`

脚本会打印每条路径的 A/B/C/D 统计；若任一条不满足验收标准，会以非 0 退出码结束（便于自动化验收）。

## 2) 常用参数
- 指定输出目录：`python.cmd PPO_project\\tools\\offset_debug.py --outdir PPO_project\\tools\\offset_debug_out`
- 覆盖走廊半宽 `d`：`python.cmd PPO_project\\tools\\offset_debug.py --half-width 0.5`
- 额外加入“锐角拐点”用例：`python.cmd PPO_project\\tools\\offset_debug.py --include-sharp-angle`

## 3) 如何解读失败
- `A(PlSelf)` / `B(PrSelf)`：对应 Pl/Pr 折线自交计数（应为 0）
- `C(QuadBad)`：走廊四边形中“退化或自交”的数量（应为 0）
- `D(SemFail)`：左右语义失败点数量（应为 0）；图中会以红点标注失败点并标注 index
