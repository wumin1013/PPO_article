# 当前科学归档的源码与配置副本

本目录仅供上传论文、代码和模型的远程仓库使用，来源于项目内 JMP 发布包的 `scientific_reproduction_v10.zip`，保留归档成员的相对路径。没有重新训练、执行策略、生成实验结果或更改算法。

本次保留程序源码、类型与构建定义、依赖配置、许可，以及经检查的环境配置、源码来源清单和绘图步骤配置。归档中不同阶段的源码保持原位置；目录名 `data`、`results` 或 `evidence` 下若有程序，仅保留程序及明确配置，不包含该目录原有的数据或运行结果。

原始数据、参考轨迹文件、教师数据、保存的完整执行轨迹、实验结果 JSON/CSV、图源数组、运行日志及包含这些文件的 ZIP 均不在本副本内，仍只保留于本地完整归档。依赖的二进制安装包也未复制。因数据按作者要求不上传，远程副本本身不能完成依赖这些数据的全量结果再生成；完整本地流程仍使用原发布包的复现入口。

`EXTRACTION_MANIFEST.json` 逐项记录归档来源、字节数及提取前后的 SHA256。文本统一为 UTF-8 无 BOM；仅移除原有 UTF-8 BOM，不改正文和换行。源码中的历史路径及配置说明按原证据保留，不代表应直接启动新的训练。

主要入口与配置：

- `v9/src/`：最终启动与制动执行层及相关评价程序。
- `v9/portable_reproduction/`：本地完整数据齐备时使用的结果核查与再生成程序。
- `v10/src/plot_kinematics_clarity_v10.py` 与 `v10/presentation_manifest.json`：图件生成程序及步骤配置。
- `src/`、`evidence/frozen_main_v6/`、`runtime/`：归档中的研究源码与模型执行代码。
- `v9/portable_reproduction/environment_v9.json`：已记录的 Python 和库版本；第三方源码下保留许可与构建配置。

模型独立保存在 `../../论文用到的数据和模型/模型/`。原有 48 个 `.pt` 文件保留；新增 `nnc_checkpoint_v10.pth` 是归档中的 NNC 适配器检查点，SHA256 为 `24d263a0e07fe471e7995ebae2d0a0ee7a2ee03e2feec80ea50aa5b94b0cfe6b`。其配置原件位于本目录 `evidence/frozen_main_v6/nnc_runtime/baseline_policy.yaml`。模型文件字节未修改。
