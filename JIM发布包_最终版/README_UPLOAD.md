# Journal of Intelligent Manufacturing 上传说明

本包未投稿、未付款。两套备选稿件共享科学正文，仅期刊模板和投稿材料分开；请勿同时投稿。

按订阅／非OA路线准备。数据获取和利益冲突声明已按作者确认内容加入主稿参考文献前的Statements and Declarations。独立声明文件仅在投稿系统另行要求时使用。

供作者按投稿系统选择的文件：

- `01_manuscript/manuscript.pdf`：LaTeX实际编译的主稿。
- `02_title_page/title_page.pdf`：作者和通讯信息。
- `03_cover_letter/cover_letter.pdf`：本刊Cover Letter。
- `05_declarations/data_availability.pdf`：与主稿一致的独立数据获取声明，仅系统另行要求时上传。
- `06_latex_source_files/latex_source`：可独立编译的LaTeX、参考文献、十幅引用图、样式和构建入口。
- `06_latex_source_files/JIM_editorial_source.zip`：编辑部源码ZIP。
- `05_declarations/declarations.pdf`：Funding、Author contributions、Data availability与Competing interests，仅系统另行要求时上传。

09_preparation_checks、MODIFICATIONS_AND_GAPS及外层整包ZIP为本地核查材料，默认不上传。JIM未将Highlights列为必需。

在对应LaTeX源码目录独立编译，输出到新目录：

```powershell
.\build.ps1 -Engine pdflatex -OutputDirectory D:\JNN_JIM_v17_build
```

从发布包根目录再生成全部指标及实验图，输出目录须不存在：

```powershell
& 'D:\Anaconda\envs\PPO\python.exe' -B .\09_preparation_checks\v10_reproducibility\reproduce_v10.py --output D:\JNN_JIM_v10_repro
```

再生成命令核验四程序56格和五程序10格，包含未完成记录；不重新训练、不执行轴运动。

本刊v17 PDF为27页、五章、十幅正文图、四张表和三个附录。编译和核查记录见09_preparation_checks/journal_build_audit.json。两刊字号、版心、参考文献和作者展示方式不同，页数不要求一致。

## 本次上传文件（v17）

主稿使用 pdfLaTeX，首行为 `%!TEX TS-program = pdflatex`。页面的 xelatex 指令仅适用于实际采用 XeLaTeX 的稿件，本包不切换引擎。

编辑部源码 ZIP 已平铺，无子文件夹；解压后按以下顺序上传：

1. `manuscript.tex`：Manuscript，第一个上传。
2. `figure_1.pdf` 至 `figure_10.pdf`：Figure，编号与正文一致。
3. `references.bib`、`sn-jnl.cls`、`sn-apacite.bst`：LaTeX Supporting File(s)。

四张表已写在主稿中，无单独 Table 文件。投稿所需的编译输入共 14 个文件。构建脚本、LICENSES、README 和校验清单留作本地使用，不作为 LaTeX 支持文件上传。

JIM 官方确认采用 single-anonymous 审稿，保留主稿中的作者、单位与通讯信息。声明已并入主稿，不再需要上传 `journal_declarations.tex`。

官方审稿方式：https://link.springer.com/journal/10845/ethics-and-disclosures

主稿 PDF 为参考附件。作者提供的当前下拉菜单没有 Supplementary/Reference PDF 类型；若后续出现该类型，按页面提示上传 `01_manuscript/manuscript.pdf`。不要把参考主稿 PDF 误选为 Figure、LaTeX Supporting 或数据链接；该类型缺失时需由系统或编辑部确认。

数据链接栏仅填写真实的数据仓库地址。本包沿用作者确认的合理请求获取声明，没有生成公开数据链接。
