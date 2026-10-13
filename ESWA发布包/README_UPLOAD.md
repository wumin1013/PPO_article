# ESWA 投稿文件说明

当前发布文件最终核验日期：2026-10-07，匿名英文正文22页，投稿信2页。交付ZIP保留原文件名。

唯一发布目录：D:/Nutstore/1/基于强化学习的运动规划方法/ESWA发布包

发布ZIP：D:/Nutstore/1/基于强化学习的运动规划方法/ESWA发布包/ESWA_resubmission_20261006.zip

请在投稿系统中按文件类别选择材料，不要将整个交付ZIP作为论文正文上传。以下路径相对发布目录。

| 用途 | 文件 | 说明 |
| --- | --- | --- |
| 匿名正文 | 01_manuscript_without_author_details/manuscript_without_author_details.pdf | 22页干净稿 |
| 署名标题页 | 02_title_page_with_author_details/title_page.docx、.pdf | 五位作者信息和机构邮箱齐全；选择系统接受的一个格式 |
| 投稿信 | 03_cover_letter/cover_letter_ESWA.docx、.pdf | 2页，含可编辑文献对照表；选择一个格式上传 |
| 图件 | 05_figures/*.pdf | 如要求单独图件，按图号选择 |
| LaTeX源码 | 06_latex_source_files/ESWA_anonymous_source.zip | 匿名、自包含；上传到LaTeX/source类别 |
| Highlights | 07_highlights/highlights.docx | 5条，每条不超过85字符 |
| 研究补充材料 | 08_supplementary_reproduction/supplementary_reproduction.zip | 已有轨迹、选定模型、参考路径及核验脚本 |
| ORCID信息 | ORCID/ORCID.docx、.pdf | 按系统要求录入或关联 |

## 投稿信息

题名：Reinforcement Learning-Based Computer Numerical Control Motion Planning with Adaptive Preview and Jerk-Constrained Action Projection

作者顺序：Min Wu、Huicheng Zhou、Zifang Hu、Chao Wang、Weihao Li。

通讯作者：Zifang Hu，huzifang@hust.edu.cn。

全体作者同意、独家投稿及作者信息准确性已由作者确认。发表模式选择订阅／非OA。

| 作者 | 机构邮箱 |
| --- | --- |
| Min Wu | d202580388@hust.edu.cn |
| Huicheng Zhou | Zhouhuicheng@hust.edu.cn |
| Zifang Hu | huzifang@hust.edu.cn |
| Chao Wang | chao_wang@hust.edu.cn |
| Weihao Li | d202480379@hust.edu.cn |

请将上述信息同步到系统字段，并在最终提交前查看系统生成的合并稿。

## 编辑与编译

标题页、投稿信及其他文字附件均保留DOCX和TeX。PDF由对应TeX生成；后续编辑时同步两种源文件。

将ESWA_anonymous_source.zip解压到新目录，在该目录运行：

~~~powershell
& '.\build.ps1' -TeXBin 'D:\texlive\2025\bin\windows'
~~~

可将TeXBin换为其他已有TeX Live安装路径。输出位于_build/manuscript.pdf。附件编译方法见BUILD_AUXILIARY.md。
