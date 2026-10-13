# 附件TeX编译

正式附件PDF均由对应TeX生成，未用Word导出替代。后续改动需同步DOCX及TeX。

在独立副本中编译，避免中间文件混入上传目录。Cover Letter示例：

~~~powershell
$texBin = 'D:\texlive\2025\bin\windows'
$env:PATH = $texBin + [IO.Path]::PathSeparator + $env:PATH
Push-Location -LiteralPath '.\03_cover_letter'
try {
    & xelatex -interaction=nonstopmode -halt-on-error -file-line-error -recorder cover_letter_ESWA.tex
    if ($LASTEXITCODE -ne 0) { throw 'XeLaTeX failed' }
    & bibtex cover_letter_ESWA
    if ($LASTEXITCODE -ne 0) { throw 'BibTeX failed' }
    & xelatex -interaction=nonstopmode -halt-on-error -file-line-error -recorder cover_letter_ESWA.tex
    if ($LASTEXITCODE -ne 0) { throw 'XeLaTeX failed' }
    & xelatex -interaction=nonstopmode -halt-on-error -file-line-error -recorder cover_letter_ESWA.tex
    if ($LASTEXITCODE -ne 0) { throw 'XeLaTeX failed' }
} finally { Pop-Location }
~~~

其他入口：02_title_page_with_author_details/title_page.tex、07_highlights/highlights.tex、ORCID/ORCID.tex。在各自目录执行XeLaTeX，本轮各执行3遍且每步退出码0，无需BibTeX。

编译环境：TeX Live 2025 / XeTeX、BibTeX 0.99d。也可使用安装了相同模板及宏包的兼容TeX环境。

投稿信中的apacite.sty、apacite.bst、english.apc为未经改动的LPPL文件，版权头保留；许可说明同匿名源码LICENSES.txt。
