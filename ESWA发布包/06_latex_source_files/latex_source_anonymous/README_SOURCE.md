# Anonymous ESWA manuscript source

The manuscript uses `elsarticle` with `nonatbib` and the APA sixth-edition `apacite` package (`natbibapa` mode), processed by XeLaTeX and BibTeX. This is not a generic Harvard bibliography.

From PowerShell, run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\build.ps1
```

If TeX Live is not on PATH, pass `-TeXBin 'C:\path with spaces\texlive\bin\windows'`. Standard TeX Live 2025 packages and fonts are required. No private fonts or project research directories are needed. `build.ps1` handles paths with spaces and non-ASCII characters, keeps all files at one source level and writes logs/PDF to `_build`.

Equivalent manual sequence in this directory:

```text
xelatex -interaction=nonstopmode -halt-on-error -file-line-error -recorder manuscript.tex
bibtex manuscript
xelatex -interaction=nonstopmode -halt-on-error -file-line-error -recorder manuscript.tex
xelatex -interaction=nonstopmode -halt-on-error -file-line-error -recorder manuscript.tex
```

All 11 referenced figure PDFs, the bibliography, a generated BBL, unmodified class/style files and their license notices are included. The source archive contains no title page, cover letter, author forms or internal audit logs. Upload source files using the system's LaTeX/source item type; they should not be classified as research supplementary material. The submission system may use a different TeX version, so inspect its generated PDF before final submission.
