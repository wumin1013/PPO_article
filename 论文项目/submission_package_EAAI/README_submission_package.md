# EAAI Submission Package

This folder contains the Engineering Applications of Artificial Intelligence submission package generated from `论文项目/main.tex`. Original manuscript, bibliography, figure, class/style, and PDF files outside `submission_package_EAAI/` were not modified.

## Folder Contents and Editorial Manager File Types

| Folder | Contents | Editorial Manager file type | Status |
|---|---|---|---|
| `01_manuscript_without_author_details/` | `manuscript_without_author_details.tex`, `manuscript_without_author_details.pdf` | Manuscript without author details | Mandatory |
| `02_title_page_with_author_details/` | `title_page_with_author_details.docx` | Title page with author details | Mandatory |
| `03_cover_letter/` | `cover_letter.docx` | Cover letter | Recommended/usually required |
| `04_declaration_of_competing_interests/` | `declaration_of_competing_interests.docx` | Declaration of competing interests | Mandatory |
| `05_figures/` | `Figure_1.pdf` to `Figure_7.pdf`, `figure_mapping.md` | Figure | Mandatory if figures are uploaded separately |
| `06_latex_source_files/` | `latex_source_anonymous/`, `latex_source_files_anonymous.zip` | LaTeX source files | Mandatory if source files are requested |
| `07_highlights/` | `highlights.docx`, `highlights.txt` | Highlights | Recommended/commonly requested |

## Mandatory Files

- Anonymous manuscript PDF and TeX source.
- Separate title page with author details or TODO fields.
- Declaration of competing interests.
- Figure files used in the manuscript.
- Anonymous LaTeX source zip if Editorial Manager requests source files.

## Recommended Files

- Cover letter.
- Highlights file.
- `figure_mapping.md` for upload traceability.

## Verification Notes

- Anonymous manuscript PDF compiled successfully with XeLaTeX and BibTeX.
- The anonymous source zip was extracted and compiled independently.
- The anonymous first page was checked by text extraction and contains no author or affiliation block.
- PDF metadata has no `Author` field.
- Figure count check passed: 7 `\includegraphics` commands and 7 copied standalone figures.
- Highlights are 61, 63, 63, and 61 characters respectively, all under the 85-character limit.
- DOCX visual render QA could not be completed because LibreOffice/`soffice` is not installed in this environment. DOCX files were generated structurally with `python-docx`.

## TODO Fields Requiring Supervisor Confirmation

- Complete author names and exact author order.
- Full affiliations, postal address, corresponding author name, and email.
- Acknowledgments, funding statement, and author contribution statement.
- Cover letter editor name, corresponding author name, and signature.
- Competing-interest declaration confirmation by the corresponding author.
