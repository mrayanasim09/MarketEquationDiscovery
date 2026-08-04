Submission Checklist

- [x] Manuscript PDF compiled without errors
- [x] All figures embedded and high-resolution
- [x] Tables formatted and not overfull
- [x] Bibliography formatted according to journal style
- [x] Author contributions statement included
- [x] Data and code availability statement included
- [x] Cover letter prepared
- [x] Suggested reviewers list prepared
- [x] Supplementary materials ready (appendices, extended tables)
- [x] License and ethics statements, if required
- [x] Figures numbered and referenced in the text
- [x] LaTeX source prepared as a single archive for submission

Commands to replicate final PDF:

```bash
pdflatex -interaction=nonstopmode manuscript.tex
bibtex manuscript
pdflatex -interaction=nonstopmode manuscript.tex
pdflatex -interaction=nonstopmode manuscript.tex
```