Submission Checklist

- [ ] Manuscript PDF compiled without errors
- [ ] All figures embedded and high-resolution
- [ ] Tables formatted and not overfull
- [ ] Bibliography formatted according to journal style
- [ ] Author contributions statement included
- [ ] Data and code availability statement included
- [ ] Cover letter prepared
- [ ] Suggested reviewers list prepared
- [ ] Supplementary materials ready (appendices, extended tables)
- [ ] License and ethics statements, if required
- [ ] Figures numbered and referenced in the text
- [ ] LaTeX source prepared as a single archive for submission

Commands to replicate final PDF:

```bash
pdflatex -interaction=nonstopmode manuscript.tex
bibtex manuscript
pdflatex -interaction=nonstopmode manuscript.tex
pdflatex -interaction=nonstopmode manuscript.tex
```