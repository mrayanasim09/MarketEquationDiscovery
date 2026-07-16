#!/usr/bin/env bash
# Build Milestone 9 deliverables: Main_Manuscript.pdf, Online_Appendix.pdf, Replication_Code.zip
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PAPER="$ROOT/paper"
DELIV="$PAPER/deliverables"
TECTonic="$ROOT/.tools/tectonic"

cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true

echo "==> Generating LaTeX tables from JSON..."
python -m src.build_paper_tables

echo "==> Copying figures..."
mkdir -p "$PAPER/figures"
cp "$ROOT/results/explainability/figures/"*.png "$PAPER/figures/"

mkdir -p "$DELIV"

if [[ -x "$TECTonic" ]]; then
  LATEX="$TECTonic"
  LATEX_MODE="tectonic"
elif command -v pdflatex &>/dev/null; then
  LATEX="pdflatex"
  LATEX_MODE="pdflatex"
else
  echo "ERROR: No LaTeX compiler found."
  echo "  Either install BasicTeX (brew install --cask basictex)"
  echo "  or run: curl -sL ... | tar -xzf - -C $ROOT/.tools  (see paper/README)"
  exit 1
fi

compile_tex() {
  local src="$1"
  local out="$2"
  cd "$PAPER"
  if [[ "$LATEX_MODE" == "tectonic" ]]; then
    "$LATEX" -X compile "$src" 2>/dev/null || "$LATEX" "$src"
    mv -f "${src%.tex}.pdf" "$out"
  else
    pdflatex -interaction=nonstopmode "$src" >/dev/null
    bibtex "${src%.tex}" >/dev/null || true
    pdflatex -interaction=nonstopmode "$src" >/dev/null
    pdflatex -interaction=nonstopmode "$src" >/dev/null
    mv -f "${src%.tex}.pdf" "$out"
    rm -f "${src%.tex}.aux" "${src%.tex}.log" "${src%.tex}.bbl" "${src%.tex}.blg" "${src%.tex}.out"
  fi
}

echo "==> Compiling Main_Manuscript.pdf (via $LATEX_MODE)..."
compile_tex main.tex "$DELIV/Main_Manuscript.pdf"

echo "==> Compiling Online_Appendix.pdf..."
compile_tex appendix.tex "$DELIV/Online_Appendix.pdf"

echo "==> Building Replication_Code.zip..."
cd "$ROOT"
zip -r "$DELIV/Replication_Code.zip" \
  src/ \
  config/ \
  requirements.txt \
  paper/REPLICATION_README.md \
  docs/research_protocol.md \
  -x "*.pyc" -x "*__pycache__*" -x "*.DS_Store" >/dev/null

echo ""
echo "Deliverables written to $DELIV/:"
ls -lh "$DELIV/"
