"""Update model count from 12 to 13 and benchmark counts to 39,188 / 783,760 across the manuscript and documentation."""
from pathlib import Path

ROOT = Path("/Users/rayyan/preprint/ssrn")

files_to_update = [
    ROOT / "paper" / "manuscript.md",
    ROOT / "paper" / "manuscript.tex",
    ROOT / "main.tex",
    ROOT / "README.md",
    ROOT / "CITATION.cff",
    ROOT / "paper" / "submission_docs" / "cover_letter.md",
    ROOT / "paper" / "submission_docs" / "reproducibility_README.md",
]

for file_path in files_to_update:
    if not file_path.exists():
        continue
    text = file_path.read_text()

    # 1. Replace 12 model families / 12 models with 13 model families / 13 models
    text = text.replace("12 model families", "13 model families")
    text = text.replace("12 models", "13 models")
    text = text.replace("12 model", "13 model")

    # 2. Replace 38,380 / 38,000 model fits with 39,188 model fits
    text = text.replace("38,380 model fits", "39,188 model fits")
    text = text.replace("38,380 model", "39,188 model")
    text = text.replace("38,380 fits", "39,188 fits")
    text = text.replace("38,380", "39,188")
    text = text.replace("over 38,000 model fits", "39,188 model fits")

    # 3. Replace 781,740 forecast rows with 783,760 forecast rows
    text = text.replace("781,740 forecast rows", "783,760 forecast rows")
    text = text.replace("781,740", "783,760")

    file_path.write_text(text)
    print(f"Updated {file_path.relative_to(ROOT)}")

print("Model count and benchmark total reconciliation complete.")

