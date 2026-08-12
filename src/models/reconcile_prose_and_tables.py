"""Reconcile all prose numbers and remove leftover grid fragments across manuscript files."""
import re
from pathlib import Path

ROOT = Path("/Users/rayyan/preprint/ssrn")

md_file = ROOT / "paper" / "manuscript.md"
tex_file = ROOT / "paper" / "manuscript.tex"
main_file = ROOT / "main.tex"

# 1. Clean up paper/manuscript.md
md_text = md_file.read_text()

# Remove duplicate grid table 3 fragment (lines 529-555 in previous view)
md_text = re.sub(
    r'(\n\| VAR \| --- \| 3\.046 \| 5\.174 \| 8\.973 \| 4\.524 \| 7\.923 \| 14\.088 \|\n)'
    r'(?:\| ARIMA\s+\| ---\s+\|.*?\n\+[-+]*\+\n)+',
    r'\1',
    md_text,
    flags=re.DOTALL
)

# Remove duplicate grid table 5 fragment (lines 641-655 in previous view)
md_text = re.sub(
    r'(\n\| BVAR \| --- \| N/A \| N/A \| N/A \| N/A \| N/A \|\n)'
    r'(?:\| ARIMA\s+\| ---\s+\|.*?\n\+[-+]*\+\n)+',
    r'\1',
    md_text,
    flags=re.DOTALL
)

# Replace prose numbers in manuscript.md
# §5.1 bullets
md_text = md_text.replace(
    "- At **h = 1**, GCN with `identity_no_trade` achieves the lowest point MAE\n"
    "  (1.707 pp), narrowly ahead of ARIMA (1.751 pp). However, this point advantage\n"
    "  is not statistically significant under DM testing for any seed, and GCN's RMSE is\n"
    "  only marginally lower than ARIMA's (2.750 vs 2.884), suggesting comparable\n"
    "  absolute performance.",
    "- At **h = 1**, ARIMA achieves the lowest point MAE (1.699 pp) and RMSE (2.736 pp),\n"
    "  narrowly ahead of GCN with `identity_no_trade` (MAE = 1.707 pp, RMSE = 2.750 pp).\n"
    "  However, this difference is not statistically significant under DM testing for any seed,\n"
    "  suggesting comparable performance."
)

md_text = md_text.replace(
    "- At **h = 2**, Temporal Graph with `identity_no_trade` ranks first in point MAE\n"
    "  (2.358 pp), compared to ARIMA (2.433 pp). While this represents a 3% nominal\n"
    "  reduction in point MAE, the difference is not statistically significant under",
    "- At **h = 2**, Temporal Graph with `identity_no_trade` ranks first in point MAE\n"
    "  (2.358 pp), compared to ARIMA (2.494 pp). While this represents a 5.4% nominal\n"
    "  reduction in point MAE, the difference is not statistically significant under"
)

md_text = md_text.replace(
    "- At **h = 4**, Temporal Graph with `identity_no_trade` achieves the lowest point\n"
    "  MAE (2.840 pp) compared to ARIMA (3.382 pp), a 16% nominal improvement.",
    "- At **h = 4**, Temporal Graph with `identity_no_trade` achieves the lowest point\n"
    "  MAE (2.840 pp) compared to ARIMA (3.716 pp), a 23.6% nominal improvement."
)

# §6.3 Horizon dependence
md_text = md_text.replace(
    "MAE gap versus ARIMA is small at h = 1 (1.999 vs 1.751), widens at h = 2\n"
    "(2.358 vs 2.433; Temporal Graph is now better), and is most pronounced\n"
    "at h = 4 (2.840 vs 3.382).",
    "MAE gap versus ARIMA is small at h = 1 (1.999 vs 1.699; Temporal Graph vs ARIMA), widens at h = 2\n"
    "(2.358 vs 2.494; Temporal Graph is now better), and is most pronounced\n"
    "at h = 4 (2.840 vs 3.716)."
)

# §6.7 Claim to Evidence table
md_text = md_text.replace(
    "MAE = 2.358 pp (h=2), 2.840 pp (h=4). Next best: ARIMA (2.433; 3.382 pp).",
    "MAE = 2.358 pp (h=2), 2.840 pp (h=4). Next best: ARIMA (2.494; 3.716 pp)."
)
md_text = md_text.replace(
    "BVAR MAE = 2.341/3.857/6.312 pp at h=1/2/4 vs ARIMA (1.751/2.433/3.382 pp).",
    "BVAR MAE = 2.341/3.857/6.312 pp at h=1/2/4 vs ARIMA (1.699/2.494/3.716 pp)."
)

# §7 Conclusion
md_text = md_text.replace(
    "achieves MAE of 2.358 pp at $h = 2$ (ARIMA: 2.433 pp) and 2.840 pp\n"
    "    at $h = 4$ (ARIMA: 3.382 pp), suggesting that temporal recurrence in",
    "achieves MAE of 2.358 pp at $h = 2$ (ARIMA: 2.494 pp) and 2.840 pp\n"
    "    at $h = 4$ (ARIMA: 3.716 pp), suggesting that temporal recurrence in"
)

md_file.write_text(md_text)
print("Updated paper/manuscript.md!")


# 2. Update LaTeX files (paper/manuscript.tex and main.tex)
for path in [tex_file, main_file]:
    tex_text = path.read_text()

    # Table 3 row for GCN and ARIMA in LaTeX
    tex_text = tex_text.replace(
        "\\textbf{GCN} & \\textbf{identity\\_no\\_trade} & \\textbf{1.707} & 2.499 &\n"
        "3.668 & 2.750 & 4.249 & 6.164 \\\\\n"
        "ARIMA & --- & 1.751 & \\textbf{2.433} & 3.382 & 2.884 & 3.990 & 5.489 \\\\",
        "GCN & \\texttt{identity\\_no\\_trade} & 1.707 & 2.499 &\n"
        "3.668 & 2.750 & 4.248 & 6.164 \\\\\n"
        "ARIMA & --- & \\textbf{1.699} & 2.494 & 3.716 & \\textbf{2.736} & 4.156 & 6.377 \\\\"
    )
    # Also handle alternate line break variant if any
    tex_text = tex_text.replace(
        "ARIMA & --- & 1.751 & \\textbf{2.433} & 3.382 & 2.884 & 3.990 & 5.489 \\\\",
        "ARIMA & --- & \\textbf{1.699} & 2.494 & 3.716 & \\textbf{2.736} & 4.156 & 6.377 \\\\"
    )
    tex_text = tex_text.replace(
        "\\textbf{GCN} & \\textbf{identity\\_no\\_trade} & \\textbf{1.707} & 2.499 &",
        "GCN & \\texttt{identity\\_no\\_trade} & 1.707 & 2.499 &"
    )

    # Prose replacements in LaTeX
    tex_text = tex_text.replace(
        "lowest point MAE (1.707 pp), narrowly ahead of ARIMA (1.751 pp). However,",
        "lowest point MAE (1.699 pp) achieved by ARIMA, narrowly ahead of GCN (1.707 pp). However,"
    )
    tex_text = tex_text.replace(
        "compared to ARIMA (2.433 pp). While",
        "compared to ARIMA (2.494 pp). While"
    )
    tex_text = tex_text.replace(
        "compared to ARIMA (3.382 pp), a",
        "compared to ARIMA (3.716 pp), a"
    )
    tex_text = tex_text.replace(
        "at h = 1 (1.999 vs 1.751), widens at h = 2",
        "at h = 1 (1.999 vs 1.699), widens at h = 2"
    )
    tex_text = tex_text.replace(
        "at h = 2 (2.358 vs 2.433; Temporal Graph is now better), and is most pronounced\n"
        "at h = 4 (2.840 vs 3.382).",
        "at h = 2 (2.358 vs 2.494; Temporal Graph is now better), and is most pronounced\n"
        "at h = 4 (2.840 vs 3.716)."
    )
    tex_text = tex_text.replace(
        "(2.358 vs 2.433; Temporal Graph is now better), and is most pronounced\nat h = 4 (2.840 vs 3.382).",
        "(2.358 vs 2.494; Temporal Graph is now better), and is most pronounced\nat h = 4 (2.840 vs 3.716)."
    )
    tex_text = tex_text.replace(
        "Next best: ARIMA (2.433; 3.382 pp).",
        "Next best: ARIMA (2.494; 3.716 pp)."
    )
    tex_text = tex_text.replace(
        "vs ARIMA (1.751/2.433/3.382 pp).",
        "vs ARIMA (1.699/2.494/3.716 pp)."
    )
    tex_text = tex_text.replace(
        "achieves MAE of 2.358 pp at $h = 2$ (ARIMA: 2.433 pp) and 2.840 pp at $h = 4$ (ARIMA: 3.382 pp)",
        "achieves MAE of 2.358 pp at $h = 2$ (ARIMA: 2.494 pp) and 2.840 pp at $h = 4$ (ARIMA: 3.716 pp)"
    )

    path.write_text(tex_text)
    print(f"Updated {path.name}!")

print("Reconciliation complete.")
