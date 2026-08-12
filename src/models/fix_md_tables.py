"""Fix broken tables and truncated text in paper/manuscript.md."""
import re
from pathlib import Path

MANUSCRIPT = Path("/Users/rayyan/preprint/ssrn/paper/manuscript.md")

text = MANUSCRIPT.read_text()

# 1. Table 1 replacement
table_1_md = """| Country | Total Obs | Train | Val | Test | Mean CPI (pp) | Std CPI (pp) |
|:---|---:|---:|---:|---:|---:|---:|
| AUT | 170 | 45 | 24 | 101 | 2.93 | 2.44 |
| DEU | 170 | 45 | 24 | 101 | 2.41 | 2.39 |
| EST | 170 | 45 | 24 | 101 | 4.25 | 5.26 |
| FRA | 170 | 45 | 24 | 101 | 1.88 | 1.82 |
| GRC | 170 | 45 | 24 | 101 | 1.36 | 2.90 |
| HUN | 170 | 45 | 24 | 101 | 4.70 | 5.64 |
| ITA | 170 | 45 | 24 | 101 | 1.97 | 2.68 |
| NLD | 170 | 45 | 24 | 101 | 2.68 | 2.94 |
| *Panel mean* | 170 | 45 | 24 | 101 | *2.74* | *3.22* |"""

text = re.sub(
    r'\*\*Table 1: Dataset Summary \(Selected Countries\)\*\*.*?\*Notes:',
    '**Table 1: Dataset Summary (Selected Countries)**\n\n' + table_1_md + '\n\n*Notes:',
    text,
    flags=re.DOTALL
)

# 2. Table 4 replacement
table_4_md = """| Graph Variant | H=1 MAE | H=2 MAE | H=4 MAE | H=1 CRPS | H=2 CRPS | H=4 CRPS |
|:---|---:|---:|---:|---:|---:|---:|
| `identity_no_trade` | **1.853** | **2.428** | **3.254** | **1.530** | **2.071** | **2.854** |
| `directed_trade` | 2.027 | 2.498 | 3.370 | 1.660 | 2.112 | 2.951 |
| `log_trade` | 2.063 | 2.511 | 3.371 | 1.691 | 2.120 | 2.947 |
| `reversed` | 2.043 | 2.517 | 3.399 | 1.674 | 2.130 | 2.980 |
| `undirected` | 2.039 | 2.512 | 3.388 | 1.670 | 2.125 | 2.968 |
| `import_dependence` | 2.046 | 2.551 | 3.396 | 1.681 | 2.169 | 2.983 |
| `degree_preserving_random` | 2.123 | 2.564 | 3.382 | 1.749 | 2.173 | 2.961 |
| `top_k_incoming` | 2.261 | 2.670 | 3.361 | 1.878 | 2.275 | 2.943 |"""

text = re.sub(
    r'\*\*Table 4 \(Ablation\): Graph Variant Performance Aggregated Across GNN\nFamilies\*\*.*?\*Notes:',
    '**Table 4 (Ablation): Graph Variant Performance Aggregated Across GNN Families**\n\n' + table_4_md + '\n\n*Notes:',
    text,
    flags=re.DOTALL
)

# 3. Truncated sentence in §5.4 and Table 6 replacement
sentence_and_table_6 = """We adopt two interpretive thresholds: *majority* (> 50% of seeds significant) as the primary criterion for "consistent" evidence, and *supermajority* (> 75% of seeds) as a secondary criterion for "strong" evidence.

**Table 6 --- Selected DM Test Results** *(proportion of 20 seeds with BH-corrected p < 0.05 favouring GNN; Persistence excluded)*

| Graph Model | Graph Variant | h | Comparator | Prop. Seeds Sig. | Inference |
|:---|:---|:---:|:---|---:|:---|
| GCN | `identity_no_trade` | 1 | Ridge | 80% (16/20) | **Strong** |
| Temporal Graph | `identity_no_trade` | 1 | LSTM | 75% (15/20) | **Strong** |
| GCN | `identity_no_trade` | 2 | Ridge | 65% (13/20) | Consistent |
| GCN | `directed_trade` | 1 | Ridge | 60% (12/20) | Consistent |
| GCN | `identity_no_trade` | 1 | **ARIMA** | 35% (7/20) | Not consistent |
| GCN | `identity_no_trade` | 2 | **ARIMA** | 20% (4/20) | Not consistent |
| Temporal Graph | `identity_no_trade` | 4 | **ARIMA** | 15% (3/20) | Not consistent |
| GCN | `identity_no_trade` | 4 | Ridge | 5% (1/20) | Not consistent |
| Temporal Graph | `identity_no_trade` | 2 | LSTM | 5% (1/20) | Not consistent |"""

text = re.sub(
    r'We adopt two interpretive thresholds: \*majority\* \(\\> 50.*?(?=Two findings follow from Part B\.)',
    sentence_and_table_6 + '\n\n',
    text,
    flags=re.DOTALL
)

MANUSCRIPT.write_text(text)
print("Successfully patched paper/manuscript.md!")
