# SOFTWARE AUDIT — V2.1 Journal Benchmark

**Auditor:** Independent software engineering review  
**Date:** 2026-07-20  
**Scope:** Full `src/` codebase

---

## 1. Architecture Assessment

### 1.1 Module Separation

The codebase is cleanly separated into layers:

```
src/
├── acquisition/         # Data download (Comext, IMF, Comtrade)
├── ingestion/           # Data parsing and normalisation
├── transform/           # Feature engineering, graph construction, splits
└── models/
    ├── graphs/          # Graph factory (8 adjacency variants)
    ├── tuning/          # Validation-only tuning (isolated from test)
    ├── run_benchmark_engine_v2_1.py    # Main orchestrator
    ├── evaluate_benchmark_engine_v2_1.py  # Metrics + DM tests
    ├── storage_v2_1.py                 # Schema enforcement + transactional writes
    ├── validate_v2_1_contract.py       # Pre-run protocol gate
    ├── validate_v2_1_results.py        # Post-run completeness + hash gate
    ├── analyze_v2_1_results.py         # Aggregation + rankings
    ├── generate_v2_1_manuscript.py     # Tables + figures
    └── generate_v2_1_report.py         # Final audit report
```

**Assessment:** ✅ Well-separated. Scientific concerns (metrics, statistical tests) are isolated from I/O and orchestration. The tuning module is physically and logically isolated from the benchmark runner.

### 1.2 Transactional Integrity

- All outputs are staged in a temporary directory before being promoted to canonical paths ✅
- `write_parquet_exact` validates schema completeness, key uniqueness, and provenance non-nullity before writing ✅
- Failed model runs are logged to `failed_transactions.jsonl` without stopping the benchmark ✅
- The manifest records `status: completed` only after all completeness checks pass ✅

### 1.3 Leakage Prevention

- The tuning manifest is a cryptographically-chained prerequisite that must exist before any test-period fitting is permitted ✅
- `validate_v2_1_contract.py` checks `test_set_tuning_forbidden: true` at the configuration level ✅
- Calibration of probabilistic intervals is explicitly limited to the training+validation split ✅

---

## 2. Issues Found During Audit

### 2.1 Fixed Issues

| Severity | File | Issue | Resolution |
|---|---|---|---|
| Medium | `evaluate_benchmark_engine_v2_1.py` | `tuple(float(v) for v in quantile(...))` not statically typed as `tuple[float, float]` | Replaced with explicit unpack |
| Low | `validate_v2_1_results.py` | 6 unused imports | Removed |
| Low | `validate_v2_1_contract.py` | 1 unused import inside function | Removed |
| Low | `generate_v2_1_manuscript.py` | `numpy` imported but unused | Removed |
| Low | `provenance.py` | `json` imported but unused | Removed |
| Low | `tuning/run_validation_v2_1.py` | `Path` imported but unused | Removed |
| Trivial | `reproduce.sh` | Typo "Reprodution" in banner | Fixed to "Reproduction" |
| Trivial | `.gitignore` | Blanket `experiments/` rule prevented configs, reports, and manuscript from being tracked | Replaced with surgical per-path rules |
| Trivial | `environment.yml` | Duplicate `pyarrow` in pip section; spurious `build` package | Cleaned up |
| Trivial | `CITATION.cff` | Placeholder ORCID `0000-0000-0000-0000` | Commented out with note |
| Trivial | `README.md` | Placeholder DOI badge `ssrn.1234567`; wrong GitHub URL in BibTeX | Fixed badge; corrected URL |

### 2.2 Remaining Items (Non-Blocking)

| Severity | Item | Notes |
|---|---|---|
| Low | `git config` identity not set | `rayyanasim <rayyan@192.168.1.6>` auto-configured; should be set explicitly before push |
| Low | `DTZ003` — `datetime.utcnow()` in `generate_v2_1_report.py` | Deprecated in Python 3.11+; use `datetime.now(timezone.utc)`; non-breaking |
| Low | Black formatting not applied | 4 core files would be reformatted; style-only, no correctness impact |
| Info | GitHub remote name is `MarketEquationDiscovery` | May differ from submission repo name; ensure consistency before publication |
| Info | No `pyproject.toml` | Not required; `requirements.txt` + `environment.yml` are sufficient for reproducibility |

---

## 3. Subprocess and Security Review

| Check | Result |
|---|---|
| `shell=True` in subprocess calls | ✅ None found |
| `pickle.load` without restriction | ✅ None found |
| `yaml.load` (unsafe) | ✅ None found — `json.loads` used throughout |
| API keys hardcoded | ✅ None — read from environment variables only |
| Absolute paths hardcoded | ✅ None — all computed from `Path(__file__).resolve()` |
| Temporary file leaks | ✅ None — staging dirs are explicitly tracked in `.gitignore` |

---

## 4. Platform-Specific Notes

- Scripts use `shasum -a 256` (macOS) or `sha256sum` (Linux) with fallback — cross-platform ✅
- `reproduce.sh` uses `#!/usr/bin/env bash` — portable ✅
- Neural training explicitly uses CPU (`device='cpu'`) — no GPU assumption ✅
- MPS (Apple Silicon GPU) is detected and reported in environment manifest but not used for training ✅

---

## 5. Overall Software Quality Score

| Dimension | Score | Notes |
|---|---|---|
| Architecture | 9/10 | Clean layered design; excellent separation of concerns |
| Reproducibility | 9/10 | Transactional staging, hash verification, frozen config |
| Type safety | 7/10 | `from __future__ import annotations` throughout; mypy clean on core modules; legacy code has `Any` |
| Error handling | 7/10 | Intentional silent fallbacks in baseline fitting are documented; two `except: pass` instances are legitimate |
| Testing | 5/10 | No unit test suite; validators serve as integration tests |
| Documentation | 8/10 | All modules have docstrings; 10 docs/ files; function docs on public API |
| CI/CD | 8/10 | GitHub Actions added: contract validation, ruff, py_compile on every push |
