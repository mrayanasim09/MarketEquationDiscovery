# Research Log

## 2026-07-18 — v2.1 journal-readiness amendment initiated

- **Starting commit:** `09ebac44d096fb5ab0ed0bae9150e14ece7db1be`.
- **Historical v2 records preserved:** the append-only `experiments/results/v2/run_manifest.json` contains two prior run records. The first (`v2-4450a858851c-3bb63b1b4018-2026-07-18T154406.982550+0000`) produced no retained forecasts or checkpoints. The second (`v2-09ebac44d096-3bb63b1b4018-2026-07-18T160130.457012+0000`) retained four checkpoints for seed 42, horizon 1, origin 2017Q1 (MLP, LSTM, TCN, and directed GCN) but no forecast store, metrics, or DM tests.
- **Known execution-integrity failure:** the v2 runner permitted a selected model subset while its manifest declared the full locked registry. It also wrote checkpoints during execution. Consequently, the retained v2 records are historical failed/partial evidence only and must never be used for scientific claims.
- **Known scientific gaps:** the v2 contract omits ETS and dynamic-factor baselines, probabilistic forecasts and their scoring/coverage criteria, sMAPE, a fully pre-specified comparator family, a correct multiplicity rule, and an auditable validation-only tuning record.
- **Reason for v2.1:** a fresh prospective protocol is required before any final test forecasts. It leaves the raw/processed data hashes, v2 configuration, failed manifests, checkpoints, and prior commits unchanged. V2.1 is a new configuration identity and result namespace; it does not reinterpret or repair v2 results.
- **Validation planned:** Python compilation, the read-only reproducibility gate, and the v2.1-aware experiment validator only. No model fitting, benchmark execution, forecasts, metrics, or figures are authorized in this amendment.
