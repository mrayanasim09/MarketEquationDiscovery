# V2 Data Acquisition Blockers

**Status:** Raw acquisition milestone complete — official Eurostat macro and bilateral trade observations are acquired, registered, and validated.  
**Last checked:** 2026-07-18

## Principle

The v2 raw package must not be populated from legacy v1 files, generated values, annual-to-quarterly allocations, or observations with inferred release dates. The remaining raw-data risk is scope rather than validity: current trade coverage is a 20-country European regional panel, and the raw macro package contains all-items HICP but not yet the protocol-required energy component.

## Access attempts

| Dataset | Intended use | Official endpoint / source | Result | Required resolution |
|---|---|---|---|---|
| UN Comtrade monthly bilateral merchandise trade | Global/monthly alternate and future product-level extension | `https://comtradeapi.un.org/public/v1/preview/C/M/HS` | Still blocked: the execution environment could not resolve `comtradeapi.un.org` (`curl: (6) Could not resolve host`) | Optional future expansion; use a network-enabled environment and a registered API key or licensed access where necessary. |
| IMF International Trade in Goods by partner country (IMTS; formerly DOTS) | Global/monthly alternate and coverage expansion | IMF IMTS portal / SDMX API | Not acquired: no authenticated IMF credential is available | Optional future expansion; obtain an IMF beta portal/API account and record release/vintage metadata. |
| Eurostat Comext monthly exports | Directed monthly bilateral trade network | Eurostat Comext dataset `DS-045409`, product `TOTAL`, export flow `2`, value indicator in euros | **Acquired and ingested:** 72,737 observed 2010-01–2025-12 country-pair-month exports across 20 aligned countries. Response reports `UPDATE_DATA=2026-07-16T11:00:00+0200`; payloads, SHA-256 hashes, URLs, retrieval times, license notes, registry, and manifest entries are stored in the v2 package. | Raw validation passed. Preserve the Comext regional-coverage boundary in all downstream documentation. |
| Eurostat HICP monthly index | Native monthly macro observations | Eurostat dissemination API, dataset `prc_hicp_midx`, CP00 all-items and CP045 electricity/gas/other-fuels component, index 2015=100 | **Acquired and ingested:** 10,368 observed 2010-01–2025-12 values across 27 countries, including separately archived `EL` Greece slices. Response reports `UPDATE_DATA=2026-02-06T23:00:00+0100`; provenance is stored in the v2 package. | Raw validation passed. Later features must retain the exact CP045 component definition. |

## Acquisition requirements before replacing templates

1. Confirm an admissible official source and its license/terms.
2. Save the source file locally in `data/raw/v2/macro/` or `data/raw/v2/trade/` without alteration where redistribution is permitted.
3. Calculate its SHA-256 and add a corresponding `raw_manifest.json` download entry.
4. Register the source and its release-lag policy in `metadata/source_registry.csv`.
5. Copy only observed source values into the canonical long-form raw table. Do not fill gaps or transform frequency.
6. Record the source release/vintage date, rather than using the local retrieval date as a substitute.
7. Run `.venv/bin/python -m src.validate_v2_inputs` and resolve every failure before downstream transformation.

## Research risk

Raw validation now covers 20 aligned countries over 192 monthly periods (64 future quarters after aggregation), with 223 absent non-self trade country-month cells out of 72,960 possible (0.3056%) and no missing macro values for the two HICP components. Before the transformation stage, the forecasting design must still pre-specify a split with at least 32 final-test origins. A 32-origin test is feasible in principle, but leaves only 32 preceding quarters for training and validation combined; extending pre-2010 history or using a rigorously documented rolling design should be considered before neural-model training. Countries without a complete release-valid CPI and trade history must be excluded rather than filled.
