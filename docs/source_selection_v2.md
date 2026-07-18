# V2 Official Source Selection and Acquisition Pathway

**Decision date:** 2026-07-18  
**Status:** Eurostat all-items HICP and Comext bilateral exports have been acquired, ingested, registered, and validated.  
**Scope:** Official-source downloads only. Raw-source ingestion and all transformations remain separate from this decision.

## Selection criteria

A source is usable only if it provides native monthly or quarterly observations, permits preservation of a release/vintage date, has documented terms, and can be archived with an exact request and checksum. For trade, the source must provide reporter–partner merchandise trade. A current “latest” API response is not proof of a past vintage; the acquisition process therefore archives the response bytes and release metadata at download time.

## Trade-source audit

| Candidate | Access / authentication | Native bilateral frequency | Release / vintage metadata | License or operational constraint | Decision |
|---|---|---|---|---|---|
| **IMF IMTS** (formerly DOTS) | IMF portal/downloads and SDMX APIs. Confirm endpoint-specific access; portal/Swagger access may require a Beta Portal account and `IMF_API_TOKEN`. | Monthly reporter–partner total merchandise imports/exports; IMF estimates may be present. | Dataset release calendar, release notices, SDMX metadata, and archived download response. | IMF usage terms apply; do not redistribute raw files without checking permission. | **Conditional primary** for a global, monthly total-goods backbone. Preserve estimate/status flags during later ingestion. |
| **UN Comtrade** | REST API, downloads, and bulk service. A registered `COMTRADE_API_KEY` is needed for reliable/large API extraction; large bulk access may require a premium entitlement. | Monthly global reporter–partner trade, including total goods and product detail where reported. | Continuously updated with no fixed country posting schedule; retain exact query, data availability/release record, release date, and returned file. | UN Comtrade license agreement and subscription limits restrict reuse/redistribution. | **Conditional primary for as-reported/product-level trade;** audit/backfill source for IMTS total goods. |
| **Eurostat Comext** | Public Comext bulk/filtered API; no key normally required. | Monthly bilateral detail for EU/EFTA/candidate reporters, with partner/product dimensions. | Monthly bulk updates and metadata; archive file/version and release date. | Eurostat reuse policy applies; some non-EU and detailed-data restrictions apply. | **Regional fallback / preferred EU source.** |
| **National customs or statistical agencies** | Varies by reporter; APIs, release tables, or bulk files may be public. | Usually monthly bilateral for the reporting country; coverage/history vary. | Typically strong release calendars, versioned tables, and revision notices. | Check agency/table-specific terms. | **Targeted fallback** for priority reporters and source-of-record audit. |
| **World Integrated Trade Solution (WITS)** | Public web/API interface. | Not a dependable current monthly bilateral trade API; key trade products are annual/aggregate. | Does not provide the needed monthly release-vintage control. | Underlying UN rights restrict Comtrade reuse. | **Rejected as primary.** |
| **OECD trade databases** | Public SDMX/Data Explorer subject to terms and rate limits. | Detailed bilateral international trade products are principally annual. | Structure/dataflow version metadata exists, but frequency is unsuitable. | OECD and possible third-party terms apply. | **Rejected as primary.** |

### Trade decision

Use **IMF IMTS monthly bilateral total-goods data** when authenticated, documented extraction is available. Retain all source status/estimate flags in the future ingestion layer; no IMF estimate may be silently represented as a reported customs value. Use **UN Comtrade monthly `TOTAL`** as an audit/fallback source and as the primary source if product-level bilateral analysis is retained. Use **Comext** and national sources only for their documented coverage, without presenting a mixed-source panel as globally complete.

## Macro-source audit

| Candidate | Access / authentication | Native frequency and coverage | Release / vintage support | Terms | Decision |
|---|---|---|---|---|---|
| **OECD Data Explorer SDMX** | Public SDMX REST API; no key documented, with responsible-use/rate limits. Obtain exact request URLs from the Developer API control. | Monthly headline CPI plus energy/food/core measures for OECD members and selected partners. | Dated CPI release schedule and SDMX structural metadata; API is usually latest-revised, so archive every response. | OECD terms and dataset-specific conditions apply. | **Primary harmonised macro feed** for OECD-covered countries. |
| **IMF CPI / IFS** | Portal/download/SDMX; endpoint-specific account or token may be needed. | Broad national monthly CPI coverage; detail varies by reporter, including some energy components. | Dataset release calendar and release metadata; archive each acquired release. | IMF terms apply. | **Broader-country fallback** for CPI and available components. |
| **Eurostat HICP** | Public REST/SDMX and bulk download; no key normally required. | Monthly harmonised headline and energy component data for EU/EEA. | Dated Euro-indicators releases; database is latest-version only, so preserve release files at acquisition. | EU reuse conditions and dataset-specific terms apply. | **Preferred European cross-check / energy source.** |
| **National statistical offices** | Varies. ONS, BLS, Statistics Canada, and ABS are practical models with public releases/APIs. | Definitive national monthly CPI and often energy components. | Often the strongest release/vintage evidence through versioned or archived releases. | Agency-specific open-government or other terms. | **Source of record** for countries requiring release-vintage evidence. |
| **World Bank Global Database of Inflation** | Downloadable research snapshot, updated semiannually. | Broad historical monthly/quarterly coverage where available, including energy measures. | Edition date, not a monthly real-time vintage feed. | Check dataset-specific World Bank terms and attribution. | **Historical coverage check only; not the live primary feed.** |
| **World Bank WDI indicators** | Public indicators API. | Widely used CPI inflation indicator is annual. | Frequency fails the v2 input criterion. | World Bank terms apply. | **Rejected for v2 node inputs.** |

### Macro decision

Use **OECD SDMX** for harmonised monthly headline and energy CPI where coverage permits, with **IMF CPI/IFS** for broader coverage and **Eurostat HICP** for European energy comparability. Where point-in-time release evidence matters, replace or validate those series against the relevant **national statistical office’s archived release**. All source responses must be archived at retrieval; no current API response may be relabeled as a historical vintage.

## Acquisition scripts

The scripts are raw-only: they save response bytes under `data/raw/v2/` and write a provenance sidecar under `data/raw/v2/metadata/acquisition_records/`. They do **not** parse SDMX/JSON, create canonical observations, or modify legacy files.

### UN Comtrade

Set a credential if the chosen endpoint requires it:

```bash
export COMTRADE_API_KEY='your-key'
```

Use a release date from an official Comtrade availability/release record, not the date you run the command:

```bash
.venv/bin/python -m src.acquisition.comtrade \
  --period 2024-01 \
  --reporter-code 842 \
  --partner-code 124 \
  --source-release-date YYYY-MM-DD \
  --source-identifier un_comtrade_monthly_us_can_2024_01_release_YYYY_MM_DD \
  --license-note 'UN Comtrade terms: https://comtradeplus.un.org/LicenseAgreement' \
  --release-lag-description 'Record the official availability/release information for this extract.'
```

The default is the official preview endpoint. Replace `--api-url` only with an official licensed endpoint when the account tier requires it. Do not use the preview response as a production global-panel substitute.

### IMF IMTS

If the selected IMF endpoint requires bearer authentication:

```bash
export IMF_API_TOKEN='your-token'
```

Obtain the exact IMTS SDMX/download URL and release date from the IMF portal/API metadata, then run:

```bash
.venv/bin/python -m src.acquisition.imf \
  --url 'https://official-imf-query-url-from-the-portal' \
  --source-release-date YYYY-MM-DD \
  --source-identifier imf_imts_release_YYYY_MM_DD_slice_001 \
  --coverage-description 'Document reporter, partner, flow, and period selection.' \
  --release-lag-description 'Document the IMF release/vintage and publication timing.' \
  --license-note 'IMF terms: https://www.imf.org/en/About/Copyright-and-Usage'
```

### Official macro source

Get the exact, official query URL from OECD’s Developer API, IMF, Eurostat, or an NSO release/API. The release date is required:

```bash
.venv/bin/python -m src.acquisition.macro_sources \
  --url 'https://official-provider-query-url' \
  --provider 'Organisation for Economic Co-operation and Development' \
  --dataset-name 'Exact official dataset and CPI/energy definition' \
  --frequency monthly \
  --coverage-description 'Country codes, variable codes, and reference-period range' \
  --release-lag-description 'Official published release schedule or release notice URL' \
  --license-note 'Official dataset terms URL' \
  --source-release-date YYYY-MM-DD \
  --source-identifier oecd_cpi_release_YYYY_MM_DD_slice_001
```

## Provenance and manifest procedure

After every downloader succeeds, inspect the archived source response and its sidecar. To verify all source-file hashes before modifying any registry/manifest file:

```bash
.venv/bin/python -m src.acquisition.build_manifest --dry-run
```

When all sidecars are correct:

```bash
.venv/bin/python -m src.acquisition.build_manifest
```

This creates the registry and raw acquisition manifest from verified source files. It still does **not** populate the canonical observation CSVs. A later source-specific **ingestion** step must map only actual observed source values to those schemas, retain source provenance, and then run:

```bash
.venv/bin/python -m src.validate_v2_inputs
```

The validation target requires non-zero canonical macro and trade observations; downloading payloads and building a manifest alone does not satisfy it.

## Current limitation

This environment could not resolve the official UN Comtrade host and has no IMF credential. The credential-free Eurostat pathway nevertheless produced a validated regional panel: Comext `DS-045409` total-goods monthly exports for 20 aligned countries from 2020-01 through 2025-12 (27,341 observed values; data update `2026-07-16`), and HICP `prc_hicp_midx` CP00 monthly indices for the same period (1,944 observed values; data update `2026-02-06`). The current panel is regional rather than global, and it does not yet contain the protocol-required energy component; these boundaries must be preserved downstream. See [`data_acquisition_v2_blockers.md`](data_acquisition_v2_blockers.md) for the live risk record.
