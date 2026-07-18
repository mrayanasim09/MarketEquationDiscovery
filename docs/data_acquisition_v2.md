# V2 Data Acquisition and Raw-Input Guide

## Purpose and boundary

This guide documents the raw-data layer for the v2 quarterly inflation forecasting study. Its purpose is to retain source observations and the information needed to reproduce every download. It does **not** construct quarterly aggregates, CPI inflation, node features, graph weights, imputations, or forecasts. Those are later transformations and must remain separately auditable.

The v1 legacy files outside `data/raw/v2/` are permanently archived and must not be copied, converted, or mixed into the v2 raw package.

## Source-selection criteria

A source is admissible only when it provides native monthly or quarterly observations, has a documented release schedule, allows recording a release or vintage date, and has terms that permit the planned research use and redistribution of derived outputs.

| Input | Preferred admissible sources | Excluded source/use |
|---|---|---|
| Bilateral merchandise trade | IMF IMTS/DOTS monthly bilateral trade; UN Comtrade monthly bilateral trade | Annual CEPII BACI values, including any annual-to-quarterly allocation |
| CPI | Official national statistics series, FRED/OECD republished native monthly or quarterly indices with release metadata | Annual CPI rates repeated across quarters |
| Energy price index | Native daily or monthly official/market series, retained at its source frequency | Invented or gap-filled observations |
| Optional macro variables | Native monthly/quarterly policy rates, exchange rates, industrial production, or documented GDP nowcasts | Annual World Bank variables repeated at quarterly frequency |

Before acquisition, record the provider, dataset/table/API endpoint, variable definitions, frequency, coverage, documented release lag, and license in `metadata/source_registry.csv`.

## Required raw-package layout

```text
data/raw/v2/
├── macro/
│   └── macro_observations.csv
├── trade/
│   └── trade_observations.csv
├── metadata/
│   ├── source_registry.csv
│   ├── macro_observations.schema.json
│   ├── trade_observations.schema.json
│   ├── source_registry.schema.json
│   └── raw_manifest.schema.json
└── raw_manifest.json
```

The supplied CSVs contain headers only and the manifest is explicitly marked `placeholder_no_data`. They are templates, not data, and intentionally fail validation until replaced with real source observations and metadata.

## Raw schemas

The JSON schema files in `data/raw/v2/metadata/` are the machine-readable definitions for the macro, trade, source-registry, and manifest inputs. CSV columns are ordered deliberately and must match exactly.

### `macro/macro_observations.csv`

| Column | Meaning |
|---|---|
| `entity_id` | Country/entity identifier; use ISO3 where available |
| `observation_date` | Source observation date, ISO `YYYY-MM-DD` |
| `variable` | Source variable name/code; never a derived v2 feature |
| `value` | Observed numeric source value |
| `unit` | Source-reported unit/index basis |
| `frequency` | `monthly` or `quarterly` |
| `source_release_date` | Date on which this source release/vintage became available |
| `source_identifier` | Stable ID present in the registry and manifest |
| `source_url` | Official source/API query URL |
| `retrieved_at` | UTC timestamp at which the download was obtained |
| `observation_status` | Must be `observed` |
| `missing_reason` | Must be blank; missing or filled observations are not raw observations |

### `trade/trade_observations.csv`

| Column | Meaning |
|---|---|
| `exporter`, `importer` | Exporting/importing country identifiers; use ISO3 where available |
| `observation_period` | `YYYY-MM` for monthly or `YYYYQn` for source-reported quarterly data |
| `trade_value` | Observed non-negative trade value, before graph transformations |
| `currency_unit` | Source-reported currency and unit/scaling (for example, `USD, current, thousands`) |
| `frequency` | `monthly` or `quarterly` |
| `source_release_date`, `source_identifier`, `source_url`, `retrieved_at` | Provenance fields as defined above |
| `observation_status` | Must be `observed` |
| `missing_reason` | Must be blank |

No trade direction aggregation, currency conversion, quarterly aggregation, log transform, or graph normalization belongs in this file.

## Acquisition procedure

1. **Register the source.** Add one complete row in `metadata/source_registry.csv` for every source identifier.
2. **Acquire official data.** Download from an official endpoint or archive. Preserve the exact query URL (including parameters), the provider release/vintage date, and the UTC retrieval timestamp.
3. **Retain original files outside the processed tables where licensing permits.** Place them under a source-specific location within `macro/` or `trade/`, record their relative path and SHA-256 in the manifest, and do not alter them.
4. **Create the canonical raw observation table.** Transcribe only source observations and their provenance into the appropriate CSV. Values must be observed—not interpolated, forward-filled, backfilled, estimated, or derived.
5. **Update `raw_manifest.json`.** Use a non-placeholder status and add a `downloads` record for every registered source file:

   ```json
   {
     "source_identifier": "provider_dataset_variable_release",
     "source_url": "https://official.example/api?exact=query",
     "retrieved_at": "2026-07-18T12:00:00Z",
     "source_release_date": "2026-07-15",
     "sha256": "lowercase-64-character-sha256",
     "local_file": "data/raw/v2/macro/provider_file.csv",
     "license_note": "Provider terms URL or short license statement"
   }
   ```

   `local_file` is project-relative and the validator verifies both its existence and hash. A source identifier in an observation table must occur in both the source registry and the manifest.
6. **Validate before any transformation.** From the project root, run:

   ```bash
   python3 -m src.validate_v2_inputs
   ```

   A passing run writes `data/raw/v2/validation_manifest.json`, containing hashes and row counts. This output is generated and must not replace `raw_manifest.json`.

## Release timing and information set

For each record, `source_release_date` is the date the relevant release/vintage became publicly available—not the observation date and not the local retrieval time. Preserve the release-lag description in the registry. Later forecast construction must use only values available by the forecast origin; in particular, the latest graph available for an end-of-quarter `t` forecast cannot be newer than trade quarter `t-1`.

## Licenses and reproducibility

Do not commit source data that cannot legally be redistributed. Commit the source registry, manifest metadata, acquisition script/query instructions, and hashes instead. For API-only sources, preserve query parameters and pagination choices. For manual downloads, record the archive URL, release label, and a short retrieval note in `license_note` or an accompanying documented acquisition script.

The validator is intentionally strict: it rejects missing provenance, schema drift, duplicate observations, annual frequencies, quarterly trade values with the annual-copy signature, and any row flagged as interpolated, imputed, filled, or missing. A failure is a data-quality finding, not a cue to manufacture replacements.
