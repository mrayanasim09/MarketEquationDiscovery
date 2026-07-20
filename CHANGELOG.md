# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2026-07-20
### Added
- V2.1 Journal Benchmark implementation.
- Prospective test configuration with 20 distinct random seeds for robustness.
- Evaluation using 8 different graph structure variants.
- Support for probabilistic forecasting metrics (e.g., CRPS).
- Moving-block bootstrap Diebold-Mariano tests for statistical significance in the presence of serial correlation.

## [2.0.0] - 2026-07-18
### Added
- V2 Benchmark Engine architecture.
- Transactional storage mechanisms for reliable experimental artifact persistence.
- Expanded model registry to support additional baseline and experimental architectures.

## [1.0.0] - 2026-06-27
### Added
- Initial v1 study release (archived).
- Data ingestion pipelines for CEPII BACI.
- Baseline forecasting for 23 countries.
