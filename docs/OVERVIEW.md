# Project Overview

## Title
Predicting Inflation Contagion: A Spatio-Temporal Graph Neural Network Approach to Trade-Linked Economies

## Author
Rayyan Asim, Independent Researcher (mrayanasim09@gmail.com)

## Abstract & Motivation
Inflationary pressures often transcend national borders, cascading through complex global trade networks. Traditional econometric models (e.g., VAR, ARIMA) and standard machine learning architectures typically struggle to explicitly model the structural dependencies between international markets. This research investigates the efficacy of Graph Neural Networks (GNNs)—specifically, Graph Convolutional Networks (GCN) and Temporal Graph Networks—in capturing and forecasting inflation contagion across 20 closely linked European economies. 

The primary motivation is to ascertain whether explicitly defining the cross-border linkages through bilateral trade flows enhances predictive performance over horizons of 1, 2, and 4 quarters ahead, compared to robust deterministic baselines and graph-free neural networks.

## Research Questions
1. Do Spatio-Temporal Graph Neural Networks outperform deterministic and graph-free baselines in forecasting quarterly inflation rates?
2. Does the incorporation of bilateral trade network topologies (e.g., import dependence, directed trade) provide statistically significant improvements over non-trade and identity graph structures?
3. How does predictive performance vary across different forecast horizons (h=1, 2, 4 quarters)?

## Key Contributions
- **Comprehensive Benchmarking Framework**: Development of a rigorous, transaction-safe benchmarking engine evaluating 12 distinct models (7 deterministic, 3 graph-free neural, 2 graph neural) over 20 stochastic seeds.
- **Novel Empirical Graph Construction**: Formulation and testing of 8 variant graph structures based on Eurostat Comext bilateral trade flows, exploring different structural assumptions (e.g., directed trade, log-scaled trade, import dependence, top-K incoming).
- **Rigorous Statistical Evaluation**: Application of the Harvey-Leybourne-Newbold modification of the Diebold-Mariano test with Bartlett HAC kernel, alongside moving-block bootstrap confidence intervals and Benjamini-Hochberg FDR correction.
- **Empirical Findings**: The study reveals that while `temporal_graph` and `GCN` with an `identity_no_trade` graph structure ranked highest at specific horizons (h=2, 4 and h=1 respectively), no single graph-based model consistently maintained statistical significance across all 20 stochastic seeds compared to the best baselines.

## Repository Structure
This repository contains the complete source code, documentation, and data processing pipelines required to replicate the experiments and analyses detailed in the accompanying manuscript. Please refer to `ARCHITECTURE.md` and `REPRODUCIBILITY.md` for technical implementation details.
