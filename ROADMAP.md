# Project Roadmap

This document outlines the high-level milestones and future directions for the "Graph Neural Networks for Macroeconomic Forecasting" research project.

## Completed Milestones

- **v1.0.0**: Initial baseline study establishing feasibility. Data ingestion for 23 countries utilizing CEPII BACI.
- **v2.0.0**: Development of the robust Benchmark Engine, enabling systematic evaluation and reproducible artifact storage.
- **v2.1.0 (Current Release)**: Final journal submission benchmark. Includes full prospective evaluation with expanding windows, probabilistic forecasting, multi-graph variant analysis, and rigorous statistical testing (Moving-block bootstrap Diebold-Mariano).

## Future Work

The following avenues are under consideration for future research extensions:

- **Real-Time Data Integration**: Extending the framework to operate on mixed-frequency datasets and unrevised real-time macroeconomic vintages to simulate operational central bank environments.
- **Global Panel Expansion**: Scaling the methodology beyond 20 European economies to a comprehensive global panel, incorporating emerging markets and assessing differential network effects.
- **Alternative GNN Architectures**: Investigating advanced temporal graph methodologies, including dynamic edge weighting and attention mechanisms tailored specifically to macroeconomic transmission channels.
- **Causal Inference**: Bridging the gap between predictive accuracy and interpretability by incorporating structural causal models into the graph forecasting framework.
