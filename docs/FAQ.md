# Frequently Asked Questions

## Methodology

**Q: Why use the Harvey-Leybourne-Newbold modification for the DM test?**
A: Standard DM tests over-reject the null hypothesis in small samples (our test set is ~35 quarters). The HLN adjustment corrects for this, providing more conservative and reliable estimates of statistical significance, especially for multi-step horizons.

**Q: Why evaluate 20 different random seeds?**
A: Deep learning models exhibit significant initialization variance. A GNN might randomly converge to a highly performant state on a single seed, giving a false impression of superiority. Averaging across 20 seeds ensures that reported performance stems from architectural advantages, not "lucky" initializations.

**Q: Why did the `identity_no_trade` graph outperform the trade-based graphs?**
A: Our findings suggest that for this specific EU panel, the temporal sequence of domestic inflation and energy prices contains sufficient information to forecast future domestic inflation. Forcing the model to process external signals via trade topologies introduced noise rather than predictive signal, degrading performance.

## Reproduction and Execution

**Q: I am getting `LinAlgError` or Convergence Warnings. Is the pipeline broken?**
A: No. In the early stages of the expanding window (e.g., 2017), the training set is very small. Complex baselines like VAR may struggle to invert matrices. The pipeline logs these warnings and falls back to robust pseudo-inverses or simpler persistence forecasts automatically.

**Q: How long does the full benchmark take to run?**
A: On a modern workstation (e.g., 16 cores, 32GB RAM), the full suite (38,380 fits) takes approximately 12 to 18 hours. You can reduce `n_seeds` in `config/benchmark.yaml` to 1 or 2 for a rapid smoke test.

**Q: Can I add my own country data?**
A: Yes, but you must supply both the domestic macroeconomic indicators and the $N \times N$ bilateral trade matrices, formatted identically to the Eurostat schemas defined in `DATA_DESCRIPTION.md`. Update `config/data_hashes.json` to bypass strict integrity checks.
