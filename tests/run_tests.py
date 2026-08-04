"""
Run all tests without pytest — uses Python's built-in unittest runner.
Equivalent to: pytest tests/ -v
"""
import sys
import math
import unittest
import numpy as np
import torch
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models.graphs.factory import row_normalize, top_k_incoming, build
from src.models.neural import SequenceLSTM, TemporalConvNet, GraphConvolutionForecaster, TemporalGraphForecaster
from src.models.run_benchmark_engine_v2 import MLP
from src.models.evaluate_benchmark_engine_v2_1 import _dm_hln


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestGraphFactory(unittest.TestCase):

    def test_row_normalize_standard(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        norm_a = row_normalize(a)
        self.assertTrue(np.allclose(norm_a.sum(axis=1), 1.0))
        self.assertTrue(np.allclose(norm_a[0], [1/3, 2/3]))

    def test_row_normalize_zero_row(self):
        a = np.array([[0.0, 0.0], [3.0, 4.0]])
        norm_a = row_normalize(a)
        self.assertTrue(np.allclose(norm_a[0], 0.0),
                        "Zero row should remain zero, not produce NaN")
        self.assertTrue(np.allclose(norm_a[1].sum(), 1.0))

    def test_top_k_incoming_k1(self):
        a = np.array([
            [0.0, 2.0, 3.0],
            [4.0, 0.0, 6.0],
            [7.0, 8.0, 0.0]
        ])
        res = top_k_incoming(a, k=1)
        expected = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 6.0],
            [7.0, 8.0, 0.0]
        ])
        self.assertTrue(np.allclose(res, expected))

    def test_top_k_retains_exactly_k(self):
        a = np.random.default_rng(42).random((6, 6))
        k = 3
        res = top_k_incoming(a, k=k)
        for col in range(a.shape[1]):
            nonzero = np.count_nonzero(res[:, col])
            self.assertLessEqual(nonzero, k,
                f"Column {col} has {nonzero} partners, expected <= {k}")

    def test_build_identity(self):
        raw = np.random.rand(5, 5)
        rng = np.random.default_rng(0)
        adj = build(raw, "identity_no_trade", rng)
        self.assertTrue(np.allclose(adj, np.eye(5)))

    def test_build_unknown_variant_raises(self):
        raw = np.random.rand(4, 4)
        rng = np.random.default_rng(0)
        with self.assertRaises(ValueError):
            build(raw, "nonexistent_variant", rng)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL SMOKE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestModels(unittest.TestCase):

    def test_mlp_output_shape(self):
        model = MLP(input_dim=10, hidden_dim=16)
        out = model(torch.randn(5, 10))
        self.assertEqual(out.shape, (5,))

    def test_mlp_no_nan(self):
        model = MLP(input_dim=8, hidden_dim=16)
        out = model(torch.randn(4, 8))
        self.assertFalse(torch.isnan(out).any())

    def test_lstm_output_shape(self):
        model = SequenceLSTM(input_dim=6, hidden_dim=16)
        out = model(torch.randn(4, 4, 6))
        self.assertEqual(out.shape, (4,))

    def test_lstm_no_nan(self):
        model = SequenceLSTM(input_dim=6, hidden_dim=16)
        out = model(torch.randn(3, 4, 6))
        self.assertFalse(torch.isnan(out).any())

    def test_tcn_output_shape(self):
        model = TemporalConvNet(input_dim=6, hidden_dim=16)
        out = model(torch.randn(4, 4, 6))
        self.assertEqual(out.shape, (4,))

    def test_tcn_no_nan(self):
        model = TemporalConvNet(input_dim=6, hidden_dim=16)
        out = model(torch.randn(3, 4, 6))
        self.assertFalse(torch.isnan(out).any())

    def test_gcn_output_shape(self):
        model = GraphConvolutionForecaster(input_dim=8, hidden_dim=16)
        x, adj = torch.randn(5, 8), torch.eye(5)
        out = model(x, adj)
        self.assertEqual(out.shape, (5,))

    def test_gcn_no_nan(self):
        model = GraphConvolutionForecaster(input_dim=8, hidden_dim=16)
        x = torch.randn(5, 8)
        adj = torch.softmax(torch.rand(5, 5), dim=1)
        out = model(x, adj)
        self.assertFalse(torch.isnan(out).any())

    def test_temporal_graph_output_shape(self):
        model = TemporalGraphForecaster(input_dim=8, hidden_dim=16)
        x = torch.randn(4, 5, 8)
        adj = torch.eye(5).unsqueeze(0).expand(4, -1, -1)
        out = model(x, adj)
        self.assertEqual(out.shape, (5,))

    def test_temporal_graph_no_nan(self):
        model = TemporalGraphForecaster(input_dim=8, hidden_dim=16)
        x = torch.randn(4, 5, 8)
        adj = torch.eye(5).unsqueeze(0).expand(4, -1, -1)
        out = model(x, adj)
        self.assertFalse(torch.isnan(out).any())


# ═══════════════════════════════════════════════════════════════════════════════
# DM-HLN STATISTIC TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestDMHLN(unittest.TestCase):

    def test_zero_difference_returns_nan(self):
        stat, pval = _dm_hln(np.zeros(20), horizon=1)
        self.assertTrue(math.isnan(stat))
        self.assertTrue(math.isnan(pval))

    def test_large_positive_difference(self):
        # Use a noisy series with a strongly positive mean — not a constant,
        # since a constant array has zero centered variance and returns NaN (correct).
        rng = np.random.default_rng(7)
        diff = rng.normal(loc=5.0, scale=0.5, size=30)   # mean=5, strongly positive
        stat, pval = _dm_hln(diff, horizon=1)
        self.assertTrue(np.isfinite(stat), f"stat was {stat}")
        self.assertGreater(stat, 0)
        self.assertGreaterEqual(pval, 0)
        self.assertLessEqual(pval, 1)

    def test_sign_flip_negates_statistic(self):
        diff = np.random.default_rng(42).normal(1.0, 0.5, 40)
        stat_pos, _ = _dm_hln(diff, horizon=1)
        stat_neg, _ = _dm_hln(-diff, horizon=1)
        self.assertTrue(np.isclose(stat_pos, -stat_neg, rtol=1e-6))

    def test_too_short_returns_nan(self):
        stat, pval = _dm_hln(np.array([0.1, -0.1]), horizon=1)
        self.assertTrue(math.isnan(stat))
        self.assertTrue(math.isnan(pval))

    def test_known_iid_case(self):
        """For i.i.d. series, h=1 (no HAC lags), DM stat should match formula exactly."""
        rng = np.random.default_rng(123)
        diff = rng.normal(2.0, 1.0, 50)
        stat, pval = _dm_hln(diff, horizon=1)
        n = len(diff)
        var_est = float(np.mean((diff - diff.mean()) ** 2))
        se = math.sqrt(var_est / n)
        hln = math.sqrt((n + 1 - 2 * 1 + 1 * 0 / n) / n)
        expected = (diff.mean() / se) * hln
        self.assertTrue(np.isclose(stat, expected, rtol=1e-5),
                        f"DM stat {stat:.4f} != expected {expected:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestGraphFactory))
    suite.addTests(loader.loadTestsFromTestCase(TestModels))
    suite.addTests(loader.loadTestsFromTestCase(TestDMHLN))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
