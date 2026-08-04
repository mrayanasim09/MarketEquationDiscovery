import pytest
import numpy as np
from src.models.graphs.factory import row_normalize, top_k_incoming, build

def test_row_normalize_standard():
    # Standard matrix
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    norm_a = row_normalize(a)
    # Check row sums are 1.0
    assert np.allclose(norm_a.sum(axis=1), 1.0)
    assert np.allclose(norm_a[0], [1/3, 2/3])

def test_row_normalize_zero_row():
    # Matrix with a row of zeros
    a = np.array([[0.0, 0.0], [3.0, 4.0]])
    norm_a = row_normalize(a)
    # Row with zeros should remain zeros (no division by zero error)
    assert np.allclose(norm_a[0], 0.0)
    assert np.allclose(norm_a[1].sum(), 1.0)

def test_top_k_incoming():
    # Matrix of export flows: shape (3, 3)
    # A(i, j) is flow from i to j (exports of i to j / imports of j from i)
    a = np.array([
        [0.0, 2.0, 3.0],
        [4.0, 0.0, 6.0],
        [7.0, 8.0, 0.0]
    ])
    # For k=1, only the largest incoming flow per column (importer) should be kept
    # Col 0 (importer 0): flows are [0, 4, 7] -> keep 7 (row 2)
    # Col 1 (importer 1): flows are [2, 0, 8] -> keep 8 (row 2)
    # Col 2 (importer 2): flows are [3, 6, 0] -> keep 6 (row 1)
    res = top_k_incoming(a, k=1)
    expected = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 6.0],
        [7.0, 8.0, 0.0]
    ])
    assert np.allclose(res, expected)

def test_build_identity():
    # Test building the identity graph variant
    raw_trade = np.random.rand(5, 5)
    rng = np.random.default_rng(42)
    adj = build(raw_trade, "identity_no_trade", rng)
    assert np.allclose(adj, np.eye(5))
