"""Tests for the closed-form hillshade PCA."""

import numpy as np
import pytest

from disco_tif.normals import unit_normals
from disco_tif.pca import Moments


def _normals(n=80, cell=10.0, seed=0):
    """Normals of a smooth, anisotropic synthetic DEM with a little noise."""
    rng = np.random.default_rng(seed)
    e, nn = np.meshgrid(np.arange(n) * cell, np.arange(n)[::-1] * cell)
    z = (40 * np.sin(e / 90) * np.cos(nn / 140) + 25 * np.exp(-((e - 500) ** 2 + (nn - 300) ** 2) / 2e4)
         + 0.1 * e + rng.normal(0, 0.5, e.shape))
    return unit_normals(z, cell, -cell)


def test_moments_match_numpy_and_skip_nan():
    u = _normals()
    u[3, 3] = np.nan
    flat = u.reshape(-1, 3)
    flat = flat[~np.isnan(flat).any(axis=1)]
    m = Moments.of(u)
    assert m.count == len(flat)
    np.testing.assert_allclose(m.mean, flat.mean(axis=0))
    np.testing.assert_allclose(m.cov, np.cov(flat, rowvar=False), rtol=1e-10, atol=1e-14)


def test_moments_add_across_blocks():
    """Block moments sum to the whole-array moments, up to float summation order."""
    u = _normals()
    whole = Moments.of(u)
    parts = Moments.of(u[:30]) + Moments.of(u[30:55]) + Moments.of(u[55:])
    assert parts.count == whole.count
    np.testing.assert_allclose(parts.cov, whole.cov, rtol=1e-10, atol=1e-13)


def test_covariance_needs_two_cells():
    with pytest.raises(ValueError):
        Moments.of(np.full((1, 1, 3), 0.5)).cov
