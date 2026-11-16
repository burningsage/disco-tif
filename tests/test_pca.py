"""Tests for the closed-form hillshade PCA."""

import numpy as np
import pytest

from disco_tif.hillshade import shade
from disco_tif.normals import unit_normals
from disco_tif.pca import DEFAULT_ALTITUDES, DEFAULT_AZIMUTHS, REFERENCE_SUN, Moments, basis_from_moments, sun_matrix, transform


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


def _explicit_stack(normals, azimuths, altitudes):
    """Explicit unclipped hillshade stack, one column per sun direction (valid cells only)."""
    flat = normals.reshape(-1, 3)
    flat = flat[~np.isnan(flat).any(axis=1)]
    return np.stack([shade(flat, a, h) for a in azimuths for h in altitudes], axis=1), flat


def test_closed_form_matches_sklearn_on_explicit_hillshades():
    from sklearn.decomposition import PCA

    u = _normals()
    stack, flat = _explicit_stack(u, DEFAULT_AZIMUTHS, DEFAULT_ALTITUDES)
    ref = PCA(n_components=3).fit(stack)
    basis = basis_from_moments(Moments.of(u))
    np.testing.assert_allclose(basis.eigenvalues, ref.explained_variance_, rtol=1e-9)
    np.testing.assert_allclose(basis.explained, ref.explained_variance_ratio_, rtol=1e-9)
    ours, theirs = transform(flat, basis), ref.transform(stack)
    np.testing.assert_allclose(np.abs(ours), np.abs(theirs), atol=1e-9)


def test_transform_keeps_nan_and_shape():
    u = _normals(20)
    u[4, 5] = np.nan
    scores = transform(u, basis_from_moments(Moments.of(u)))
    assert scores.shape == u.shape
    assert np.isnan(scores[4, 5]).all() and np.isfinite(np.delete(scores.reshape(-1, 3), 4 * 20 + 5, 0)).all()


@pytest.mark.parametrize("azimuths, altitudes", [((0, 180), (30,)), ((0, 90, 180), (90,))])
def test_sun_matrix_rejects_degenerate_sets(azimuths, altitudes):
    with pytest.raises(ValueError, match="span 3-D"):
        sun_matrix(azimuths, altitudes)


def _hill_normals(z_sign=1.0, n=80, cell=10.0):
    """Normals of the standard synthetic DEM, optionally turned upside down."""
    rng = np.random.default_rng(0)
    e, nn = np.meshgrid(np.arange(n) * cell, np.arange(n)[::-1] * cell)
    z = 40 * np.sin(e / 90) * np.cos(nn / 140) + 0.1 * e + rng.normal(0, 0.5, e.shape)
    return unit_normals(z_sign * z, cell, -cell)


@pytest.mark.parametrize("z_sign", [1.0, -1.0])
def test_component_1_reads_like_a_northwest_hillshade(z_sign):
    u = _hill_normals(z_sign)
    basis = basis_from_moments(Moments.of(u))
    flat = u.reshape(-1, 3)
    r = np.corrcoef(transform(flat, basis)[:, 0], shade(flat, *REFERENCE_SUN))[0, 1]
    assert r > 0


@pytest.mark.parametrize("z_sign", [1.0, -1.0])
def test_components_2_and_3_largest_weight_positive(z_sign):
    w = basis_from_moments(Moments.of(_hill_normals(z_sign))).weights
    for j in (1, 2):
        assert w[np.argmax(np.abs(w[:, j])), j] > 0


def test_scores_do_not_depend_on_pixel_order():
    u = _hill_normals().reshape(-1, 3)
    shuffled = u[np.random.default_rng(3).permutation(len(u))]
    a, b = basis_from_moments(Moments.of(u)), basis_from_moments(Moments.of(shuffled))
    np.testing.assert_allclose(a.weights, b.weights, atol=1e-12)


def test_near_tie_warns_on_isotropic_terrain(caplog):
    """A round hill has no preferred horizontal direction, so components 1 and 2 tie."""
    e, nn = np.meshgrid(np.arange(-60, 61) * 5.0, np.arange(-60, 61)[::-1] * 5.0)
    u = unit_normals(80 * np.exp(-(e**2 + nn**2) / 2e4), 5.0, -5.0)
    with caplog.at_level("WARNING", logger="disco_tif.pca"):
        basis_from_moments(Moments.of(u))
    assert "nearly equal variance" in caplog.text


def test_no_warning_on_anisotropic_terrain(caplog):
    with caplog.at_level("WARNING", logger="disco_tif.pca"):
        basis_from_moments(Moments.of(_hill_normals()))
    assert caplog.text == ""
