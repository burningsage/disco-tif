"""Tests for the streamed hillshade PCA of raster files."""

import numpy as np
import pytest
from rasterio.windows import Window

from disco_tif.hspca import _sampled, fit_basis, score_blocks
from disco_tif.normals import dem_normals
from disco_tif.pca import Moments, basis_from_moments, stretch_limits, transform
from rasters import write_tif


def _dem(rows=150, cols=130, cell=10.0, seed=0):
    """Anisotropic synthetic DEM [m] with a nodata hole and a little noise."""
    rng = np.random.default_rng(seed)
    e, nn = np.meshgrid(np.arange(cols) * cell, np.arange(rows)[::-1] * cell)
    z = (40 * np.sin(e / 90) * np.cos(nn / 140) + 25 * np.exp(-((e - 500) ** 2 + (nn - 300) ** 2) / 2e4)
         + 0.1 * e + rng.normal(0, 0.5, e.shape)).astype(np.float32)
    z[40:44, 70:75] = -9999
    return z


@pytest.fixture
def dem_path(tmp_path):
    return write_tif(tmp_path / "dem.tif", _dem(), nodata=-9999)


def test_streaming_equals_in_memory(dem_path):
    normals, _ = dem_normals(dem_path)
    whole = basis_from_moments(Moments.of(normals))
    streamed = fit_basis(dem_path, block_size=64)
    np.testing.assert_allclose(streamed.weights, whole.weights, rtol=1e-12, atol=1e-12)  # summation order
    np.testing.assert_allclose(streamed.eigenvalues, whole.eigenvalues, rtol=1e-9)  # PC3 is small
    low, high = stretch_limits(transform(normals, streamed))  # every cell is in the sample here
    np.testing.assert_array_equal(streamed.stretch_low, low)
    np.testing.assert_array_equal(streamed.stretch_high, high)
    scores = np.full(normals.shape, -1.0)
    for window, block in score_blocks(dem_path, streamed, block_size=64):
        scores[window.toslices()] = block
    np.testing.assert_array_equal(scores, transform(normals, streamed))


def test_inputs_record_name_cell_size_and_count(dem_path):
    (entry,) = fit_basis(dem_path).inputs
    assert entry == {"name": "dem.tif", "cell_size_m": (10.0, 10.0), "count": Moments.of(dem_normals(dem_path)[0]).count}


def test_stretch_sample_does_not_depend_on_block_size(dem_path):
    a = fit_basis(dem_path, block_size=64, sample_size=2000)
    b = fit_basis(dem_path, block_size=17, sample_size=2000)
    np.testing.assert_allclose(a.stretch_low, b.stretch_low, rtol=1e-12)
    np.testing.assert_allclose(a.stretch_high, b.stretch_high, rtol=1e-12)
    c = fit_basis(dem_path, block_size=64, sample_size=2000, seed=1)
    assert not np.allclose(a.stretch_low, c.stretch_low, rtol=1e-6)


def test_sample_keeps_about_the_requested_fraction():
    mask = _sampled(Window(0, 0, 400, 500), 400, keep=0.1, seed=0, raster=0)
    assert abs(mask.mean() - 0.1) < 0.005
    other = _sampled(Window(0, 0, 400, 500), 400, keep=0.1, seed=0, raster=1)
    assert (mask != other).any()


def test_sample_with_no_valid_cells_raises(dem_path):
    with pytest.raises(ValueError, match="sample_size"):
        fit_basis(dem_path, sample_size=0)
