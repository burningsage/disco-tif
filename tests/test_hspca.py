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


def test_mixed_resolution_is_refused_unless_allowed(tmp_path):
    fine = write_tif(tmp_path / "fine.tif", _dem(), cell=1.0, nodata=-9999)
    near = write_tif(tmp_path / "near.tif", _dem(seed=1), cell=1.5, nodata=-9999)
    coarse = write_tif(tmp_path / "coarse.tif", _dem(seed=2), cell=5.0, nodata=-9999)
    fit_basis([fine, near])
    with pytest.raises(ValueError, match="allow_mixed_resolution"):
        fit_basis([fine, coarse])
    assert len(fit_basis([fine, coarse], allow_mixed_resolution=True).inputs) == 2


def test_basis_json_round_trips(dem_path, tmp_path):
    import json

    from disco_tif import __version__
    from disco_tif.pca import Basis

    basis = fit_basis(dem_path, block_size=64)
    path = basis.save(tmp_path / "basis.json")
    doc = json.loads(path.read_text())
    assert doc["disco_tif_version"] == __version__ and "component 1" in doc["sign_convention"]
    assert doc["eigenvalue_ratios"][0] == basis.eigenvalues[1] / basis.eigenvalues[0]
    loaded = Basis.load(path)
    for name in ("mean", "weights", "eigenvalues", "explained", "stretch_low", "stretch_high"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(basis, name))
    for name in ("azimuths", "altitudes", "stretch_percentiles", "inputs"):
        assert getattr(loaded, name) == getattr(basis, name)
    (_, a), (_, b) = next(score_blocks(dem_path, basis)), next(score_blocks(dem_path, loaded))
    np.testing.assert_array_equal(a, b)


def test_basis_load_rejects_other_json(tmp_path):
    from disco_tif.pca import Basis

    path = tmp_path / "other.json"
    path.write_text('{"format": "something else"}')
    with pytest.raises(ValueError, match="not a disco-tif basis"):
        Basis.load(path)


def _four_tiles(tmp_path, z, cell=10.0, split=(70, 60)):
    """Write `z` as four edge-to-edge tiles; return their paths and row/column slices."""
    r, c = split
    tiles = []
    for i, (rows, cols) in enumerate([(slice(0, r), slice(0, c)), (slice(0, r), slice(c, None)),
                                      (slice(r, None), slice(0, c)), (slice(r, None), slice(c, None))]):
        origin = (500_000 + cols.start * cell, 4_000_000 - rows.start * cell)
        path = write_tif(tmp_path / f"tile{i}.tif", np.ascontiguousarray(z[rows, cols]), cell=cell,
                         nodata=-9999, origin=origin)
        tiles.append((path, rows, cols))
    return tiles


def test_four_tiles_with_a_pooled_basis_match_the_whole_raster(tmp_path, dem_path):
    """A pooled basis gives the whole raster's pixels everywhere but at the tile seams.

    The pooled basis is exactly that of the tiles' normals taken together. A tile edge
    has no neighbor to difference with, so its outer row or column gets a one-sided
    gradient: the two cells either side of each seam differ from a whole read.
    """
    z = _dem()
    tiles = _four_tiles(tmp_path, z)
    pooled = fit_basis([path for path, *_ in tiles], block_size=32)
    together = sum((Moments.of(dem_normals(path)[0]) for path, *_ in tiles), Moments())
    np.testing.assert_allclose(pooled.weights, basis_from_moments(together).weights, rtol=1e-12, atol=1e-12)

    normals, _ = dem_normals(dem_path)
    whole = transform(normals, pooled)
    pieced = np.full(whole.shape, -5.0)
    for path, rows, cols in tiles:
        for window, block in score_blocks(path, pooled, block_size=32):
            pieced[rows, cols][window.toslices()] = block
    seam = np.zeros(z.shape, bool)
    seam[69:71], seam[:, 59:61] = True, True
    same = np.isclose(pieced, whole, rtol=0, atol=1e-12, equal_nan=True).all(axis=-1)
    assert same[~seam].all()
    assert not same[seam].any()
    whole_basis = basis_from_moments(Moments.of(normals))
    np.testing.assert_allclose(pooled.weights, whole_basis.weights, atol=1e-3 * np.abs(whole_basis.weights).max())
