"""Tests for the streamed hillshade PCA of raster files."""

import numpy as np
import pytest
from rasterio.windows import Window

from disco_tif.hspca import _sampled, fit_basis, score_blocks
from disco_tif.normals import dem_normals
from disco_tif.pca import Moments, basis_from_moments, stretch_limits, to_uint8, transform
from rasters import synthetic_dem, write_tif


@pytest.fixture
def dem_path(tmp_path):
    return write_tif(tmp_path / "dem.tif", synthetic_dem(), nodata=-9999)


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
    fine = write_tif(tmp_path / "fine.tif", synthetic_dem(), cell=1.0, nodata=-9999)
    near = write_tif(tmp_path / "near.tif", synthetic_dem(seed=1), cell=1.5, nodata=-9999)
    coarse = write_tif(tmp_path / "coarse.tif", synthetic_dem(seed=2), cell=5.0, nodata=-9999)
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
    z = synthetic_dem()
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


def test_apply_basis_writes_stretched_components_rgb_and_basis(dem_path, tmp_path):
    import rasterio

    from disco_tif.hspca import apply_basis

    basis = fit_basis(dem_path)
    paths = apply_basis(dem_path, basis, tmp_path / "out", block_size=64)
    assert {k: p.name for k, p in paths.items()} == {
        "c1": "dem_hspca_c1.tif", "c2": "dem_hspca_c2.tif", "c3": "dem_hspca_c3.tif",
        "rgb": "dem_hspca_rgb.tif", "basis": "dem_hspca_basis.json"}
    normals, _ = dem_normals(dem_path)
    expected = to_uint8(transform(normals, basis), basis.stretch_low, basis.stretch_high)
    with rasterio.open(dem_path) as src:
        crs, grid = src.crs, src.transform
    for j in (1, 2, 3):
        with rasterio.open(paths[f"c{j}"]) as dst:
            assert (dst.dtypes[0], dst.nodata, dst.crs, dst.transform) == ("uint8", 0, crs, grid)
            assert dst.profile["tiled"] and dst.compression.name == "deflate"
            np.testing.assert_array_equal(dst.read(1), expected[..., j - 1])
    with rasterio.open(paths["rgb"]) as dst:
        np.testing.assert_array_equal(np.moveaxis(dst.read(), 0, -1), expected)
    assert (expected[41, 72] == 0).all()  # the nodata hole stays nodata
    np.testing.assert_array_equal(type(basis).load(paths["basis"]).weights, basis.weights)


def test_apply_basis_float32_components_are_raw_scores(dem_path, tmp_path):
    import rasterio

    from disco_tif.hspca import apply_basis

    basis = fit_basis(dem_path)
    paths = apply_basis(dem_path, basis, tmp_path, outputs=("components",), dtype="float32")
    assert "rgb" not in paths
    with rasterio.open(paths["c2"]) as dst:
        assert dst.dtypes[0] == "float32" and np.isnan(dst.nodata)
        np.testing.assert_allclose(dst.read(1), transform(dem_normals(dem_path)[0], basis)[..., 1], rtol=1e-6)


def test_apply_basis_rejects_bad_requests(dem_path, tmp_path):
    from disco_tif.hspca import apply_basis

    basis = fit_basis(dem_path)
    with pytest.raises(ValueError, match="outputs"):
        apply_basis(dem_path, basis, tmp_path, outputs=("components", "png"))
    with pytest.raises(ValueError, match="dtype"):
        apply_basis(dem_path, basis, tmp_path, dtype="int16")
    with pytest.raises(ValueError, match="stretch"):
        apply_basis(dem_path, basis_from_moments(Moments.of(dem_normals(dem_path)[0])), tmp_path)


def test_hillshade_pca_is_fit_then_apply_and_reruns_identically(dem_path, tmp_path):
    import rasterio

    from disco_tif.hspca import apply_basis, hillshade_pca

    first = hillshade_pca(dem_path, tmp_path / "a")
    with rasterio.open(first["rgb"]) as dst:
        a = dst.read()
    again = hillshade_pca(dem_path, tmp_path / "a")
    by_hand = apply_basis(dem_path, fit_basis(dem_path), tmp_path / "b")
    for paths in (again, by_hand):
        with rasterio.open(paths["rgb"]) as dst:
            np.testing.assert_array_equal(dst.read(), a)


def test_hillshade_follows_gdaldem_uint8_and_raw_float32(dem_path, tmp_path):
    import rasterio

    from disco_tif.hspca import hillshade
    from disco_tif.shading import shade

    shading = shade(dem_normals(dem_path)[0], 300, 40)
    with rasterio.open(hillshade(dem_path, tmp_path / "hs.tif", azimuth=300, altitude=40, block_size=64)) as dst:
        np.testing.assert_array_equal(dst.read(1), to_uint8(shading, 0, 1))
        assert dst.nodata == 0
    with rasterio.open(hillshade(dem_path, tmp_path / "hs32.tif", azimuth=300, altitude=40, dtype="float32")) as dst:
        np.testing.assert_allclose(dst.read(1), shading, rtol=1e-6)
