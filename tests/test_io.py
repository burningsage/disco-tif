"""Tests for reading DEM files: nodata, units, windows."""

import numpy as np
import pytest
from rasterio.windows import Window

from disco_tif.io import read_elevation
from disco_tif.normals import dem_normals
from rasters import hill, write_tif

US_FT = 1200 / 3937  # meters per US survey foot


def test_sentinel_nodata_becomes_nan(tmp_path):
    z = hill().astype(np.float32)
    z[3, 4] = -9999
    out, *_ = read_elevation(write_tif(tmp_path / "a.tif", z, nodata=-9999))
    assert np.isnan(out[3, 4]) and np.isfinite(np.delete(out.ravel(), 3 * 20 + 4)).all()


def test_nan_nodata_becomes_nan(tmp_path):
    z = hill().astype(np.float32)
    z[5, 5] = np.nan
    out, *_ = read_elevation(write_tif(tmp_path / "b.tif", z, nodata=np.nan))
    assert np.isnan(out).sum() == 1 and np.isnan(out[5, 5])


def test_int_raster_without_nodata_reads(tmp_path):
    """The 0.0.x code raised TypeError here."""
    z = np.round(hill()).astype(np.int16)
    normals, profile = dem_normals(write_tif(tmp_path / "c.tif", z))
    assert np.isfinite(normals).all() and profile["nodata"] is None


def test_feet_elevations_on_meter_crs(tmp_path):
    z = hill()
    ref, _ = dem_normals(write_tif(tmp_path / "m.tif", z))
    ft, _ = dem_normals(write_tif(tmp_path / "ft.tif", z / 0.3048), z_units="ft")
    np.testing.assert_allclose(ft, ref, atol=1e-12)


def test_state_plane_feet_default_units(tmp_path):
    """Horizontal and vertical both in US feet: the default z_units follows the CRS."""
    z = hill(cell=10.0)
    ref, _ = dem_normals(write_tif(tmp_path / "m.tif", z, cell=10.0))
    ft, _ = dem_normals(write_tif(tmp_path / "sp.tif", z / US_FT, crs="EPSG:2232",
                               cell=10.0 / US_FT, origin=(3_000_000, 1_500_000)))
    np.testing.assert_allclose(ft, ref, atol=1e-12)


def test_geographic_needs_z_units(tmp_path):
    path = write_tif(tmp_path / "g.tif", hill(), crs="EPSG:4326", cell=1 / 3600, origin=(-106, 36))
    with pytest.raises(ValueError, match="z_units is required"):
        dem_normals(path)
    normals, _ = dem_normals(path, z_units="m")
    assert np.isfinite(normals).all()


def test_window_matches_full_read_inside_its_edge(tmp_path):
    path = write_tif(tmp_path / "w.tif", hill(30))
    full, _ = dem_normals(path)
    part, profile = dem_normals(path, window=Window(5, 7, 12, 10))  # col_off, row_off, w, h
    np.testing.assert_allclose(part[1:-1, 1:-1], full[8:16, 6:16], atol=1e-12)
    assert (profile["width"], profile["height"]) == (12, 10)


@pytest.mark.parametrize("height, width, block_size", [(10, 7, 3), (5, 5, 5), (4, 9, 1), (6, 6, 100)])
def test_block_windows_tile_the_raster_exactly(height, width, block_size):
    from disco_tif.io import block_windows

    hits = np.zeros((height, width), int)
    for block, padded, inner in block_windows(height, width, block_size):
        hits[block.toslices()] += 1
        rows = np.arange(padded.row_off, padded.row_off + padded.height)[inner[0]]
        cols = np.arange(padded.col_off, padded.col_off + padded.width)[inner[1]]
        assert rows[0] == block.row_off and len(rows) == block.height
        assert cols[0] == block.col_off and len(cols) == block.width
        assert padded.row_off >= 0 and padded.row_off + padded.height <= height
    assert (hits == 1).all()


@pytest.mark.parametrize("crs, cell", [("EPSG:32613", 10.0), ("EPSG:4326", 1e-4)])
def test_normal_blocks_equal_a_whole_read(tmp_path, crs, cell):
    """The halo makes block seams invisible, also with per-row spacing in degrees.

    Geographic blocks recompute their latitudes from the window origin, which can
    differ from the whole-raster value in the last bit, hence the 1e-15 tolerance.
    """
    from disco_tif.normals import normal_blocks

    z = hill(23).astype(np.float32)
    z[7, 8] = -9999
    origin = (-106.0, 36.0) if crs == "EPSG:4326" else (500_000, 4_000_000)
    path = write_tif(tmp_path / "a.tif", z, crs=crs, cell=cell, nodata=-9999, origin=origin)
    z_units = "m" if crs == "EPSG:4326" else None
    whole, _ = dem_normals(path, z_units=z_units)
    pieced = np.full_like(whole, -1.0)
    for window, normals in normal_blocks(path, z_units=z_units, block_size=5):
        pieced[window.toslices()] = normals
    np.testing.assert_allclose(pieced, whole, rtol=0, atol=0 if crs == "EPSG:32613" else 1e-15)
