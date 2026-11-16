"""Tests for reading DEM files: nodata, units, windows."""

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.windows import Window

from disco_tif.io import read_elevation
from disco_tif.normals import dem_normals

US_FT = 1200 / 3937  # meters per US survey foot


def _write(path, data, *, crs="EPSG:32613", cell=10.0, nodata=None, origin=(500_000, 4_000_000)):
    """Write a single-band north-up GeoTIFF; `cell` is in CRS units."""
    rows, cols = data.shape
    with rasterio.open(path, "w", driver="GTiff", width=cols, height=rows, count=1,
                       dtype=data.dtype, crs=CRS.from_user_input(crs), nodata=nodata,
                       transform=Affine(cell, 0, origin[0], 0, -cell, origin[1])) as dst:
        dst.write(data, 1)
    return path


def _hill(n=20, cell=10.0):
    """Smooth synthetic surface [m] on an n x n grid with `cell` spacing [m]."""
    e, nn = np.meshgrid(np.arange(n) * cell, np.arange(n)[::-1] * cell)
    return 50 * np.sin(e / 60) * np.cos(nn / 80) + 0.2 * e


def test_sentinel_nodata_becomes_nan(tmp_path):
    z = _hill().astype(np.float32)
    z[3, 4] = -9999
    out, *_ = read_elevation(_write(tmp_path / "a.tif", z, nodata=-9999))
    assert np.isnan(out[3, 4]) and np.isfinite(np.delete(out.ravel(), 3 * 20 + 4)).all()


def test_nan_nodata_becomes_nan(tmp_path):
    z = _hill().astype(np.float32)
    z[5, 5] = np.nan
    out, *_ = read_elevation(_write(tmp_path / "b.tif", z, nodata=np.nan))
    assert np.isnan(out).sum() == 1 and np.isnan(out[5, 5])


def test_int_raster_without_nodata_reads(tmp_path):
    """The 0.0.x code raised TypeError here."""
    z = np.round(_hill()).astype(np.int16)
    normals, profile = dem_normals(_write(tmp_path / "c.tif", z))
    assert np.isfinite(normals).all() and profile["nodata"] is None


def test_feet_elevations_on_meter_crs(tmp_path):
    z = _hill()
    ref, _ = dem_normals(_write(tmp_path / "m.tif", z))
    ft, _ = dem_normals(_write(tmp_path / "ft.tif", z / 0.3048), z_units="ft")
    np.testing.assert_allclose(ft, ref, atol=1e-12)


def test_state_plane_feet_default_units(tmp_path):
    """Horizontal and vertical both in US feet: the default z_units follows the CRS."""
    z = _hill(cell=10.0)
    ref, _ = dem_normals(_write(tmp_path / "m.tif", z, cell=10.0))
    ft, _ = dem_normals(_write(tmp_path / "sp.tif", z / US_FT, crs="EPSG:2232",
                               cell=10.0 / US_FT, origin=(3_000_000, 1_500_000)))
    np.testing.assert_allclose(ft, ref, atol=1e-12)


def test_geographic_needs_z_units(tmp_path):
    path = _write(tmp_path / "g.tif", _hill(), crs="EPSG:4326", cell=1 / 3600, origin=(-106, 36))
    with pytest.raises(ValueError, match="z_units is required"):
        dem_normals(path)
    normals, _ = dem_normals(path, z_units="m")
    assert np.isfinite(normals).all()


def test_window_matches_full_read_inside_its_edge(tmp_path):
    path = _write(tmp_path / "w.tif", _hill(30))
    full, _ = dem_normals(path)
    part, profile = dem_normals(path, window=Window(5, 7, 12, 10))  # col_off, row_off, w, h
    np.testing.assert_allclose(part[1:-1, 1:-1], full[8:16, 6:16], atol=1e-12)
    assert (profile["width"], profile["height"]) == (12, 10)
