"""Tests for cell spacing, vertical units and surface normals."""

import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS

from disco_tif.normals import EARTH_RADIUS_M, ground_spacing, z_to_meters

UTM = CRS.from_epsg(32613)          # meters
STATE_PLANE_FT = CRS.from_epsg(2232)  # US survey feet
GEOGRAPHIC = CRS.from_epsg(4326)


def test_spacing_projected_meters_north_up():
    dx, dy = ground_spacing(Affine(30, 0, 500_000, 0, -30, 4_000_000), UTM, height=10)
    assert (dx, dy) == (30, -30)


def test_spacing_projected_us_feet():
    dx, dy = ground_spacing(Affine(10, 0, 0, 0, -10, 0), STATE_PLANE_FT, height=10)
    np.testing.assert_allclose([dx, dy], [10 * 1200 / 3937, -10 * 1200 / 3937])


def test_spacing_geographic_shrinks_with_latitude():
    """One row at the equator and one at 60 N: east spacing halves, north spacing does not."""
    cell = 1 / 3600  # 1 arc-second
    m_per_deg = np.deg2rad(1.0) * EARTH_RADIUS_M
    dx0, dy0 = ground_spacing(Affine(cell, 0, 0, 0, -cell, cell / 2), GEOGRAPHIC, height=1)
    dx60, dy60 = ground_spacing(Affine(cell, 0, 0, 0, -cell, 60 + cell / 2), GEOGRAPHIC, height=1)
    np.testing.assert_allclose(dx0, cell * m_per_deg)
    np.testing.assert_allclose(dx60 / dx0, 0.5, rtol=1e-6)
    np.testing.assert_allclose([dy0, dy60], -cell * m_per_deg)


def test_spacing_rejects_rotation_and_missing_crs():
    with pytest.raises(NotImplementedError):
        ground_spacing(Affine(30, 1, 0, 1, -30, 0), UTM, height=1)
    with pytest.raises(ValueError):
        ground_spacing(Affine(30, 0, 0, 0, -30, 0), None, height=1)


def test_z_units_default_follows_projected_crs():
    assert z_to_meters(UTM) == 1.0
    np.testing.assert_allclose(z_to_meters(STATE_PLANE_FT), 1200 / 3937)


def test_z_units_explicit_and_invalid():
    assert z_to_meters(UTM, "ft") == 0.3048
    assert z_to_meters(GEOGRAPHIC, "m") == 1.0
    with pytest.raises(ValueError, match="z_units is required"):
        z_to_meters(GEOGRAPHIC)
    with pytest.raises(ValueError, match="must be one of"):
        z_to_meters(UTM, "furlong")
