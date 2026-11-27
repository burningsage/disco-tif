"""Tests for cell spacing, vertical units and surface normals."""

import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS

from disco_tif.shading import shade
from disco_tif.io import EARTH_RADIUS_M, ground_spacing, z_to_meters
from disco_tif.normals import unit_normals

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


def _grid(n, cell):
    """East and north coordinates [m] of an n x n north-up grid with the given cell size [m]."""
    east = np.arange(n) * cell
    north = (n - 1 - np.arange(n)) * cell  # row 0 is the northern edge
    return np.meshgrid(east, north)


def test_flat_surface_points_up():
    np.testing.assert_allclose(unit_normals(np.full((4, 5), 100.0), 30, -30), [[[0, 0, 1]] * 5] * 4)


def test_plane_rising_east_is_lit_from_the_west():
    """z = east, a 45 degree slope facing west: fully lit from 270, edge-on from 90."""
    east, _ = _grid(6, 10.0)
    n = unit_normals(east, 10.0, -10.0)
    np.testing.assert_allclose(n, np.broadcast_to([-1, 0, 1] / np.sqrt(2), n.shape), atol=1e-12)
    np.testing.assert_allclose(shade(n, 270, 45), 1.0, atol=1e-12)
    np.testing.assert_allclose(shade(n, 90, 45), 0.0, atol=1e-12)


def test_plane_rising_north_is_lit_from_the_south():
    _, north = _grid(6, 10.0)
    n = unit_normals(north, 10.0, -10.0)
    np.testing.assert_allclose(shade(n, 180, 45), 1.0, atol=1e-12)
    np.testing.assert_allclose(shade(n, 0, 45), 0.0, atol=1e-12)


def test_normals_do_not_depend_on_resolution():
    """A quadratic surface sampled at 1 m and at 10 m gives the same interior normals.

    Central differences are exact for quadratics, so the match is to rounding error.
    Unit spacing (earthpy's behavior) would differ by the factor of 10.
    """
    def surface(e, n):
        return 0.002 * e**2 - 0.001 * e * n + 0.003 * n**2 + 0.5 * e

    e1, n1 = _grid(101, 1.0)
    e10, n10 = _grid(11, 10.0)
    fine = unit_normals(surface(e1, n1), 1.0, -1.0)[::10, ::10]
    coarse = unit_normals(surface(e10, n10), 10.0, -10.0)
    np.testing.assert_allclose(coarse[1:-1, 1:-1], fine[1:-1, 1:-1], atol=1e-10)
    wrong = unit_normals(surface(e10, n10), 1.0, -1.0)
    assert not np.allclose(wrong[1:-1, 1:-1], fine[1:-1, 1:-1], atol=1e-3)


def test_per_row_east_spacing_broadcasts():
    east, _ = _grid(5, 1.0)
    n = unit_normals(east * 2.0, np.full(5, 2.0), -1.0)  # slope 1 once spacing is applied
    np.testing.assert_allclose(n[..., 0], -1 / np.sqrt(2), atol=1e-12)


def test_nan_cell_and_its_neighbors_are_invalid():
    z = np.zeros((5, 5))
    z[2, 2] = np.nan
    n = unit_normals(z, 1.0, -1.0)
    invalid = np.isnan(n).all(axis=-1)
    expected = np.zeros((5, 5), bool)
    expected[2, 1:4] = expected[1:4, 2] = True  # the cell and its 4 neighbors
    np.testing.assert_array_equal(invalid, expected)


def test_unit_normals_rejects_bad_shape():
    with pytest.raises(ValueError):
        unit_normals(np.zeros((1, 5)), 1.0, -1.0)
