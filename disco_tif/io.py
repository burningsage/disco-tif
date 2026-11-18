"""Raster georeferencing and reading: cell spacing, vertical units, elevation in meters."""

import numpy as np
import rasterio
from rasterio.windows import Window

EARTH_RADIUS_M = 6_371_008.8
"""Mean Earth radius [m], used to turn degrees into ground distance."""

Z_UNITS_TO_M = {"m": 1.0, "ft": 0.3048, "us_ft": 1200 / 3937}
"""Meters per vertical unit for the supported ``z_units`` values."""


def ground_spacing(transform, crs, height):
    """Signed ground distance between neighboring cells, in meters.

    For a geographic CRS the east spacing shrinks with cos(latitude), so it is returned
    per row. Degrees are converted on a sphere of radius `EARTH_RADIUS_M`, which is an
    approximation that is fine for shading but not for survey work.

    Parameters
    ----------
    transform : affine.Affine
        Raster geotransform. Rotated or sheared transforms are not supported.
    crs : rasterio.crs.CRS
        Horizontal CRS of the raster, projected or geographic.
    height : int
        Number of rows, needed for the per-row spacing of a geographic CRS.

    Returns
    -------
    dx_east : float or ndarray
        East distance per column step [m]; an array of shape ``(height,)`` for a
        geographic CRS. Positive when columns run east.
    dy_north : float
        North distance per row step [m]. Negative for the usual north-up raster, whose
        rows run south.

    Raises
    ------
    ValueError
        If `crs` is None.
    NotImplementedError
        If the transform is rotated or sheared.
    """
    if transform.b != 0 or transform.d != 0:
        raise NotImplementedError("rotated or sheared geotransforms are not supported")
    if crs is None:
        raise ValueError("the raster has no CRS, so its cell spacing is unknown")
    if crs.is_geographic:
        m_per_deg = np.deg2rad(1.0) * EARTH_RADIUS_M
        lat = np.deg2rad(transform.f + transform.e * (np.arange(height) + 0.5))
        return transform.a * m_per_deg * np.cos(lat), transform.e * m_per_deg
    m_per_unit = crs.linear_units_factor[1]
    return transform.a * m_per_unit, transform.e * m_per_unit


def z_to_meters(crs, z_units=None):
    """Meters per unit of the elevation values.

    Parameters
    ----------
    crs : rasterio.crs.CRS
        Horizontal CRS of the raster.
    z_units : {None, "m", "ft", "us_ft"}
        Unit of the elevation values. None means the same unit as the projected CRS
        (meters for UTM, US survey feet for most State Plane CRSs).

    Returns
    -------
    float
        Meters per elevation unit.

    Raises
    ------
    ValueError
        If `z_units` is None and the CRS is geographic (degrees say nothing about the
        vertical unit), or if `z_units` is not a supported value.
    """
    if z_units is None:
        if crs is None or crs.is_geographic:
            raise ValueError("z_units is required when the CRS is geographic or missing; "
                             f"use one of {sorted(Z_UNITS_TO_M)}")
        return crs.linear_units_factor[1]
    try:
        return Z_UNITS_TO_M[z_units]
    except KeyError:
        raise ValueError(f"z_units must be one of {sorted(Z_UNITS_TO_M)}, got {z_units!r}")


def block_windows(height, width, block_size, halo=1):
    """Split a raster into square blocks, each with a halo of neighboring cells.

    Parameters
    ----------
    height, width : int
        Raster size [cells].
    block_size : int
        Block side [cells]; blocks at the right and bottom edges are smaller.
    halo : int
        Extra cells read on each side of a block, clipped at the raster edge.

    Yields
    ------
    block : rasterio.windows.Window
        The cells this block is responsible for. Blocks tile the raster exactly.
    padded : rasterio.windows.Window
        `block` grown by `halo`, the window to read.
    inner : tuple of slice
        Row and column slices that crop an array read with `padded` back to `block`.
    """
    if block_size < 1:
        raise ValueError(f"block_size must be at least 1, got {block_size}")
    for row in range(0, height, block_size):
        for col in range(0, width, block_size):
            rows, cols = min(block_size, height - row), min(block_size, width - col)
            r0, c0 = max(row - halo, 0), max(col - halo, 0)
            r1, c1 = min(row + rows + halo, height), min(col + cols + halo, width)
            yield (Window(col, row, cols, rows), Window(c0, r0, c1 - c0, r1 - r0),
                   (slice(row - r0, row - r0 + rows), slice(col - c0, col - c0 + cols)))


def read_elevation(path, *, z_units=None, window=None):
    """Read band 1 of a DEM as elevation in meters, with nodata as NaN.

    Nodata is taken from the raster: the profile's nodata value (a sentinel or NaN)
    and any internal mask. Non-finite values are always treated as nodata. A raster
    with no nodata value set is read as fully valid.

    Parameters
    ----------
    path : str or path-like
        Single-band (or first-band) elevation raster readable by rasterio.
    z_units : {None, "m", "ft", "us_ft"}
        Unit of the elevation values. None means the projected CRS unit; required for
        a geographic CRS. See `z_to_meters`.
    window : rasterio.windows.Window, optional
        Sub-window to read. None reads the whole raster.

    Returns
    -------
    z : ndarray
        Elevation [m], float64, shape ``(rows, cols)``; NaN where nodata.
    dx_east : float or ndarray
        East ground distance per column step [m]; per row for a geographic CRS.
    dy_north : float
        North ground distance per row step [m]; negative for north-up rasters.
    profile : dict
        The raster profile with width, height and transform set for `window`.
    """
    with rasterio.open(path) as src:
        band = src.read(1, window=window, masked=True)
        transform = src.window_transform(window) if window is not None else src.transform
        crs = src.crs
        profile = src.profile.copy()
    z = band.filled(np.nan).astype(np.float64) if np.ma.is_masked(band) else band.data.astype(np.float64)
    z[~np.isfinite(z)] = np.nan
    z *= z_to_meters(crs, z_units)
    dx_east, dy_north = ground_spacing(transform, crs, height=z.shape[0])
    profile.update(width=z.shape[1], height=z.shape[0], transform=transform)
    return z, dx_east, dy_north, profile
