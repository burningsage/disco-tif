"""Unit surface normals of a DEM, with real cell spacing and units."""

import numpy as np

from disco_tif.io import read_elevation


def unit_normals(z, dx_east, dy_north):
    """Unit surface normals of an elevation grid.

    Gradients are central differences in the interior and one-sided at the array
    edges (``numpy.gradient`` with ``edge_order=1``, the same as earthpy). A cell is
    invalid when its own elevation is NaN or a neighbor used in its gradient is NaN.

    Parameters
    ----------
    z : ndarray
        Elevation [m], 2-D, rows by columns. NaN marks nodata.
    dx_east : float or ndarray
        East ground distance per column step [m], a scalar or one value per row (as
        returned by `ground_spacing`).
    dy_north : float
        North ground distance per row step [m]; negative for a north-up raster.

    Returns
    -------
    ndarray
        Shape ``(rows, cols, 3)``, float64, components (east, north, up),
        dimensionless. NaN in all three components for invalid cells.

    Raises
    ------
    ValueError
        If `z` is not 2-D or has fewer than 2 rows or columns.
    """
    z = np.asarray(z, dtype=float)
    if z.ndim != 2 or min(z.shape) < 2:
        raise ValueError(f"z must be 2-D with at least 2 rows and columns, got shape {z.shape}")
    dx = np.asarray(dx_east, dtype=float)
    if dx.ndim == 1:
        dx = dx[:, None]
    dz_drow, dz_dcol = np.gradient(z)
    p = dz_dcol / dx          # dz / d(east)
    q = dz_drow / dy_north    # dz / d(north)
    norm = np.sqrt(1.0 + p * p + q * q)
    normals = np.stack([-p / norm, -q / norm, 1.0 / norm], axis=-1)
    normals[np.isnan(z) | np.isnan(norm)] = np.nan
    return normals


def dem_normals(path, *, z_units=None, window=None):
    """Unit surface normals of a DEM file, using its real cell spacing and units.

    Parameters
    ----------
    path : str or path-like
        Elevation raster readable by rasterio; band 1 is used.
    z_units : {None, "m", "ft", "us_ft"}
        Unit of the elevation values. None means the projected CRS unit; required for
        a geographic CRS.
    window : rasterio.windows.Window, optional
        Sub-window to read. The outer row and column of a window get one-sided
        gradients, so callers tiling a raster should read with a one-cell halo.

    Returns
    -------
    normals : ndarray
        Shape ``(rows, cols, 3)``, (east, north, up), dimensionless; NaN where invalid.
    profile : dict
        Raster profile for the area read.
    """
    z, dx_east, dy_north, profile = read_elevation(path, z_units=z_units, window=window)
    return unit_normals(z, dx_east, dy_north), profile
