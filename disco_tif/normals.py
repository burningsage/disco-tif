"""Surface normals of a DEM with real cell spacing, CRS units and vertical units."""

import numpy as np

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
