"""Sun directions and Lambertian shading of surface normals."""

import numpy as np


def sun_vector(azimuth, altitude):
    """Unit vector pointing from the ground toward the sun.

    Parameters
    ----------
    azimuth : float or array_like
        Sun azimuth [degrees], clockwise from grid north (0 = north, 90 = east), the
        GDAL / ESRI convention.
    altitude : float or array_like
        Sun altitude above the horizon [degrees], 0 to 90.

    Returns
    -------
    ndarray
        Shape ``(..., 3)``; components are (east, north, up), dimensionless.

    Raises
    ------
    ValueError
        If any altitude is outside 0 to 90 degrees.

    Examples
    --------
    >>> sun_vector(90, 0).round(12)
    array([1., 0., 0.])
    """
    az = np.deg2rad(np.asarray(azimuth, dtype=float))
    alt_deg = np.asarray(altitude, dtype=float)
    if np.any((alt_deg < 0) | (alt_deg > 90)):
        raise ValueError(f"altitude must be between 0 and 90 degrees, got {altitude}")
    alt = np.deg2rad(alt_deg)
    az, alt = np.broadcast_arrays(az, alt)
    return np.stack([np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az), np.sin(alt)], axis=-1)


def shade(normals, azimuth, altitude):
    """Unclipped Lambertian shading: the dot product of each normal with the sun vector.

    The result is not clipped at zero, so faces turned away from the sun go negative.
    This keeps every hillshade linear in the normal, which is what makes the hillshade
    PCA a 3x3 problem.

    Parameters
    ----------
    normals : ndarray
        Unit surface normals, shape ``(..., 3)`` as (east, north, up). NaN marks
        invalid cells.
    azimuth : float
        Sun azimuth [degrees], clockwise from north.
    altitude : float
        Sun altitude above the horizon [degrees].

    Returns
    -------
    ndarray
        Shading, shape ``normals.shape[:-1]``, in -1 to 1 (dimensionless). NaN where
        the normal is NaN.
    """
    return normals @ sun_vector(azimuth, altitude)
