"""Synthetic GeoTIFFs for the tests."""

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS


def write_tif(path, data, *, crs="EPSG:32613", cell=10.0, nodata=None, origin=(500_000, 4_000_000)):
    """Write a single-band north-up GeoTIFF; `cell` is in CRS units."""
    rows, cols = data.shape
    with rasterio.open(path, "w", driver="GTiff", width=cols, height=rows, count=1,
                       dtype=data.dtype, crs=CRS.from_user_input(crs), nodata=nodata,
                       transform=Affine(cell, 0, origin[0], 0, -cell, origin[1])) as dst:
        dst.write(data, 1)
    return path


def hill(n=20, cell=10.0):
    """Smooth synthetic surface [m] on an n x n grid with `cell` spacing [m]."""
    e, nn = np.meshgrid(np.arange(n) * cell, np.arange(n)[::-1] * cell)
    return 50 * np.sin(e / 60) * np.cos(nn / 80) + 0.2 * e
