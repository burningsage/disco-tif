"""Hillshade PCA of raster files, streamed block by block.

Pass 1 (`fit_basis`) adds up the moments of the unit normals of every block of
every input and keeps a seeded sample of cells for the stretch. Pass 2
(`score_blocks`) applies the basis block by block. Memory is set by
``block_size``, not by the size of the rasters.
"""

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import rasterio

from disco_tif.io import cell_size
from disco_tif.normals import normal_blocks
from disco_tif.pca import DEFAULT_ALTITUDES, DEFAULT_AZIMUTHS, Moments, basis_from_moments, stretch_limits, transform


def fit_basis(src_paths, *, azimuths=DEFAULT_AZIMUTHS, altitudes=DEFAULT_ALTITUDES, z_units=None,
              block_size=1024, sample_size=1_000_000, stretch=(1, 99), seed=0,
              allow_mixed_resolution=False):
    """Fit one hillshade PCA basis to one or more DEMs, streaming.

    The moments of all inputs are added, so the basis is exactly that of all their
    cells taken together, without building a mosaic. One caveat for tiles cut from
    one DEM: a tile's outer row and column have no neighbor to difference with, so
    they get one-sided gradients, and the cells either side of a seam differ from a
    whole read (the basis barely changes; the pixels on the seam do). For seamless
    products, run on a mosaic of the tiles instead, such as a GDAL VRT, which is read
    block by block like any other raster.

    Parameters
    ----------
    src_paths : str, path-like or sequence of them
        Elevation rasters readable by rasterio; band 1 is used.
    azimuths : sequence of float
        Sun azimuths [degrees clockwise from north].
    altitudes : sequence of float
        Sun altitudes [degrees above the horizon].
    z_units : {None, "m", "ft", "us_ft"}
        Unit of the elevation values, the same for every input. None means the
        projected CRS unit; required for a geographic CRS.
    block_size : int
        Block side [cells]. Changes memory use only, not the result.
    sample_size : int
        About this many cells, chosen by a seeded hash of their position, set the
        stretch limits (fewer when the rasters hold nodata). With fewer cells than
        this in total, every cell is used. The choice does not depend on `block_size`.
    stretch : tuple of float
        Low and high percentile [0-100] of the scores that map to 1 and 255.
    seed : int
        Seed of the sample.
    allow_mixed_resolution : bool
        Slope statistics change with cell size, so by default inputs whose east or
        north cell sizes differ by more than `MAX_CELL_SIZE_RATIO` are refused. True
        pools them anyway.

    Returns
    -------
    Basis
        With the stretch limits and one `inputs` entry per raster filled in.

    Raises
    ------
    ValueError
        If the cell sizes are mixed and not allowed, the inputs hold fewer than two
        valid cells, or the sample holds none.
    """
    paths = [src_paths] if isinstance(src_paths, (str, os.PathLike)) else list(src_paths)
    cells = [cell_size(path) for path in paths]
    if not allow_mixed_resolution:
        _check_resolution(paths, cells)
    shapes = []
    for path in paths:
        with rasterio.open(path) as src:
            shapes.append((src.height, src.width))
    keep = min(1.0, sample_size / sum(h * w for h, w in shapes))
    moments, sample, inputs = Moments(), [], []
    for i, (path, (_, width)) in enumerate(zip(paths, shapes)):
        own = Moments()
        for window, normals in normal_blocks(path, z_units=z_units, block_size=block_size):
            own += Moments.of(normals)
            sample.append(normals[_sampled(window, width, keep, seed, i)])
        moments += own
        inputs.append({"name": Path(path).name, "cell_size_m": cells[i], "count": own.count})
    basis = basis_from_moments(moments, azimuths, altitudes)
    sample = np.concatenate(sample).reshape(-1, 3)
    sample = sample[~np.isnan(sample).any(axis=1)]
    if len(sample) == 0:
        raise ValueError("the stretch sample holds no valid cells; increase sample_size")
    low, high = stretch_limits(transform(sample, basis), stretch)
    return replace(basis, stretch_percentiles=tuple(float(p) for p in stretch),
                   stretch_low=low, stretch_high=high, inputs=tuple(inputs))


def score_blocks(path, basis, *, z_units=None, block_size=1024):
    """Component scores of a DEM under a basis, block by block.

    Parameters
    ----------
    path : str or path-like
        Elevation raster readable by rasterio; band 1 is used.
    basis : Basis
        A fitted basis, from this raster or any other.
    z_units : {None, "m", "ft", "us_ft"}
        Unit of the elevation values. See `fit_basis`.
    block_size : int
        Block side [cells].

    Yields
    ------
    window : rasterio.windows.Window
        The block's place in the raster; blocks tile it exactly.
    scores : ndarray
        Shape ``(window.height, window.width, 3)``, dimensionless; NaN where invalid.
    """
    for window, normals in normal_blocks(path, z_units=z_units, block_size=block_size):
        yield window, transform(normals, basis)


MAX_CELL_SIZE_RATIO = 2.0
"""Largest ratio of input cell sizes `fit_basis` pools without being told to."""


def _check_resolution(paths, cells):
    """Raise if any east or north cell size differs by more than `MAX_CELL_SIZE_RATIO`."""
    sizes = np.array(cells)
    ratio = (sizes.max(axis=0) / sizes.min(axis=0)).max()
    if ratio > MAX_CELL_SIZE_RATIO:
        listing = ", ".join(f"{Path(p).name}: {e:.3g} x {n:.3g} m" for p, (e, n) in zip(paths, cells))
        raise ValueError(f"input cell sizes differ by {ratio:.3g}x ({listing}); slope statistics "
                         "depend on resolution, so pass allow_mixed_resolution=True to pool them anyway")


_GOLDEN = 0x9E3779B97F4A7C15


def _sampled(window, width, keep, seed, raster):
    """Mask of the block's cells in the stretch sample: a seeded hash of each cell's index."""
    if keep >= 1:
        return np.ones((window.height, window.width), bool)
    rows = np.arange(window.row_off, window.row_off + window.height, dtype=np.uint64)
    cols = np.arange(window.col_off, window.col_off + window.width, dtype=np.uint64)
    key = np.uint64((seed * _GOLDEN + raster * (_GOLDEN >> 1)) % 2**64)
    x = rows[:, None] * np.uint64(width) + cols[None, :] + key
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)  # splitmix64 finalizer
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return (x >> np.uint64(11)) < np.uint64(int(keep * 2**53))
