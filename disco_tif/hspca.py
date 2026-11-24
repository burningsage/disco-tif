"""Hillshade PCA and single hillshades of raster files, streamed block by block.

Pass 1 (`fit_basis`) adds up the moments of the unit normals of every block of
every input and keeps a seeded sample of cells for the stretch. Pass 2
(`score_blocks`, `apply_basis`) applies the basis block by block. Memory is set by
``block_size``, not by the size of the rasters.
"""

import logging
import os
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import numpy as np
import rasterio

from disco_tif.io import cell_size, create_geotiff
from disco_tif.normals import normal_blocks
from disco_tif.pca import (DEFAULT_ALTITUDES, DEFAULT_AZIMUTHS, Moments, basis_from_moments, stretch_limits,
                           to_uint8, transform)
from disco_tif.shading import shade

log = logging.getLogger(__name__)

OUTPUTS = ("components", "rgb")
"""Products `apply_basis` can write: one file per component, and an RGB composite."""


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
    log.info("fitted a basis to %d raster(s), %d valid cells; explained %s", len(paths), moments.count,
             " / ".join(f"{100 * x:.1f} %" for x in basis.explained))
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


def apply_basis(src_path, basis, out_dir, *, outputs=OUTPUTS, dtype="uint8", z_units=None, block_size=1024):
    """Write the hillshade PCA products of a DEM under a basis.

    Files are named after the DEM, ``<stem>_hspca_c1.tif`` to ``_c3.tif``,
    ``<stem>_hspca_rgb.tif`` and ``<stem>_hspca_basis.json``, and reruns overwrite
    them. The basis JSON is always written, so each product records how it was made.

    Parameters
    ----------
    src_path : str or path-like
        Elevation raster readable by rasterio; band 1 is used.
    basis : Basis
        A basis from `fit_basis`, fitted to this DEM or to others (for products
        that match an earlier delivery).
    out_dir : str or path-like
        Output folder, created if missing.
    outputs : sequence of {"components", "rgb"}
        Products to write. The RGB composite holds components 1, 2 and 3 as red,
        green and blue, always as uint8.
    dtype : {"uint8", "float32"}
        Component data type. uint8 is the basis's stretch to 1-255 with 0 for
        nodata; float32 is the raw score (dimensionless) with NaN for nodata.
    z_units : {None, "m", "ft", "us_ft"}
        Unit of the elevation values. See `fit_basis`.
    block_size : int
        Block side [cells].

    Returns
    -------
    dict of str to pathlib.Path
        Keys ``"c1"``, ``"c2"``, ``"c3"``, ``"rgb"`` for the products written, and
        ``"basis"``.

    Raises
    ------
    ValueError
        For an unknown output or dtype, or a uint8 product from a basis without
        stretch limits.
    """
    if set(outputs) - set(OUTPUTS):
        raise ValueError(f"outputs must be drawn from {OUTPUTS}, got {tuple(outputs)}")
    if dtype not in ("uint8", "float32"):
        raise ValueError(f"dtype must be 'uint8' or 'float32', got {dtype!r}")
    stretched = "rgb" in outputs or ("components" in outputs and dtype == "uint8")
    if stretched and basis.stretch_low is None:
        raise ValueError("the basis has no stretch limits; fit it with fit_basis, or write float32 components")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(src_path).stem
    paths = {}
    with ExitStack() as stack:
        components = []
        if "components" in outputs:
            for j in (1, 2, 3):
                paths[f"c{j}"] = out_dir / f"{stem}_hspca_c{j}.tif"
                components.append(stack.enter_context(create_geotiff(
                    paths[f"c{j}"], src_path, dtype=dtype, nodata=0 if dtype == "uint8" else np.nan)))
        rgb = None
        if "rgb" in outputs:
            paths["rgb"] = out_dir / f"{stem}_hspca_rgb.tif"
            rgb = stack.enter_context(create_geotiff(paths["rgb"], src_path, count=3, photometric="RGB"))
        for window, scores in score_blocks(src_path, basis, z_units=z_units, block_size=block_size):
            u8 = to_uint8(scores, basis.stretch_low, basis.stretch_high) if stretched else None
            for j, dst in enumerate(components):
                band = u8[..., j] if dtype == "uint8" else scores[..., j].astype(np.float32)
                dst.write(band, 1, window=window)
            if rgb is not None:
                rgb.write(np.moveaxis(u8, -1, 0), window=window)
    paths["basis"] = basis.save(out_dir / f"{stem}_hspca_basis.json")
    log.info("wrote %s", ", ".join(p.name for p in paths.values()))
    return paths


def hillshade_pca(src_path, out_dir, *, azimuths=DEFAULT_AZIMUTHS, altitudes=DEFAULT_ALTITUDES, z_units=None,
                  block_size=1024, sample_size=1_000_000, stretch=(1, 99), seed=0, outputs=OUTPUTS,
                  dtype="uint8"):
    """Fit a hillshade PCA to one DEM and write its products: `fit_basis`, then `apply_basis`.

    To reuse a basis across rasters, fit it once with `fit_basis` (or load it with
    `Basis.load`) and call `apply_basis` for each raster instead.

    Parameters
    ----------
    src_path : str or path-like
        Elevation raster readable by rasterio; band 1 is used.
    out_dir : str or path-like
        Output folder, created if missing.
    azimuths, altitudes, z_units, block_size, sample_size, stretch, seed
        As for `fit_basis`.
    outputs, dtype
        As for `apply_basis`.

    Returns
    -------
    dict of str to pathlib.Path
        As for `apply_basis`.
    """
    basis = fit_basis(src_path, azimuths=azimuths, altitudes=altitudes, z_units=z_units, block_size=block_size,
                      sample_size=sample_size, stretch=stretch, seed=seed)
    return apply_basis(src_path, basis, out_dir, outputs=outputs, dtype=dtype, z_units=z_units,
                       block_size=block_size)


def hillshade(src_path, out_path, *, azimuth=315, altitude=45, z_units=None, block_size=1024, dtype="uint8"):
    """Write a single hillshade of a DEM, using its real cell spacing and units.

    Parameters
    ----------
    src_path : str or path-like
        Elevation raster readable by rasterio; band 1 is used.
    out_path : str or path-like
        Output GeoTIFF, overwritten if it exists.
    azimuth : float
        Sun azimuth [degrees clockwise from north].
    altitude : float
        Sun altitude [degrees above the horizon].
    z_units : {None, "m", "ft", "us_ft"}
        Unit of the elevation values. See `fit_basis`.
    block_size : int
        Block side [cells].
    dtype : {"uint8", "float32"}
        uint8 follows gdaldem: slopes facing away from the sun are 1, full sun is
        255, nodata is 0. float32 is the unclipped shading, cos of the angle between
        normal and sun (-1 to 1, dimensionless), with NaN for nodata.

    Returns
    -------
    pathlib.Path
    """
    if dtype not in ("uint8", "float32"):
        raise ValueError(f"dtype must be 'uint8' or 'float32', got {dtype!r}")
    with create_geotiff(out_path, src_path, dtype=dtype, nodata=0 if dtype == "uint8" else np.nan) as dst:
        for window, normals in normal_blocks(src_path, z_units=z_units, block_size=block_size):
            shading = shade(normals, azimuth, altitude)
            dst.write(to_uint8(shading, 0, 1) if dtype == "uint8" else shading.astype(np.float32), 1, window=window)
    log.info("wrote %s", Path(out_path).name)
    return Path(out_path)


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
