# disco-tif rewrite plan: hillshade PCA first

Drafted 2026-11-16. Status: decisions settled 2026-11-16 (see the end), Phase 0 under way.

## Goal

Turn disco-tif into a small, tested, pip-installable library whose main job is the hillshade PCA, and make it work on rasters of any size, any resolution and any CRS. Rasters split into tiles should produce matching results across the seams. Release it from the burningsage fork as 0.1.0 on the existing `disco-tif` PyPI project. GeoWrangler then depends on it and wraps it as one pipeline step.

## What we know going in

### Prior art

PCA of multi-direction hillshades is published. Devereux, Amable and Crow (2008, Antiquity 82:470-479) introduced it for archaeological LiDAR. RVT / `rvt-py` ships "PCA of hillshading", and GRASS has the `r.shaded.pca` add-on. RVT's own manual says the PCA option does not work with large files and gives inconsistent results between datasets. Those two gaps are what this rewrite fixes, and the README should credit the prior art and say plainly what is different here.

### The PCA reduces to a 3x3 problem

earthpy's `hillshade` is unclipped Lambertian shading, so every hillshade is a linear combination of the same three per-pixel terms from the surface normal: cos(slope), sin(slope)·cos(aspect) and sin(slope)·sin(aspect). Stacking N hillshades is therefore `H = U Aᵀ`, where `U` is the pixel-by-3 matrix of normal terms and `A` is the N-by-3 matrix of sun vectors. The covariance of the stack is `A C_u Aᵀ`, with rank at most 3.

This was checked on a synthetic 300x300 DEM with 24 hillshades (8 azimuths by 3 altitudes). sklearn's full PCA gave exactly three non-zero components. A 3x3 calculation on the normals reproduced the reconstruction to 1e-13 and gave the same eigenvalues, and its component images correlated with sklearn's at 1.000. The script was a throwaway in the session scratchpad; Phase 2 turns it into a real test.

Consequences:

- The hillshade stack never needs to exist. The azimuths and altitudes only set a fixed 3x3 weighting, `AᵀA`.
- The fit needs one pass that accumulates a mean (3 values) and a covariance (3x3) over the pixels. These sums add exactly across blocks and across rasters, so there is no sampling error and no size limit.
- The 24-hillshade cap, the `n_features >= 4` rule and the `n_components < n_hillshades` rule all go away. There are always exactly 3 components.
- For evenly spaced azimuths (3 or more), `AᵀA` is diagonal and the same whatever the count, so the result should depend only on the altitudes. Uneven azimuth sets give directional emphasis. This needs a test before the docs claim it.
- The reduction holds only for unclipped shading. Clipped shading (negative illumination set to 0) or cast shadows are nonlinear, and the shortcut no longer equals the explicit PCA. See open decision 5.

### Bugs and limits in the current code (Dev, 272a1cb)

- A raster with no nodata set breaks `make_rgba_tiff_from_single_Band`: `np.array(None, dtype)` gives NaN for float rasters and a `TypeError` for int rasters (`process_sb_tiff.py:464`).
- NaN nodata is never masked, because `data != nan` is always True. NaN is the GeoTIFF nodata convention for our work.
- earthpy's hillshade uses `np.gradient(arr)` with unit spacing. Slopes are wrong unless the cell size equals 1 in z units, and very wrong for geographic CRSs. earthpy also names the row-direction gradient `x`, so the sun azimuth convention needs checking against a known case.
- `write_mapfile` searches for `'hillshade'` in filenames, but the writer names files `_HS_...`, so it finds nothing.
- A named matplotlib colormap reaches only the plot and the 4-band output. The QGIS LUT and the paletted tiff are hard-coded to the EMerald colors.
- `assert data_min_max[1] > 0` rejects all-negative data such as bathymetry. Integer rounding of min/max and breaks collapses small-range float data.
- Each PCA component is stretched to its own min/max, so outliers flatten the contrast. Component signs are arbitrary and can flip between runs.
- Output filenames carry `stoday()`, so reruns pile up files instead of replacing them.
- `setup.py` pins `numpy==1.26.4` and `rasterio==1.3.10`, so it does not install beside numpy 2 or on Python 3.12.
- Library functions call `print` and `plt.show()`. There is a circular import between `process_sb_tiff` and `look_up_tables`, and the datetime helpers are copied into three modules.

## Scope

### Keep, rewritten

- Hillshade PCA, the core of the package.
- Single hillshades as a by-product, since they are cheap once the normals exist.
- Colorizing in `colorize.py`: the paletted single-band uint8 GeoTIFF writer, which is compact and displays in every GIS, and the pseudo-hist-norm and pseudo-linear breaks. It takes a matplotlib colormap name or any `matplotlib.colors.Colormap` object. Color files such as Geosoft `.tbl` or `.clr` are loaded by the caller, for example with GeoWrangler's `tbl_to_mpl_cmap`, and the resulting object is passed in. disco-tif never imports GeoWrangler.

### Drop

- The 8-color EMerald palette and its name.
- The six LUT flavors and the MapServer mapfile writers, which were built for the old map-server hosting.
- The plotting helpers, apart from one optional quick-look function behind a `[plot]` extra.
- The dependencies on earthpy, seaborn, pandas and scikit-learn. earthpy is dropped because we would use only its hillshade, and that formula is about ten lines. The parts we need to change are exactly the parts it gets wrong: cell size and the axis convention. Its formula is kept as a reference inside a test, so at unit spacing the new code must reproduce the current outputs.

### Not in v1

- Clipped and shadowed hillshades.
- A YAML pipeline runner inside disco-tif. The library stays as plain functions, and GeoWrangler owns the pipeline step.

## Design

### Package layout

```
disco_tif/
  __init__.py      # public API and __version__
  normals.py       # unit normals from gradients with real spacing; dem_normals(path)
  pca.py           # moments, basis fit, transform, sign convention, stretch
  io.py            # cell spacing from CRS, vertical units, reading to meters/NaN; later writers, basis JSON
  hillshade.py     # single hillshades from normals
  cli.py           # disco-tif hspca / fit-basis / apply-basis
  colorize.py      # paletted uint8 GeoTIFF + hist-norm / linear breaks, any mpl Colormap
tests/
pyproject.toml
```

### Public API (first cut)

Every function gets a NumPy-style docstring with units on every physical parameter.

```python
hillshade_pca(src_path, out_dir, *, azimuths=(0, 45, ..., 315), altitudes=(30, 60),
              z_units=None, block_size=1024, stretch=(1, 99), outputs=("components", "rgb"),
              basis=None) -> dict[str, Path]

fit_basis(src_paths, *, azimuths, altitudes, z_units=None, block_size=1024,
          sample_size=1_000_000, stretch=(1, 99), seed=0) -> Basis

apply_basis(src_path, basis, out_dir, *, outputs=("components", "rgb")) -> dict[str, Path]

hillshade(src_path, out_path, *, azimuth=315, altitude=45, z_units=None) -> Path
```

`hillshade_pca` is just `fit_basis([src])` followed by `apply_basis`. `Basis` is a small dataclass that saves to and loads from JSON. It holds the mean, the 3x3 eigenvectors and eigenvalues, the sun angles, the stretch limits, the per-input cell sizes and pixel counts, and the disco-tif version. That file records exactly how a product was made, and it can be applied later to new tiles to match an earlier delivery.

### Gradient, units and CRS

- Gradients use the real spacing: `np.gradient(z, dy, dx)` with signs taken from the geotransform (north-up rasters have negative dy).
- Geographic CRS: convert degrees to meters per row, using cos(latitude) for x. This is good enough for visualization, and the docs should say it is an approximation.
- Vertical units: if `z_units` is None, assume z is in the horizontal CRS units. A geographic CRS has no horizontal length unit, so `z_units` is then required and the call raises an error without it. Supported values are `"m"`, `"ft"` and `"us_ft"`. The horizontal unit always comes from the CRS, never from a parameter.
- Nodata: masks come from the profile, either a sentinel value or NaN. Gradient cells that touch nodata become nodata, as they do now.
- Azimuth convention: degrees clockwise from north and altitude in degrees above the horizon, the same as GDAL and ESRI. An analytic test pins this down (Phase 1).

### Streaming

- Read in rasterio windows of `block_size` with a one-pixel halo, so block edges get correct gradients.
- Pass 1, the fit: for each block, accumulate the count, the sum of U (3 values) and the sum of UUᵀ (3x3) in float64. Keep a seeded random sample of up to `sample_size` normal rows (24 MB at one million rows), which is used only for the stretch.
- Fit: compute the 3x3 eigenproblem, apply the sign convention, transform the sample, and take the stretch percentiles from it.
- Pass 2, the write: transform each block and write it.
- Peak memory is set by `block_size`, not by the raster size.

### Multi-raster and tile consistency

- `fit_basis` takes many rasters and simply adds their sums together. The result is exactly the basis of the mosaic, without building the mosaic.
- Slope statistics change with resolution, so mixing 1 m and 30 m tiles in one basis is a real choice. `fit_basis` raises an error when the cell sizes of its inputs differ by more than 2x, unless `allow_mixed_resolution=True` is passed.

### Sign convention

- Flip PC1 so that it correlates positively with a reference hillshade (azimuth 315, altitude 45). Flip PC2 and PC3 so that the largest (east, north, up) entry of their weight vector in normal space is positive. The first draft said "largest loading over the hillshades", but with evenly spaced azimuths those loadings come in equal-magnitude symmetric pairs, so the choice would rest on rounding.
- The rule goes in the docs and in the basis JSON, so the colors of a product are reproducible.
- Near-tied eigenvalues. The first real run (a 30 m UTM DTM, 2026-11-16: 21.9 M pixels, 8 azimuths at 30 and 60 degrees) gave PC1 48%, PC2 45% and PC3 7%. With evenly spaced azimuths the two horizontal components are nearly degenerate, so how they are oriented in the horizontal plane is set by the terrain's anisotropy and can rotate between datasets. This is a likely cause of the "inconsistent between datasets" behavior that RVT reports. The basis JSON records the eigenvalue ratios, `fit_basis` logs a warning when PC1/PC2 are within 10%, and the docs explain that a pooled or saved basis is the way to get consistent products. A data-independent option, a fixed horizontal orientation such as north-south and east-west, is worth testing in Phase 2.
- That run took 6 s and 4.4 GB peak memory in memory, which confirms that streaming (Phase 3) is needed for larger rasters.

### Outputs

- Per component: a uint8 GeoTIFF by default, with 1-255 for data and 0 for nodata, which matches the current convention. float32 with NaN nodata is an option.
- `rgb`: a 3-band uint8 composite of PC1, PC2 and PC3. This is the default display in RVT and GRASS.
- Tiled, deflate-compressed GeoTIFF.
- Deterministic names, `<stem>_hspca_c1.tif`, `<stem>_hspca_rgb.tif` and `<stem>_hspca_basis.json`. Reruns overwrite; there are no date stamps.
- Logging through `logging`; no prints and no `plt.show()`.

## Phases

Each phase is one GitHub issue on the fork.

### Phase 0: housekeeping

- Enable Issues on `burningsage/disco-tif`, open the phase issues and add them to the project board.
- Merge `Dev` into `master`, so master holds the last working 0.0.x code, then branch `rewrite` from `master`.
- Update LICENSE.md: keep "Copyright (c) 2024 EMerald Geomodelling" and add "Copyright (c) 2026 Benjamin Bloss". Still MIT.
- Replace `setup.py` with `pyproject.toml`: requires-python >= 3.10, dependencies numpy >= 1.24 and rasterio >= 1.3, a `[plot]` extra for matplotlib, and a `[test]` extra for pytest and scikit-learn. The `disco-tif` console script is added in Phase 4 together with `cli.py`, so no broken entry point ships before then.
- Point the project URLs at the fork, and update the author email.
- Add `Notebook_summary_read_me.md`.
- Add a GitHub Actions workflow that runs the tests on 3.10 to 3.13.

### Phase 1: normals

- `normals.py`, with gradients using the real spacing, geographic CRS support, vertical units and nodata.
- Tests:
  - A plane sloping down to the east, lit from azimuth 90, is bright; lit from 270, it is dark. This fixes the azimuth convention without needing GDAL.
  - The same DEM sampled at 1 m and at 10 m, each with its own cell size, gives the same normals.
  - A DEM in feet with `z_units="ft"` on a meter CRS matches the meter version.
  - Sentinel and NaN nodata are both masked. Int rasters with no nodata do not raise.

### Phase 2: in-memory closed-form PCA

- `pca.py` with the moments, the fit, the transform, the sign convention and the stretch.
- Tests:
  - The closed form matches sklearn's PCA of explicit hillshades on a synthetic DEM. sklearn is a test dependency only.
  - At unit spacing, hillshades from the new normals match earthpy's formula, which is copied into the test as a reference, after any agreed azimuth correction from Phase 1.
  - Evenly spaced azimuth counts of 4, 8 and 32 give the same components.
  - Signs are stable when pixel order is permuted or a different seed is used.

### Phase 3: streaming and basis JSON

- Two-pass windowed processing, the `Basis` save and load, and multi-raster `fit_basis`.
- Tests:
  - Streaming with block_size 64 equals the in-memory result on the same raster.
  - Splitting a raster into four tiles and fitting a pooled basis gives the same pixels as the whole raster.
  - The basis JSON round-trips.

### Phase 4: writers and CLI

- The component and RGB writers, single `hillshade`, and the CLI (`disco-tif hspca`, `fit-basis`, `apply-basis`).
- Run on `example_data/USGS_LiDAR/USGS_LiDAR_1.tif` and compare against the old Dev outputs by eye. The only expected differences are the corrected slope scaling and the fixed signs.

### Phase 5: colorize

- `colorize.py` with the paletted uint8 GeoTIFF writer and the pseudo-hist-norm and pseudo-linear breaks. It takes an mpl colormap name or a `Colormap` object, and fixes the nodata, all-negative and float-range bugs.
- Test with a named mpl colormap, a `ListedColormap`, and a `.tbl` loaded through GeoWrangler in a GeoWrangler-side test.

### Phase 6: docs, release and GeoWrangler

- README rewrite covering what it does, the prior art, the 3x3 result, a quick start, and the CLI. Move the README images into the repo, because they are currently hosted on EMerald's GitHub attachments.
- One example notebook as a thin driver: a config cell, a `hillshade_pca` call, then figures.
- Tag 0.1.0 and publish to PyPI through trusted publishing from GitHub Actions. You set up the trusted publisher on pypi.org.
- GeoWrangler: add `disco-tif>=0.1` to its dependencies, add a hillshade-PCA pipeline step, and update `raster_processing_notebooks/process_tiff_from_functions.ipynb` and its summary entry. Its hard-coded project `.flt` filename moves to the config cell. Correct `docs/function_library.md`, which calls disco_tif "not publicly available".

## Decisions (2026-11-16)

1. **Old API: break it.** 0.1.0 removes `build_hillshade`, `MakeHillShadePCA`, `make_rgba_tiff_from_single_Band` and the rest, with no deprecated wrappers. pypistats shows about 110 downloads in the 180 days from 2026-05-18, likely mostly bots and mirrors. A GitHub code search finds no public code importing `disco_tif` outside the burningsage and EMerald repos. 0.0.3 stays on PyPI for anyone who pins it.
2. **Colorize: keep it in disco-tif** as `colorize.py`.
3. **Palettes: drop the EMerald palette.** Colorizing takes any matplotlib colormap name or `Colormap` object. Color files (`.tbl`, `.clr`) are loaded by the caller, for example with GeoWrangler, and passed in as objects, so disco-tif never imports GeoWrangler.
4. **Mixed resolutions: refuse unless allowed.** `fit_basis` raises an error when input cell sizes differ by more than 2x, unless `allow_mixed_resolution=True`.
5. **Clipped or shadowed hillshades: not in v1.** If they are wanted later, the route is to fit on explicit hillshades from a pixel sample, then transform block by block. That is approximate, unlike the 3x3 route.
6. **Output default: uint8**, with 0 as nodata. float32 with NaN is an option.
7. **License: keep the EMerald line and add yours.** "Copyright (c) 2024 EMerald Geomodelling" plus "Copyright (c) 2026 Benjamin Bloss", still MIT.
8. **Branches: merge Dev into master first**, then cut `rewrite` from master.
9. **Tracking: GitHub Issues on the fork**, one per phase, added to the project board. This plan is committed on the `rewrite` branch.
10. **earthpy: drop it.** Its hillshade is the only function used, and the new design needs real cell spacing and a clear axis convention, which it lacks. Its formula is kept as a reference in a test.
