# disco-tif notebooks
<!-- auto-generated 2026-11-16, updated 2026-11-16 -->

Both notebooks drive the 0.0.x API, which the 0.1.0 rewrite removes (see `plans/hillshade_pca_rewrite.md`). They will be replaced by one thin example notebook in Phase 6 (burningsage/disco-tif#7). Until then they run only against `disco-tif<0.1` with its old dependencies (earthpy, pandas, seaborn, scikit-learn).

---

### `example_notebooks/process_tiff_from_functions.ipynb`
- **Role:** visualization
- **Purpose:** Step-by-step walkthrough of every 0.0.x function on one DTM. It shows the EMerald 8-color palette, computes pseudo-hist-norm and pseudo-linear color breaks with histograms, writes every LUT flavor (QGIS color table, single- and multi-channel RGBA LUTs, data-to-uint8 and uint8 LUTs), writes paletted single-band and 4-band RGBA GeoTIFFs, builds hillshades for a grid of azimuths and altitudes with their PCA components, and writes a MapServer mapfile. Each step plots its output.
- **Key Imports:** `os`, `datetime`, `numpy`, `pandas`, `rasterio`, `matplotlib`, `seaborn`, `disco_tif.process_sb_tiff`, `disco_tif.hillshades_pca`, `disco_tif.geotiff_plotting`, `disco_tif.write_mapfile` (`disco_tif.look_up_tables` is used but reached through `process_sb_tiff`, not imported)
- **Configurable Parameters:** `datarange` (data min/max for the color stretch), `cmap_method` (`pseudo_hist_norm` / `pseudo_linear`), `lut_no_data_value`, `write_short_files`, `output_tif`, `output_QGIS_lut` / `output_rgba_lut` / `output_uint8_lut`, `min_max_method`, `clip_perc`, `color_palette_name`, hillshade azimuth and altitude settings for `build_hs_az_al`, `plot_rgba_raster`, `plot_histograms`
- **Data I/O:**
  - In: `../example_data/USGS_LiDAR/USGS_LiDAR_1.tif` (alternative `_2.tif`), single-band GeoTIFF DTM
  - Out: written beside the input: color LUTs (`.lut`, `.txt`), colorized GeoTIFFs, hillshade and PCA-component uint8 GeoTIFFs, MapServer `.map`
- **Known Input Formats:** primary format only (GeoTIFF). The GeoWrangler copy has been used with ESRI Float `.flt`.
- **Environment:** Unknown (last run in 2024-2025, before the venv naming convention; not runnable in `GeoWrangler_py312_20261102_v0.7.0.dev0`, which has no earthpy or disco_tif)
- **TODOs / FIXMEs:** None found. Hard-coded `datarange` test values are left in a commented list in one cell.
- **Proposed Rename:** None; superseded in Phase 6.
- **GeoWrangler Match:** `GeoWrangler/example_usage_notebooks/raster_processing_notebooks/process_tiff_from_functions.ipynb` (a later, edited copy)
- **Customizations vs. GeoWrangler:**
  - Uses the EMerald palette; the GeoWrangler copy builds a custom `terrain_1_hex` palette in notebook cells with blend and opacity settings
  - Reads the example USGS LiDAR DTM; the GeoWrangler copy reads a hard-coded project `.flt`
- **GeoWrangler Upstream Notes:** None. The direction is the reverse: GeoWrangler will depend on disco-tif 0.1 and its copy gets updated to the new API in Phase 6.

### `example_notebooks/streamlined_example_usage.ipynb`
- **Role:** visualization
- **Purpose:** Short automation example. One call to `make_rgba_tiff_from_single_Band` writes the colorized GeoTIFF and QGIS color table, then `build_hs_az_al` and `build_hillshade` write hillshades and their PCA components, then a mapfile is written.
- **Key Imports:** `os`, `datetime`, `numpy`, `pandas`, `rasterio`, `matplotlib`, `seaborn`, `disco_tif.process_sb_tiff`, `disco_tif.hillshades_pca`, `disco_tif.write_mapfile`
- **Configurable Parameters:** colorizing options of `make_rgba_tiff_from_single_Band` (`data_min_max`, `min_max_method`, `clip_perc`, `cmap_method`, `output_tif`), hillshade azimuth and altitude settings, `process_pca`
- **Data I/O:**
  - In: `../example_data/E020N40/E020N40.tif`, single-band GeoTIFF DEM. **That file is no longer on disk**: it was in the retired EMeraldCodes clone, now in the macOS Trash. Point the notebook at `USGS_LiDAR_1.tif` to run it.
  - Out: colorized GeoTIFF, QGIS color table, hillshade and PCA uint8 GeoTIFFs and a mapfile, written beside the input
- **Known Input Formats:** primary format only (GeoTIFF)
- **Environment:** Unknown; the saved kernel is `testing_disco_tif_py3.10`, which no longer exists among the venvs
- **TODOs / FIXMEs:** None found
- **Proposed Rename:** None; superseded in Phase 6.
- **GeoWrangler Match:** None - novel
- **Customizations vs. GeoWrangler:** Not applicable
- **GeoWrangler Upstream Notes:** None
