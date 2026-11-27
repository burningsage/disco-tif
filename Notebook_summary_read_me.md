# disco-tif notebooks
<!-- auto-generated 2026-11-16, updated 2026-11-27 -->

The two 0.0.x notebooks (`process_tiff_from_functions.ipynb`, `streamlined_example_usage.ipynb`) were removed in Phase 6 (burningsage/disco-tif#7) along with the API they drove. They remain in the history and in the `release-0.0.3` tag, and run only against `disco-tif<0.1`.

---

### `example_notebooks/hillshade_pca_example.ipynb`
- **Role:** visualization
- **Purpose:** Thin driver for the 0.1.0 API. Downloads two adjacent 1 m USGS 3DEP LiDAR tiles if they are missing, fits one hillshade PCA basis to both with `fit_basis`, writes each tile's component, RGB and basis-JSON products with `apply_basis`, writes one single hillshade for comparison, and plots a crop of the hillshade, PC1, PC2 and the RGB composite. It ends with the GDAL VRT recipe for seamless products across tiles. The saved figure is the README image.
- **Key Imports:** `pathlib`, `urllib.request`, `matplotlib`, `rasterio`, `disco_tif`
- **Configurable Parameters:** `TILES` (local name to source URL), `AZIMUTHS`, `ALTITUDES`, `Z_UNITS`, `VIEW_TILE`, `VIEW` (figure crop in cells)
- **Data I/O:**
  - In: `../example_data/USGS_LiDAR/USGS_LiDAR_1.tif` and `_2.tif`, single-band float32 GeoTIFF DTMs in meters, EPSG:26913, downloaded from the USGS staged-products bucket on first run (about 240 MB each)
  - Out: `../example_data/USGS_LiDAR/hspca/`: `<tile>_hspca_c1/c2/c3.tif` (uint8), `<tile>_hspca_rgb.tif`, `<tile>_hspca_basis.json`, `USGS_LiDAR_2_hs315.tif`, `hspca_example.png`
- **Known Input Formats:** primary format only (GeoTIFF); the library reads anything rasterio opens, including a GDAL VRT mosaic
- **Environment:** a scratch Python 3.12 venv (numpy 2.5.5, rasterio 1.5.2, matplotlib), disco-tif 0.1.0.dev0 editable; last run 2026-11-27, 76 s for both tiles
- **TODOs / FIXMEs:** None found
- **Proposed Rename:** None
- **GeoWrangler Match:** None; GeoWrangler's `raster_processing_notebooks/process_tiff_from_functions.ipynb` is an edited copy of the removed 0.0.x notebook and gets moved to this API after the 0.1.0 release
- **Customizations vs. GeoWrangler:** Not applicable
- **GeoWrangler Upstream Notes:** None. The direction is the reverse: GeoWrangler will depend on disco-tif 0.1.
