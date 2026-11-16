# History

## 0.1.0 (unreleased)

* Rewrite focused on principal-component hillshades: closed-form 3x3 PCA, streaming over raster blocks, cell-size and CRS aware gradients, and one basis shared across many rasters. See `plans/hillshade_pca_rewrite.md`.
* Development moves to https://github.com/burningsage/disco-tif.
* Packaging moves from setup.py to pyproject.toml. Dependencies drop to numpy and rasterio; earthpy, pandas, seaborn and scikit-learn are no longer required.
* Breaking: the 0.0.x functions are removed. Pin `disco-tif<0.1` to keep them.

## 0.0.3 (2024-05-14)

* Last release from emerald-geomodelling/disco-tif.

## 0.0.2 (2024-04-17)

* Updated so that it works as a licensed, stand-alone library.

## 0.0.1 (2024-04-17)

* Initial release.
