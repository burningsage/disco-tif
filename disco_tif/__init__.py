"""Principal-component hillshades for digital elevation models."""

from disco_tif._version import __version__
from disco_tif.hspca import apply_basis, fit_basis, hillshade, hillshade_pca
from disco_tif.pca import Basis

__all__ = ["Basis", "__version__", "apply_basis", "fit_basis", "hillshade", "hillshade_pca"]
