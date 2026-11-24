"""Tests for the disco-tif command line."""

import numpy as np
import pytest
import rasterio

from disco_tif.cli import main
from disco_tif.pca import Basis
from rasters import synthetic_dem, write_tif


@pytest.fixture
def dems(tmp_path):
    return [write_tif(tmp_path / f"dem{i}.tif", synthetic_dem(seed=i), nodata=-9999) for i in (0, 1)]


def _read(path):
    with rasterio.open(path) as src:
        return src.read()


def test_hspca_writes_next_to_the_dem_by_default(dems):
    assert main(["hspca", str(dems[0])]) == 0
    folder = dems[0].parent
    for suffix in ("c1.tif", "c2.tif", "c3.tif", "rgb.tif", "basis.json"):
        assert (folder / f"dem0_hspca_{suffix}").exists()


def test_fit_basis_then_apply_basis_matches_the_python_api(dems, tmp_path):
    from disco_tif import apply_basis, fit_basis

    basis_path = tmp_path / "pooled.json"
    main(["fit-basis", *map(str, dems), "-o", str(basis_path), "--altitudes", "35", "55"])
    main(["apply-basis", str(basis_path), *map(str, dems), "--out-dir", str(tmp_path / "cli"), "--outputs", "rgb"])
    basis = fit_basis(dems, altitudes=(35, 55))
    np.testing.assert_array_equal(Basis.load(basis_path).weights, basis.weights)
    for dem in dems:
        expected = apply_basis(dem, basis, tmp_path / "api", outputs=("rgb",))["rgb"]
        np.testing.assert_array_equal(_read(tmp_path / "cli" / expected.name), _read(expected))
    assert not (tmp_path / "cli" / "dem0_hspca_c1.tif").exists()


def test_hillshade_command(dems, tmp_path):
    from disco_tif import hillshade

    main(["hillshade", str(dems[0]), str(tmp_path / "cli.tif"), "--azimuth", "200", "--altitude", "30"])
    hillshade(dems[0], tmp_path / "api.tif", azimuth=200, altitude=30)
    np.testing.assert_array_equal(_read(tmp_path / "cli.tif"), _read(tmp_path / "api.tif"))


def test_version_flag(capsys):
    from disco_tif import __version__

    with pytest.raises(SystemExit):
        main(["--version"])
    assert __version__ in capsys.readouterr().out
