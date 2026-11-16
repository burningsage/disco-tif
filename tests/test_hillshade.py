"""Tests for sun vectors and shading."""

import numpy as np
import pytest

from disco_tif.hillshade import shade, sun_vector


@pytest.mark.parametrize(
    "azimuth, expected",
    [(0, [0, 1, 0]), (90, [1, 0, 0]), (180, [0, -1, 0]), (270, [-1, 0, 0])],
)
def test_sun_vector_compass_points(azimuth, expected):
    """Azimuth is clockwise from north: 0 north, 90 east, 180 south, 270 west."""
    np.testing.assert_allclose(sun_vector(azimuth, 0), expected, atol=1e-12)


def test_sun_vector_is_unit_and_zenith_at_90():
    v = sun_vector([0, 45, 200], [10, 45, 80])
    np.testing.assert_allclose(np.linalg.norm(v, axis=-1), 1.0)
    np.testing.assert_allclose(sun_vector(123, 90), [0, 0, 1], atol=1e-12)


def test_sun_vector_rejects_bad_altitude():
    with pytest.raises(ValueError):
        sun_vector(0, 95)


def test_shade_is_unclipped_and_keeps_nan():
    """A face turned away from the sun is negative, not zero; NaN normals stay NaN."""
    normals = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [np.nan] * 3])
    out = shade(normals, azimuth=90, altitude=0)
    np.testing.assert_allclose(out[:2], [0.0, -1.0], atol=1e-12)
    assert np.isnan(out[2])
