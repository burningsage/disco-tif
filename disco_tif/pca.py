"""Hillshade PCA as a 3x3 problem on surface normals.

Unclipped hillshades are linear in the unit normal: a stack of k hillshades is
``H = U Sᵀ`` with ``U`` the normals (n x 3) and ``S`` the sun vectors (k x 3). The
covariance of the stack is ``S C Sᵀ`` with ``C`` the 3x3 covariance of the normals, so
the PCA of any number of hillshades has at most three components and needs only the
mean and covariance of the normals. Those add exactly across blocks and rasters.

Sign convention: PCA signs are arbitrary, so they are fixed after the fit. Component 1
is flipped to correlate positively with a hillshade lit from `REFERENCE_SUN` (it reads
like an ordinary hillshade). Components 2 and 3 are flipped so that the largest
(east, north, up) entry of their weight vector is positive.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from disco_tif.hillshade import sun_vector

log = logging.getLogger(__name__)

DEFAULT_AZIMUTHS = (0, 45, 90, 135, 180, 225, 270, 315)
"""Default sun azimuths [degrees clockwise from north]: eight, evenly spaced."""

DEFAULT_ALTITUDES = (30, 60)
"""Default sun altitudes [degrees above the horizon]."""

REFERENCE_SUN = (315, 45)
"""Azimuth and altitude [degrees] of the hillshade component 1 is signed against."""

NEAR_TIE_RATIO = 0.9
"""Adjacent eigenvalues closer than this ratio trigger a warning (within 10 %)."""


@dataclass
class Moments:
    """Running count, sum and sum of outer products of unit normals.

    Moments from different blocks or rasters combine exactly with ``+``.
    """

    count: int = 0
    total: np.ndarray = field(default_factory=lambda: np.zeros(3))
    outer: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))

    @classmethod
    def of(cls, normals):
        """Moments of the valid normals in an array.

        Parameters
        ----------
        normals : ndarray
            Shape ``(..., 3)``, (east, north, up). Rows containing NaN are skipped.

        Returns
        -------
        Moments
        """
        u = np.asarray(normals, dtype=float).reshape(-1, 3)
        u = u[~np.isnan(u).any(axis=1)]
        return cls(count=len(u), total=u.sum(axis=0), outer=u.T @ u)

    def __add__(self, other):
        return Moments(self.count + other.count, self.total + other.total, self.outer + other.outer)

    @property
    def mean(self):
        """Mean normal, shape ``(3,)``."""
        return self.total / self.count

    @property
    def cov(self):
        """Sample covariance of the normals (``ddof=1``, as sklearn uses), shape ``(3, 3)``."""
        if self.count < 2:
            raise ValueError(f"need at least 2 valid cells for a covariance, got {self.count}")
        m = self.mean
        return (self.outer - self.count * np.outer(m, m)) / (self.count - 1)


def sun_matrix(azimuths, altitudes):
    """Sun vectors for every azimuth-altitude combination.

    Parameters
    ----------
    azimuths : sequence of float
        Sun azimuths [degrees clockwise from north].
    altitudes : sequence of float
        Sun altitudes [degrees above the horizon].

    Returns
    -------
    ndarray
        Shape ``(len(azimuths) * len(altitudes), 3)``, (east, north, up), ordered
        azimuth-major.

    Raises
    ------
    ValueError
        If the sun vectors do not span 3-D space, so a 3-component PCA is undefined
        (for example fewer than 3 directions, or every sun at the zenith).
    """
    az, alt = np.meshgrid(np.asarray(azimuths, float), np.asarray(altitudes, float), indexing="ij")
    s = sun_vector(az.ravel(), alt.ravel())
    if np.linalg.matrix_rank(s, tol=1e-9) < 3:
        raise ValueError("the sun directions must span 3-D space: use at least 3 azimuths "
                         "below 90 degrees altitude, not all in one vertical plane")
    return s


@dataclass
class Basis:
    """A fitted hillshade PCA, ready to apply to normals from any raster.

    Attributes
    ----------
    mean : ndarray
        Mean unit normal the scores are centered on, shape ``(3,)``.
    weights : ndarray
        Shape ``(3, 3)``; column ``j`` maps a centered normal to the score of
        component ``j``. Scores equal those of a PCA on the explicit hillshade stack
        (shading on the -1 to 1 scale).
    eigenvalues : ndarray
        Variance of each component's scores, shape ``(3,)``, largest first.
    explained : ndarray
        Fraction of the hillshade-stack variance in each component, shape ``(3,)``;
        sums to 1 because the stack has rank 3.
    azimuths, altitudes : tuple of float
        Sun angles [degrees] the basis was fitted with.
    """

    mean: np.ndarray
    weights: np.ndarray
    eigenvalues: np.ndarray
    explained: np.ndarray
    azimuths: tuple
    altitudes: tuple


def basis_from_moments(moments, azimuths=DEFAULT_AZIMUTHS, altitudes=DEFAULT_ALTITUDES):
    """Fit the hillshade PCA from the moments of the normals.

    Parameters
    ----------
    moments : Moments
        Moments of the valid normals of one or more rasters.
    azimuths : sequence of float
        Sun azimuths [degrees clockwise from north].
    altitudes : sequence of float
        Sun altitudes [degrees above the horizon].

    Returns
    -------
    Basis
    """
    s = sun_matrix(azimuths, altitudes)
    cov = moments.cov
    stack_cov = s @ cov @ s.T
    eigenvalues, vectors = np.linalg.eigh(stack_cov)
    top = np.argsort(eigenvalues)[::-1][:3]
    eigenvalues, vectors = eigenvalues[top], vectors[:, top]
    weights = _fix_signs(s.T @ vectors, cov)
    _warn_near_ties(eigenvalues)
    return Basis(mean=moments.mean, weights=weights, eigenvalues=eigenvalues,
                 explained=eigenvalues / np.trace(stack_cov),
                 azimuths=tuple(float(a) for a in azimuths),
                 altitudes=tuple(float(a) for a in altitudes))


def transform(normals, basis):
    """Component scores of unit normals under a fitted basis.

    Parameters
    ----------
    normals : ndarray
        Shape ``(..., 3)``, (east, north, up). NaN normals give NaN scores.
    basis : Basis

    Returns
    -------
    ndarray
        Scores, shape ``(..., 3)``: component 1, 2, 3 in the last axis, on the
        hillshade scale (dimensionless).
    """
    return (np.asarray(normals, dtype=float) - basis.mean) @ basis.weights


def _fix_signs(weights, cov):
    """Apply the module's sign convention to the three weight columns."""
    weights = weights.copy()
    reference = sun_vector(*REFERENCE_SUN)
    if weights[:, 0] @ cov @ reference < 0:  # covariance of score 1 with the reference shade
        weights[:, 0] *= -1
    for j in (1, 2):
        if weights[np.argmax(np.abs(weights[:, j])), j] < 0:
            weights[:, j] *= -1
    return weights


def _warn_near_ties(eigenvalues):
    """Log a warning when adjacent components have nearly equal variance."""
    for j in (0, 1):
        if eigenvalues[j] > 0 and eigenvalues[j + 1] / eigenvalues[j] > NEAR_TIE_RATIO:
            log.warning("components %d and %d have nearly equal variance (%.3g vs %.3g): their "
                        "orientation is set by small terrain anisotropy and can change between "
                        "datasets; use one pooled or saved basis for consistent products",
                        j + 1, j + 2, eigenvalues[j], eigenvalues[j + 1])


def stretch_limits(scores, percentiles=(1, 99)):
    """Low and high stretch limits per component, from percentiles of the valid scores.

    Parameters
    ----------
    scores : ndarray
        Shape ``(..., 3)``; NaN cells are ignored.
    percentiles : tuple of float
        Lower and upper percentile [0-100] that map to the ends of the output range.

    Returns
    -------
    low, high : ndarray
        Shape ``(3,)`` each, in score units (dimensionless).
    """
    flat = np.asarray(scores, dtype=float).reshape(-1, 3)
    low, high = np.nanpercentile(flat, percentiles, axis=0)
    return low, high


def to_uint8(values, low, high):
    """Linear stretch to uint8 with 0 reserved for nodata.

    Values at or below `low` map to 1, at or above `high` to 255; NaN maps to 0. This
    is the disco-tif uint8 convention: write the result with ``nodata=0``.

    Parameters
    ----------
    values : ndarray
        Any shape; NaN marks nodata.
    low, high : float or ndarray
        Stretch limits, broadcast against `values` (one per component for scores of
        shape ``(..., 3)``).

    Returns
    -------
    ndarray
        uint8, same shape as `values`.

    Raises
    ------
    ValueError
        If any `high` is not greater than the matching `low`.
    """
    low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
    if np.any(high <= low):
        raise ValueError(f"stretch high must exceed low, got low={low}, high={high}")
    values = np.asarray(values, dtype=float)
    scaled = 1 + np.round(254 * (np.clip(values, low, high) - low) / (high - low))
    return np.where(np.isnan(values), 0, scaled).astype(np.uint8)
