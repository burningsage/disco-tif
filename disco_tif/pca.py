"""Hillshade PCA as a 3x3 problem on surface normals.

Unclipped hillshades are linear in the unit normal: a stack of k hillshades is
``H = U Sᵀ`` with ``U`` the normals (n x 3) and ``S`` the sun vectors (k x 3). The
covariance of the stack is ``S C Sᵀ`` with ``C`` the 3x3 covariance of the normals, so
the PCA of any number of hillshades has at most three components and needs only the
mean and covariance of the normals. Those add exactly across blocks and rasters.
"""

from dataclasses import dataclass, field

import numpy as np


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
