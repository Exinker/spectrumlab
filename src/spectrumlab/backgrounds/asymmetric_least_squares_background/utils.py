from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from spectrumlab.types import Array, R

if TYPE_CHECKING:
    from spectrumlab.backgrounds.asymmetric_least_squares_background.model import AsymmetricLeastSquaresBackgroundConfig


def estimate_background(
    intensity: Array[R],
    mask: Array[bool],
    config: 'AsymmetricLeastSquaresBackgroundConfig',

):
    n = len(intensity)
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n))

    w = np.ones(n)
    for _ in range(config.n_iters):
        W = sparse.diags(w, 0, shape=(n, n))
        z = spsolve((W + config.lam*np.dot(D.T, D)).tocsr(), w*intensity)

        # w = np.where(mask, True, config.p*(intensity > z) + (1 - config.p)*(intensity < z))
        w = np.where(intensity > z, config.p, 1 - config.p)
        w[mask] = 1.0

    return z
