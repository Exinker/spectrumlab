
from functools import partial

import numpy as np
from scipy import signal, optimize

from spectrumlab.alias import Array
from spectrumlab.utils import mse


def gauss(x: float | Array[float], x0: float, w: float) -> Array[float]:
    """Normal distribution with position x0 and unit intensity.

    Params:
        x0: position;
        w: width.
    """

    F = np.exp( -(1/2)*((x - x0) / w)**2 ) / ( np.sqrt(2*np.pi) * w )

    return F


def lanczos(x: float | Array[float], x0: float, a: int = 2) -> Array[float]:
    """Lanczos distribution with position x0
    
    Params:
        x0: position;
        a: window width.
    """

    F = np.sinc(x - x0) * np.sinc((x - x0) / a)
    F[(x - x0 < -a) | (x - x0 > a)] = 0

    return F


def rectangular(x: float | Array[float], x0: float, w: float) -> Array[float]:
    """Rectangular distribution with position x0 and unit intensity.

    Params:
        x0: position;
        w: width (full width).
    """

    if isinstance(x, float):  # TODO: don't remove! It's for integrate.quad functions!
        F = np.zeros(1,)
    else:
        F = np.zeros(x.shape)

    F[(x > x0 - w/2) & (x < x0 + w/2)] = (2/w) / 2
    F[(x == x0 - w/2) | ( x == x0 + w/2)] = (2/w) / 4
    # F[(x > x0 - w/2) & (x < x0 + w/2)] = 1
    # F[(x == x0 - w/2) | ( x == x0 + w/2)] = 1/2

    return F


def voigt(x: float | Array[float], x0: float, sigma: float, gamma: float) -> Array[float]:
    """Voigt distribution with position x0 and unit intensity.

    Params:
        x0: position;
        sigma: width of Gaussian profile;
        gamma: width of Lorentzian profile.

    """
    rx = (x[-1] - x[0]) / 2

    G = np.exp(-(x - x0)**2 / (2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    L = gamma / (np.pi*(x**2 + gamma**2))
    F = signal.convolve(G, L, mode='same') * 2*rx/len(x)

    return F


def pvoigt(x: float | Array[float], x0: float, w: float, a: float, r: float) -> Array[float]:
    """Pseudo-Voigt distribution with position x0 and unit intensity.

    Params:
        x0: position;
        w: width (full width at half maximum);
        a: assymetry;
        r: ratio (in range 0-1).

    A simple asymmetric line shape profile for fitting infrared absorption spectra.
    Aaron L. Stancik, Eric B. Brauns
    https://www.sciencedirect.com/science/article/abs/pii/S0924203108000453
    """

    sigma = 2*w / (1 + np.exp(a*(x - x0)) )
    G = np.sqrt(4*np.log(2)/np.pi) / sigma * np.exp(-4*np.log(2)*((x - x0)/sigma)**2)
    L = 2/np.pi/sigma/(1 + 4*((x - x0)/sigma)**2)
    F = r*L + (1 - r)*G

    return F


# --------        utils        --------
def voigt2pvoigt(x: Array[float], x0: float, sigma: float, gamma: float) -> tuple[float, float, float]:
    """Approximate voigt function by psevdo-voigt (pvoigt) function."""

    def func(x: float, x0: float, y: Array[float], params) -> float:
        y_hat = pvoigt(x, x0, *params)

        return mse(y, y_hat)

    y = voigt(x, x0=x0, sigma=sigma, gamma=gamma)

    res = optimize.minimize(
        partial(func, x, x0, y),
        x0=[2, 0, 0],
        bounds=[
            (0, 10),
            [-0.1, +0.1],
            [0, 1],
        ]
    )
    assert res['success'], 'Optimization is not succeeded!'

    return res['x']
