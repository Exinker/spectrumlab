from functools import partial
from typing import TypeVar

import numpy as np
from scipy import optimize, signal

from spectrumlab.alias import Array, Number
from spectrumlab.utils import mse

from spectrumlab.alias import Array, Number, MicroMeter, NanoMeter, PicoMeter


T = TypeVar('T', Number, MicroMeter, NanoMeter, PicoMeter)


def gauss(x: T | Array[T], x0: T, w: T) -> Array[float]:
    """Gauss (or normal) distribution with position `x0` and unit intensity.

    Params:
        x0 - position
        w - width 
    """

    F = np.exp( -(1/2)*((x - x0) / w)**2 ) / ( np.sqrt(2*np.pi) * w )

    return F


def lanczos(x: T | Array[T], x0: T, a: int = 2) -> Array[float]:
    """Lanczos distribution with position `x0`.
    
    Params:
        x0 - position
        a - window width
    """

    F = np.sinc(x - x0) * np.sinc((x - x0) / a)
    F[(x - x0 < -a) | (x - x0 > a)] = 0

    return F


def rectangular(x: T | Array[T], x0: T, w: T) -> Array[float]:
    """Rectangular distribution with position `x0` and unit intensity.

    Params:
        x0 - position
        w - width (full width)
    """

    if isinstance(x, float):  # TODO: don't remove! It's for integrate.quad functions!
        F = np.zeros(1,)
    else:
        F = np.zeros(x.shape)

    F[(x > x0 - w/2) & (x < x0 + w/2)] = (2/w) / 2
    F[(x == x0 - w/2) | ( x == x0 + w/2)] = (2/w) / 4

    return F


def voigt(x: T | Array[T], x0: T, sigma: float, gamma: float) -> Array[float]:
    """Voigt distribution with position `x0` and unit intensity.

    Params:
        x0 - position
        sigma - width of Gaussian shape
        gamma - width of Lorentzian shape
    """
    rx = (x[-1] - x[0]) / 2

    G = np.exp(-(x - x0)**2 / (2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    L = gamma / (np.pi*(x**2 + gamma**2))
    F = signal.convolve(G, L, mode='same') * 2*rx/len(x)

    return F


def pvoigt(x: T | Array[T], x0: T, w: T, a: float, r: float) -> Array[float]:
    """Pseudo-Voigt distribution with position `x0` and unit intensity.

    Params:
        x0 - position
        w - width (full width at half maximum)
        a - assymetry
        r - ratio (in range [0; 1])

    A simple asymmetric line shape shape for fitting infrared absorption spectra.
    Aaron L. Stancik, Eric B. Brauns
    https://www.sciencedirect.com/science/article/abs/pii/S0924203108000453
    """
    sigma = 2*w / (1 + np.exp(a*(x - x0)) )

    G = np.sqrt(4*np.log(2)/np.pi) / sigma * np.exp(-4*np.log(2)*((x - x0)/sigma)**2)
    L = 2/np.pi/sigma/(1 + 4*((x - x0)/sigma)**2)
    F = r*L + (1 - r)*G

    return F


# --------        utils        --------
def voigt2pvoigt(x: Array[T], x0: T, sigma: float, gamma: float) -> tuple[T, float, float]:
    """Approximate `voigt` shape by psevdo-voigt (`pvoigt`) shape."""

    def loss(x: T, x0: T, y: Array[float], params) -> float:
        y_hat = pvoigt(x, x0, *params)

        return mse(y, y_hat)

    y = voigt(x, x0=x0, sigma=sigma, gamma=gamma)

    res = optimize.minimize(
        partial(loss, x, x0, y),
        x0=[2, 0, 0],
        bounds=[
            (0, 10),
            [-0.1, +0.1],
            [0, 1],
        ]
    )
    assert res['success'], 'Optimization is not succeeded!'

    return res['x']
