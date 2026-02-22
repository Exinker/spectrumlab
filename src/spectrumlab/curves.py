from functools import partial
from typing import TypeVar

import numpy as np
from scipy import optimize, signal

from spectrumlab.types import Array, MicroMeter, NanoMeter, Number, PicoMeter
from spectrumlab.utils import mse


T = TypeVar('T', Number, MicroMeter, NanoMeter, PicoMeter)


def gauss(x: T | Array[T], x0: T, w: T) -> Array[float]:
    """Gauss (or normal) distribution with position `x0` and unit intensity.

    Params:
        x0 - position
        w - width
    """

    f = np.exp(-(1/2)*((x - x0) / w)**2) / (np.sqrt(2*np.pi) * w)

    return f


def lanczos(x: T | Array[T], x0: T, a: int = 2) -> Array[float]:
    """Lanczos distribution with position `x0`.

    Params:
        x0 - position
        a - window width
    """

    f = np.sinc(x - x0) * np.sinc((x - x0) / a)
    f[(x - x0 < -a) | (x - x0 > a)] = 0

    return f


def rectangular(x: T | Array[T], x0: T, w: T) -> Array[float]:
    """Rectangular distribution with position `x0` and unit intensity.

    Params:
        x0 - position
        w - width (full width)
    """

    if isinstance(x, float):  # TODO: don't remove! It's for integrate.quad functions!
        f = np.zeros((1, ))
    else:
        f = np.zeros(x.shape)

    f[(x > x0 - w/2) & (x < x0 + w/2)] = (2/w) / 2
    f[(x == x0 - w/2) | (x == x0 + w/2)] = (2/w) / 4

    return f


def voigt(x: T | Array[T], x0: T, sigma: float, gamma: float) -> Array[float]:
    """Voigt distribution with position `x0` and unit intensity.

    Params:
        x0 - position
        sigma - width of Gaussian shape
        gamma - width of Lorentzian shape
    """
    rx = (x[-1] - x[0]) / 2

    gauss = np.exp(-(x - x0)**2 / (2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    lorentz = gamma / (np.pi*(x**2 + gamma**2))
    f = signal.convolve(gauss, lorentz, mode='same') * 2*rx/len(x)

    return f


def pvoigt(x: T | Array[T], x0: T, w: T, a: float, r: float) -> Array[float]:
    """Pseudo-Voigt distribution with position `x0` and unit intensity.

    Params:
        x0 - position
        w - width (full width at half maximum)
        a - assymetry
        r - ratio (in range [0; 1])

    A simple asymmetric line shape shape for fitting infrared absorption spectra.
    Aaron l. Stancik, Eric B. Brauns
    https://www.sciencedirect.com/science/article/abs/pii/S0924203108000453
    """
    sigma = 2*w / (1 + np.exp(a*(x - x0)))

    gauss = np.sqrt(4*np.log(2)/np.pi) / sigma * np.exp(-4*np.log(2)*((x - x0)/sigma)**2)
    lorentz = 2/np.pi/sigma/(1 + 4*((x - x0)/sigma)**2)
    f = r*lorentz + (1 - r)*gauss

    return f


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
            (1e-15, 100),
            [-0.25, +0.25],
            [0, 1],
        ],
    )
    assert res['success'], 'Optimization is not succeeded!'

    return res['x']
