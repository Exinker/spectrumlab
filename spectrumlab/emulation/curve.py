from functools import partial
from typing import Callable

import numpy as np
from scipy import interpolate, optimize, signal

from spectrumlab.alias import Array, Number
from spectrumlab.utils import mse


def gauss(x: Number | Array[Number], x0: Number, w: Number) -> Array[float]:
    """Gauss (or normal) distribution with position `x0` and unit intensity.

    Params:
        x0: Number - position
        w: Number - width 
    """

    F = np.exp( -(1/2)*((x - x0) / w)**2 ) / ( np.sqrt(2*np.pi) * w )

    return F


def lanczos(x: Number | Array[Number], x0: Number, a: int = 2) -> Array[float]:
    """Lanczos distribution with position `x0`.
    
    Params:
        x0: Number - position
        a: float - window width
    """

    F = np.sinc(x - x0) * np.sinc((x - x0) / a)
    F[(x - x0 < -a) | (x - x0 > a)] = 0

    return F


def rectangular(x: Number | Array[Number], x0: Number, w: Number) -> Array[float]:
    """Rectangular distribution with position `x0` and unit intensity.

    Params:
        x0: Number - position
        w: Number - width (full width)
    """

    if isinstance(x, float):  # TODO: don't remove! It's for integrate.quad functions!
        F = np.zeros(1,)
    else:
        F = np.zeros(x.shape)

    F[(x > x0 - w/2) & (x < x0 + w/2)] = (2/w) / 2
    F[(x == x0 - w/2) | ( x == x0 + w/2)] = (2/w) / 4

    return F


def voigt(x: Number | Array[Number], x0: Number, sigma: float, gamma: float) -> Array[float]:
    """Voigt distribution with position `x0` and unit intensity.

    Params:
        x0: Number - position;
        sigma: float - width of Gaussian shape
        gamma: float - width of Lorentzian shape
    """
    rx = (x[-1] - x[0]) / 2

    G = np.exp(-(x - x0)**2 / (2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    L = gamma / (np.pi*(x**2 + gamma**2))
    F = signal.convolve(G, L, mode='same') * 2*rx/len(x)

    return F


def pvoigt(x: Number | Array[Number], x0: Number, w: Number, a: float, r: float) -> Array[float]:
    """Pseudo-Voigt distribution with position `x0` and unit intensity.

    Params:
        x0: Number - position
        w: Number - width (full width at half maximum)
        a: float - assymetry
        r: float - ratio (in range [0; 1])

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
def estimate_fwhm(x: Array[float], y: Array[float]) -> float:
    """Estimate a full width at half maximum (FWHM).

    A grid values (`x`, `y`) should be centered and symmetric!
    """

    def _loss(x: float, f: Callable[[float], float]) -> float:
        return (f(0)/2 - f(x))**2

    f = interpolate.interp1d(
        x, y,
        kind='linear',
        bounds_error=False,
        fill_value=0,
    )

    res = optimize.minimize(
        partial(_loss, f=f),
        x0=0,
    )
    assert res['success'], 'Optimization is not success!'

    fwhm = np.abs(2*res['x'].item())

    # 
    return fwhm


def voigt2pvoigt(x: Array[float], x0: float, sigma: float, gamma: float) -> tuple[float, float, float]:
    """Approximate `voigt` shape by psevdo-voigt (`pvoigt`) shape."""

    def loss(x: float, x0: float, y: Array[float], params) -> float:
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
