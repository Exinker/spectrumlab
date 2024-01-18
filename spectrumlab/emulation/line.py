from dataclasses import dataclass, field
from typing import Tuple, overload

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

from spectrumlab.alias import Array, PicoMeter
from spectrumlab.core.grid import Grid, estimate_fwhm
from spectrumlab.emulation.curve import gauss, voigt, pvoigt, voigt2pvoigt
from spectrumlab.picture.config import COLOR


# --------        shape        --------
@dataclass(frozen=True)
class GaussLineShape:
    """Gauss (or normal) line profile's shape."""
    width: PicoMeter

    @overload
    def __call__(self, x: PicoMeter, position: PicoMeter, intensity: float) -> Array[float]: ...
    @overload
    def __call__(self, x: Array[PicoMeter], position: PicoMeter, intensity: float) -> Array[float]: ...
    def __call__(self, x, position, intensity):
        F = gauss(x, x0=position, w=self.width)
        f = intensity*F

        return f


@dataclass(frozen=True, slots=True)
class VoigtLineShape:
    """Voigt line shape."""
    g: PicoMeter  # fwhm of gauss profile shape (doppler broadening)
    l: PicoMeter  # fwhm of lorents profile shape (collisional broadening)

    dx: PicoMeter = field(default=0.01)  # шаг построения интерполяции
    rx: PicoMeter = field(default=10)  # границы построения интерполяции

    @property
    def sigma(self) -> PicoMeter:
        return self.g / np.sqrt(8 * np.log(2))

    @property
    def gamma(self) -> PicoMeter:
        return self.l / 2

    @property
    def x(self) -> Array[PicoMeter]:
        return np.linspace(-self.rx, +self.rx, 2*int(self.rx/self.dx) + 1)

    @property
    def y(self) -> Array[float]:
        return self(self.x, 0, 1)

    # --------        handlers        --------
    def calculate_fwhm(self) -> PicoMeter:
        """Calculate FWHM of the shape [by formula](https://en.wikipedia.org/wiki/Voigt_profile#The_width_of_the_Voigt_profile)."""
        return self.l/2 + np.sqrt((self.l/2)**2 + self.g**2)

    def estimate_fwhm(self) -> PicoMeter:
        """Estimate FWHM of the shape."""
        hwhm = estimate_fwhm(
            grid=Grid(x=self.x, y=self.y)
        )

        return hwhm
    
    def to_pseudo(self, show: bool = False) -> 'PVoigtLineShape':
        """Approx voigt shape by pvoigt shape."""
        
        params = voigt2pvoigt(self.x, x0=0, sigma=self.sigma, gamma=self.gamma)
        shape = PVoigtLineShape(*params)

        # show
        if show:
            x = self.x
            y = self.y
            y_hat = shape(x, position=0, intensity=1)

            plt.plot(
                x, y,
                color='red', linestyle='none', marker='s', markersize=3,
                label=r'voigt shape',
            )
            plt.plot(
                x, y_hat,
                label=r'pvoigt shape',
                color='black', linestyle='-', linewidth=1,
            )
            plt.plot(
                x, y_hat - y,
                color='black', linestyle='none', marker='s', markersize=0.5,
                label=r'error',
            )

            plt.xlabel('$x$ $[pm]$')
            plt.ylabel('$f(x)$')

            plt.grid(linestyle=':')
            plt.legend()
            plt.show()

        #
        return shape

    # --------        private        --------
    @overload
    def __call__(self, x: PicoMeter, position: PicoMeter, intensity: float) -> Array[float]: ...
    @overload
    def __call__(self, x: Array[PicoMeter], position: PicoMeter, intensity: float) -> Array[float]: ...
    def __call__(self, x, position, intensity):
        F = voigt(x, x0=position, sigma=self.sigma, gamma=self.gamma)
        f = intensity*F

        return f


@dataclass(frozen=True)
class PVoigtLineShape:
    """
    A simple asymmetric line shape profile for fitting infrared absorption spectra.
    Aaron L. Stancik, Eric B. Brauns
    https://www.sciencedirect.com/science/article/abs/pii/S0924203108000453
    """
    width: PicoMeter
    asymmetry: float = field(default=0)  # non asymmetric default
    ratio: float = field(default=0)  # gauss shape default

    @overload
    def __call__(self, x: PicoMeter, position: PicoMeter, intensity: float) -> Array[float]: ...
    @overload
    def __call__(self, x: Array[PicoMeter], position: PicoMeter, intensity: float) -> Array[float]: ...
    def __call__(self, x, position, intensity):
        F = pvoigt(x, x0=position, w=self.width, a=self.asymmetry, r=self.ratio)
        f = intensity*F

        return f


@dataclass(frozen=True)
class SelfReversedPVoigtLineShape:
    """Self-reversed voigt line profile's shape with self-absorption"""
    width: PicoMeter
    asymmetry: float
    ratio: float
    absorbance: float

    @overload
    def __call__(self, x: PicoMeter, position: PicoMeter, intensity: float) -> Array[float]: ...
    @overload
    def __call__(self, x: Array[PicoMeter], position: PicoMeter, intensity: float) -> Array[float]: ...
    def __call__(self, x, position, intensity):
        F = pvoigt(x, x0=position, w=self.width, a=self.asymmetry, r=self.ratio)
        f = intensity*F * 10**(-self.absorbance*F)

        return f


@dataclass(frozen=True)
class SigmoidsLineShape:
    """Time distribution."""
    width: Tuple[float, float]
    power: float

    @overload
    def __call__(self, x: PicoMeter, position: PicoMeter, intensity: float) -> Array[float]: ...
    @overload
    def __call__(self, x: Array[PicoMeter], position: PicoMeter, intensity: float) -> Array[float]: ...
    def __call__(self, x, position, intensity):
        w = self.width
        p = self.power

        F = lambda x: ( (4/np.pi) * (np.arctan(-w[0]*(x - position)) + np.pi/2) * (1/(1 + np.exp(-w[1]*(x - position)))) )**p
        F = F(x) / integrate.quad(
            lambda x: F(x),
            a=position-1e+3,
            b=position+1e+3,
        )[0]  # normalization
        f = intensity*F

        return f


LineShape = GaussLineShape | VoigtLineShape | PVoigtLineShape | SelfReversedPVoigtLineShape | SigmoidsLineShape


# --------        line        --------
@dataclass(frozen=True)
class Line:
    """
    Interface for any line's shape.

    Author: Vaschenko Pavel
     Email: vaschenko@vmk.ru
      Date: 2013.04.12
    """
    shape: LineShape

    @overload
    def __call__(self, x: PicoMeter, position: PicoMeter, intensity: float) -> Array[float]: ...
    @overload
    def __call__(self, x: Array[PicoMeter], position: PicoMeter, intensity: float) -> Array[float]: ...
    def __call__(self, x, position, intensity):
        return self.shape(x, position, intensity)

    # --------        handlers        --------
    def show(self, position: PicoMeter, intensity: float, rx: PicoMeter = 100, dx: PicoMeter = .01) -> None:
        """Show line profile's shape at the range rx with step dx."""

        #
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

        x = np.linspace(-rx, +rx, 2*int(rx/dx) + 1)
        y = self(x=x, position=position, intensity=intensity)
        plt.plot(
            x, y,
            color=COLOR['blue'],
            label=r'${I}(x)$',
        )

        plt.xlabel(r'$x, \mu$')
        plt.ylabel(r'$I(x)$')

        plt.grid(color='grey', linestyle=':')
        plt.legend()
        plt.show()


if __name__ == '__main__':

    line = Line(
        shape=PVoigtLineShape(width=25, asymmetry=0, ratio=.1),
    )
    line.show(position=0, intensity=1)
