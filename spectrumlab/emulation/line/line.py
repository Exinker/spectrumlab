from dataclasses import dataclass, field
from typing import Tuple, overload

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

from spectrumlab.emulation.curve import gauss, pvoigt, voigt
from spectrumlab.grid import Grid
from spectrumlab.grid.utils import estimate_fwhm
from spectrumlab.picture.color import COLOR
from spectrumlab.types import Array, PicoMeter


# --------        shape        --------
@dataclass(frozen=True)
class GaussLineShape:
    """Gauss (or normal) line profile's shape."""
    width: PicoMeter

    # --------        private        --------
    @overload
    def __call__(self, x: PicoMeter, position: PicoMeter, intensity: float) -> Array[float]: ...
    @overload
    def __call__(self, x: Array[PicoMeter], position: PicoMeter, intensity: float) -> Array[float]: ...
    def __call__(self, x, position, intensity):
        f = intensity*gauss(x, x0=position, w=self.width)

        return f


@dataclass(frozen=True, slots=True)
class VoigtLineShape:
    """Voigt line shape."""
    g: PicoMeter  # fwhm of gauss profile shape (doppler broadening)
    l: PicoMeter  # fwhm of lorents profile shape (collisional broadening)

    @property
    def sigma(self) -> PicoMeter:
        return self.g / np.sqrt(8 * np.log(2))

    @property
    def gamma(self) -> PicoMeter:
        return self.l / 2

    @property
    def fwhm(self) -> PicoMeter:
        """Calculate FWHM of the shape [by formula](https://en.wikipedia.org/wiki/Voigt_profile#The_width_of_the_Voigt_profile)."""
        return self.l/2 + np.sqrt((self.l/2)**2 + self.g**2)

    # --------        private        --------
    @overload
    def __call__(self, x: PicoMeter, position: PicoMeter, intensity: float) -> Array[float]: ...
    @overload
    def __call__(self, x: Array[PicoMeter], position: PicoMeter, intensity: float) -> Array[float]: ...
    def __call__(self, x, position, intensity):
        f = intensity*voigt(x, x0=position, sigma=self.sigma, gamma=self.gamma)

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

    @property
    def fwhm(self, dx: PicoMeter = 1e-2, rx: PicoMeter = 100) -> PicoMeter:
        """Estimate FWHM of the shape"""
        x = np.linspace(-rx, +rx, 2*int(rx/dx) + 1)
        y = self(x, 0, 1)

        fwhm = estimate_fwhm(
            grid=Grid(x, y),
            pitch=1,
        )

        return fwhm

    # --------        private        --------
    @overload
    def __call__(self, x: PicoMeter, position: PicoMeter, intensity: float) -> Array[float]: ...
    @overload
    def __call__(self, x: Array[PicoMeter], position: PicoMeter, intensity: float) -> Array[float]: ...
    def __call__(self, x, position, intensity):
        f = intensity*pvoigt(x, x0=position, w=self.width, a=self.asymmetry, r=self.ratio)

        return f


@dataclass(frozen=True)
class SelfReversedPVoigtLineShape:
    """Self-reversed voigt line profile's shape with self-absorption"""
    width: PicoMeter
    asymmetry: float
    ratio: float
    absorbance: float

    # --------        private        --------
    @overload
    def __call__(self, x: PicoMeter, position: PicoMeter, intensity: float) -> Array[float]: ...
    @overload
    def __call__(self, x: Array[PicoMeter], position: PicoMeter, intensity: float) -> Array[float]: ...
    def __call__(self, x, position, intensity):
        voigt = pvoigt(x, x0=position, w=self.width, a=self.asymmetry, r=self.ratio)
        f = intensity*voigt*10**(-self.absorbance*voigt)

        return f


@dataclass(frozen=True)
class SigmoidsLineShape:
    """Time distribution."""
    width: Tuple[float, float]
    power: float

    # --------        private        --------
    @overload
    def __call__(self, x: PicoMeter, position: PicoMeter, intensity: float) -> Array[float]: ...
    @overload
    def __call__(self, x: Array[PicoMeter], position: PicoMeter, intensity: float) -> Array[float]: ...
    def __call__(self, x, position, intensity):
        w = self.width
        p = self.power

        sigmoid = lambda x: ((4/np.pi) * (np.arctan(-w[0]*(x - position)) + np.pi/2) * (1/(1 + np.exp(-w[1]*(x - position)))))**p  # noqa: E731
        f = intensity*sigmoid(x)/integrate.quad(sigmoid, a=position-1e+3, b=position+1e+3)[0]

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
