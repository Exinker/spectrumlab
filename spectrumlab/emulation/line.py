"""
Spectral lines for emulation with given line shape.

Author: Vaschenko Pavel
 Email: vaschenko@vmk.ru
  Date: 2013.04.12
"""
from dataclasses import dataclass, field
from typing import Tuple, overload

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

from spectrumlab.alias import Array, Micro
from spectrumlab.emulation.curve import gauss, pvoigt
from spectrumlab.picture.config import COLOR


# --------        shape        --------
@dataclass(frozen=True)
class NormalLineShape:
    """Gauss (or normal) distribution with standart deviation `width`."""
    width: Micro

    @overload
    def __call__(self, x: Micro, position: Micro, intensity: float) -> Array[float]: ...
    @overload
    def __call__(self, x: Array[Micro], position: Micro, intensity: float) -> Array[float]: ...
    def __call__(self, x, position, intensity):
        F = gauss(x, x0=position, w=self.width)

        f = intensity*F

        return f


@dataclass(frozen=True)
class VoigtLineShape:
    """
    A simple asymmetric line shape shape for fitting infrared absorption spectra.
    Aaron L. Stancik, Eric B. Brauns
    https://www.sciencedirect.com/science/article/abs/pii/S0924203108000453
    """
    width: Micro
    asymmetry: float = field(default=0)  # non asymmetric default
    ratio: float = field(default=0)  # gauss shape default

    @overload
    def __call__(self, x: Micro, position: Micro, intensity: float) -> Array[float]: ...
    @overload
    def __call__(self, x: Array[Micro], position: Micro, intensity: float) -> Array[float]: ...
    def __call__(self, x, position, intensity):
        F = pvoigt(x, x0=position, w=self.width, a=self.asymmetry, r=self.ratio)
        f = intensity*F

        return f


@dataclass(frozen=True)
class SelfAbsorptionVoigtLineShape:
    """`VoigtLineShape` with self-absorption."""
    width: Micro
    asymmetry: float
    ratio: float
    absorbance: float

    @overload
    def __call__(self, x: Micro, position: Micro, intensity: float) -> Array[float]: ...
    @overload
    def __call__(self, x: Array[Micro], position: Micro, intensity: float) -> Array[float]: ...
    def __call__(self, x, position, intensity):
        F = pvoigt(x, x0=position, w=self.width, a=self.asymmetry, r=self.ratio)
        f = intensity*F * 10**(-self.absorbance*F)

        return f


@dataclass(frozen=True)
class SigmoidsLineShape:
    """Time intensity distribution."""
    width: Tuple[float, float]
    power: float

    @overload
    def __call__(self, t: float, position: float, intensity: float) -> Array: ...
    @overload
    def __call__(self, t: Array, position: float, intensity: float) -> Array: ...
    def __call__(self, t, position, intensity):
        w = self.width
        p = self.power

        F = lambda t: ( (4/np.pi) * (np.arctan(-w[0]*(t - position)) + np.pi/2) * (1/(1 + np.exp(-w[1]*(t - position)))) )**p
        F = F(t) / integrate.quad(
            lambda t: F(t),
            a=position-1e+3,
            b=position+1e+3,
        )[0]  # normalization

        f = intensity*F

        return f


LineShape = NormalLineShape | VoigtLineShape | SelfAbsorptionVoigtLineShape | SigmoidsLineShape


# --------        line        --------
@dataclass(frozen=True)
class Line:
    """
    Interface for any line shape function.

    Author: Vaschenko Pavel
     Email: vaschenko@vmk.ru
      Date: 2013.04.12
    """
    shape: LineShape

    @overload
    def __call__(self, x: Micro, position: Micro, intensity: float) -> Array[float]: ...
    @overload
    def __call__(self, x: Array[Micro], position: Micro, intensity: float) -> Array[float]: ...
    def __call__(self, x, position, intensity):
        return self.shape(x, position, intensity)

    # --------        handlers        --------
    def show(self, position: Micro, intensity: float, rx: Micro = 100, dx: Micro = .01) -> None:
        """Show line shape at the range `rx` with step `dx`."""

        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

        x = np.arange(-rx, rx+dx, dx)
        y = self(x=x, position=position, intensity=intensity)
        plt.plot(
            x, y,
            color=COLOR['blue'],
            label=r'${I}(x)$',
        )
        plt.plot(
            x, f(x),
            color=COLOR['blue'],
            label=r'$I(x)$',
        )

        plt.xlabel(r'$x$ [$\mu$]')
        plt.ylabel(r'$I$ $[\%]$')
        plt.grid(color='grey', linestyle=':')
        plt.legend()

        plt.show()

    # --------        fabric        --------
    @classmethod
    def from_shape(cls, shape: LineShape) -> 'Line':
        return cls(shape=shape)


if __name__ == '__main__':
    line = Line(
        shape=VoigtLineShape(25, 0, 0),
    )
    line.show(position=0, intensity=1)
