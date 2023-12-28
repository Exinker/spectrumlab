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
    """Normal (gauss) line profile's shape."""
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
    A simple asymmetric line shape profile for fitting infrared absorption spectra.
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
class SelfReversedVoigtLineShape:
    """Self-reversed voigt line profile's shape with self-absorption"""
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
    """Time distribution."""
    width: Tuple[float, float]
    power: float

    @overload
    def __call__(self, x: Micro, position: Micro, intensity: float) -> Array[float]: ...
    @overload
    def __call__(self, x: Array[Micro], position: Micro, intensity: float) -> Array[float]: ...
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


LineShape = NormalLineShape | VoigtLineShape | SelfReversedVoigtLineShape | SigmoidsLineShape


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
    def __call__(self, x: Micro, position: Micro, intensity: float) -> Array[float]: ...
    @overload
    def __call__(self, x: Array[Micro], position: Micro, intensity: float) -> Array[float]: ...
    def __call__(self, x, position, intensity):
        return self.shape(x, position, intensity)

    # --------        fabric        --------
    @classmethod
    def from_shape(cls, shape: LineShape) -> 'Line':
        return cls(shape=shape)

    def show(self, position: Micro, intensity: float, rx: Micro = 100, dx: Micro = .01) -> None:
        """Show line profile's shape at the range rx with step dx."""

        #
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

        x = np.arange(-rx, rx+dx, dx)
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
        shape=VoigtLineShape(width=25, asymmetry=0, ratio=.1),
    )
    line.show(position=0, intensity=1)
