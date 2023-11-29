
from dataclasses import dataclass, field
from typing import Tuple, overload

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

from spectrumlab.alias import Array, Micro
from spectrumlab.picture.config import COLOR
from .curve import gauss, pvoigt


# --------        profile        --------
@dataclass(frozen=True)
class GaussLineProfile:
    """Normal distribution with position and standart deviation width."""
    width: Micro

    @overload
    def __call__(self, x: float, position: Micro, intensity: float) -> Array: ...
    @overload
    def __call__(self, x: Array, position: Micro, intensity: float) -> Array: ...
    def __call__(self, x, position, intensity):
        F = gauss(x, x0=position, w=self.width)

        f = intensity*F

        return f


@dataclass(frozen=True)
class VoigtLineProfile:
    """
    A simple asymmetric line shape profile for fitting infrared absorption spectra.
    Aaron L. Stancik, Eric B. Brauns
    https://www.sciencedirect.com/science/article/abs/pii/S0924203108000453
    """
    width: Micro
    asymmetry: float = field(default=0)  # non asymmetric default
    ratio: float = field(default=0)  # gauss profile default

    @overload
    def __call__(self, x: float, position: Micro, intensity: float) -> Array: ...
    @overload
    def __call__(self, x: Array, position: Micro, intensity: float) -> Array: ...
    def __call__(self, x, position, intensity):
        F = pvoigt(x, x0=position, w=self.width, a=self.asymmetry, r=self.ratio)
        f = intensity*F

        return f


@dataclass(frozen=True)
class SelfAbsorptionVoigtLineProfile:
    """Voigt line shape profile with self-absorption"""
    width: Micro
    asymmetry: float
    ratio: float
    absorption: float

    @overload
    def __call__(self, x: float, position: Micro, intensity: float) -> Array: ...
    @overload
    def __call__(self, x: Array, position: Micro, intensity: float) -> Array: ...
    def __call__(self, x, position, intensity):
        F = pvoigt(x, x0=position, w=self.width, a=self.asymmetry, r=self.ratio)
        f = intensity*F * 10**(-self.absorption*F)

        return f


@dataclass(frozen=True)
class SigmoidsLineProfile:
    """Time distribution"""
    width: Tuple[float, float]
    power: float

    @overload
    def __call__(self, x: float, position: Micro, intensity: float) -> Array: ...
    @overload
    def __call__(self, x: Array, position: Micro, intensity: float) -> Array: ...
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


LineProfile = GaussLineProfile | VoigtLineProfile | SelfAbsorptionVoigtLineProfile | SigmoidsLineProfile


# --------        line        --------
@dataclass(frozen=True)
class Line:
    """
    Interface for any line profile function.

    Author: Vaschenko Pavel
     Email: vaschenko@vmk.ru
      Date: 2013.04.12
    """
    profile: LineProfile

    @overload
    def __call__(self, x: float, position: Micro, intensity: float) -> Array: ...
    @overload
    def __call__(self, x: Array, position: Micro, intensity: float) -> Array: ...
    def __call__(self, x, position, intensity):
        return self.profile(x, position, intensity)

    # --------        fabric        --------
    @classmethod
    def from_profile(cls, profile: LineProfile) -> 'Line':
        return cls(profile=profile)

    def show(self, position: Micro, intensity: float, rx: Micro = 100, dx: Micro = .01) -> None:
        """Show line profile at the range rx with step dx."""

        plt.style.use('seaborn-whitegrid')
        plt.rcParams.update({
            'figure.figsize': (10, 5),
            'font.size': 14,
        })
        plt.figure(figsize=(10, 5))

        #
        x = np.arange(-rx, rx+dx, dx)
        f = lambda x: self(
            x=x,
            position=position,
            intensity=intensity,
        )

        #
        plt.plot(x, f(x), color=COLOR['blue'], label=r'$\mathcal{F}(x)$')

        plt.title('Line function')
        plt.xlabel(r'$x, \mu$')
        plt.ylabel('Intensity')

        plt.grid(color='grey', linestyle=':')
        plt.legend()
        plt.show()


if __name__ == '__main__':

    line = Line(
        profile=VoigtLineProfile(25, 0, 0),
    )
    line.show(position=25, intensity=1)
