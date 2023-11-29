
from dataclasses import astuple, dataclass, field
from typing import Literal, overload

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, signal

from spectrumlab.alias import Array, Micro, Number
from spectrumlab.picture.config import COLOR
from spectrumlab.emulation.curve import rectangular, pvoigt
from spectrumlab.emulation.detector.linear_array_detector import Detector


# --------        profile        --------
@dataclass
class RectangularApertureProfile:
    """Rectangular shape aperture profile."""
    detector: Detector

    def __call__(self, x: float | Array[float], n: int) -> Array[float]:
        detector = self.detector
        step = detector.config.width

        f = rectangular(x, x0=step*n, w=step)

        return f


@dataclass
class VoigtApertureProfile:
    """Voigt shape aperture profile.
    
    A simple asymmetric line shape profile for fitting infrared absorption spectra.
    Aaron L. Stancik, Eric B. Brauns
    https://www.sciencedirect.com/science/article/abs/pii/S0924203108000453
    """
    detector: Detector
    width: Number
    asymmetry: float
    ratio: float

    def __call__(self, x: float | Array[float], n: int) -> Array[float]:
        detector = self.detector
        step = detector.config.width

        f = pvoigt(x, x0=step*n, w=step*self.width, a=self.asymmetry, r=self.ratio)

        return f


@dataclass
class ConvolutionApertureProfile:
    """Convolution of a rectangular and a pvoigt line shape profile"""
    detector: Detector

    width: Number = field(default=.2)
    dx: Micro = field(default=0.1)  # шаг построения интерполяции
    rx: Micro = field(default=100)  # границы построения интерполяции

    _x: Array[Number] = field(init=False, repr=False, default=None)
    _f: Array[float] = field(init=False, repr=False, default=None)

    def __post_init__(self):
        detector = self.detector
        step = detector.config.width

        x = np.arange(-self.rx, self.rx+self.dx, self.dx)

        f1 = lambda x: rectangular(x, x0=0, w=step)
        f2 = lambda x: pvoigt(x, 0, w=step*self.width, a=0, r=0)
        f = signal.convolve(f1(x), f2(x), mode='same') * (x[-1] - x[0])/len(x)

        self._x = x
        self._f = f

    def __call__(self, x: float | Array, n: int) -> float | Array:
        detector = self.detector
        step = detector.config.width
        
        F = interpolate.interp1d(
            self._x,
            self._f,
            kind='linear',
            bounds_error=False,
            fill_value=0,
        )
        f = F(x - step*n)

        return f


# --------        approximated profile        --------
@dataclass
class ApproximatedApertureProfile:
    """approximated aperture's profile"""
    detector: Detector

    wavelength: Literal[405] = field(default=405)
    dx: Micro = field(default=0.1)  # шаг построения интерполяции
    rx: Micro = field(default=100)  # границы построения интерполяции

    _x: Array[Number] = field(init=False, repr=False, default=None)
    _f: Array[float] = field(init=False, repr=False, default=None)

    PARAMS = {
        405: {
            Detector.BLPP369M1: (4.9173, 0, 1.0000),  # (!) bad approximation
            Detector.BLPP2000: (17.9677, 0, 0.4186),
            Detector.BLPP4000: (3.5592, 0, 0.4578),
        },  # aperture's profile params at 405 nm
    }

    def __post_init__(self):
        detector = self.detector
        wavelength = self.wavelength
        step = detector.config.width
        w, a, r = self.PARAMS[wavelength][detector]

        #
        x = np.arange(-self.rx, self.rx+self.dx, self.dx)

        f1 = lambda x: rectangular(x, x0=0, w=step)
        f2 = lambda x: pvoigt(x, 0, w=w, a=a, r=r)
        f = signal.convolve(f1(x), f2(x), mode='same') * (x[-1] - x[0])/len(x)

        self._x = x
        self._f = f

    def __call__(self, x: float | Array, n: int) -> float | Array:
        detector = self.detector
        step = detector.config.width

        F = interpolate.interp1d(
            self._x,
            self._f,
            kind='linear',
            bounds_error=False,
            fill_value=0,
        )
        f = F(x - step*n)

        return f


ApertureProfile = RectangularApertureProfile | VoigtApertureProfile | ConvolutionApertureProfile | ApproximatedApertureProfile


# --------        aperture        --------
@dataclass
class Aperture:
    """
    Interface for any aperture profile function.

    Author: Vaschenko Pavel
     Email: vaschenko@vmk.ru
      Date: 2014.03.24
    """
    profile: ApertureProfile

    def __call__(self, x: float | Array[float], n: int) -> Array[float]:
        return self.profile(x, n=n)

    # --------        handlers        --------
    @property
    def detector(self) -> Detector:
        return self.profile.detector

    def show(self, rx: Micro = 100, dx: Micro = .01, xscale: Number | Micro = Number, yscale: str | None = None) -> None:
        step = self.detector.config.width
        n_steps = rx // step + 1

        #
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), tight_layout=True)

        x = np.arange(0, rx+dx, dx)
        integral = np.zeros(x.shape)
        for n in range(n_steps):
            xvalues = x if xscale == Micro else x/step
            yvalues = self(x, n=n)
            plt.plot(
                xvalues, yvalues,
                color=COLOR['blue'],
                label='$S(x - x_{k})$' if n==0 else None,
            )

            integral += yvalues

        xvalues = x if xscale == Micro else x/step
        yvalues = integral
        plt.plot(
            xvalues, yvalues,
            color='k', linestyle=':',
            label='Integral',
        )

        if yscale:
            ax.set_yscale(yscale)

        plt.xlabel(r'$x, \mu$' if xscale == Micro else r'$k$')
        plt.ylabel('$S(x - x_{k})$')

        plt.grid(
            color='grey', linestyle=':',
        )
        plt.legend()

        plt.show()

    # --------        fabric        --------
    @classmethod
    def from_profile(cls, profile: ApertureProfile) -> 'Aperture':
        return cls(profile=profile)


if __name__ == '__main__':

    # detector
    detector = Detector.S8377_256Q

    # aperture
    # profile = RectangularApertureProfileы(detector=detector)
    # profile = VoigtApertureProfile(detector=detector, width=1.6, asymmetry=0, ratio=.1)
    profile = ConvolutionApertureProfile(detector=detector)

    aperture = Aperture(profile=profile)
    aperture.show(xscale=Micro)
