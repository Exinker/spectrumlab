from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, signal

from spectrumlab.alias import Array, MicroMeter, Number
from spectrumlab.emulation.curve import rectangular, pvoigt
from spectrumlab.emulation.detector.linear_array_detector import Detector
from spectrumlab.picture.config import COLOR


# --------        shapes        --------
@dataclass
class RectangularApertureShape:
    """Rectangular aperture's profile shape."""

    def __call__(self, x: Number | Array[Number], n: Number) -> Array[float]:
        f = rectangular(x, x0=n, w=1)

        return f


@dataclass
class RoundedRectangularApertureShape:
    """Rounded rectangular (a convolution of rectangular and pvoigt) aperture's profile shape."""

    width: Number = field(default=.2)
    dx: float = field(default=0.01)  # шаг построения интерполяции
    rx: float = field(default=10)  # границы построения интерполяции

    _x: Array[Number] = field(init=False, repr=False, default=None)
    _f: Array[float] = field(init=False, repr=False, default=None)

    def __post_init__(self):
        x = np.linspace(-self.rx, +self.rx, 2*int(self.rx/self.dx) + 1)

        f1 = lambda x: rectangular(x, x0=0, w=1)
        f2 = lambda x: pvoigt(x, 0, w=self.width, a=0, r=0)
        f = signal.convolve(f1(x), f2(x), mode='same') * (x[-1] - x[0])/len(x)

        self._x = x
        self._f = f

    def __call__(self, x: Number | Array[Number], n: Number) -> float | Array[float]:
        F = interpolate.interp1d(
            self._x,
            self._f,
            kind='linear',
            bounds_error=False,
            fill_value=0,
        )
        f = F(x - n)

        return f
    

# --------        approximated shape        --------
@dataclass
class ApproximatedApertureShape:
    """Approximated aperture's profile shape."""
    detector: Detector

    wavelength: Literal[405] = field(default=405)
    dx: float = field(default=0.01)  # шаг построения интерполяции
    rx: float = field(default=10)  # границы построения интерполяции

    _x: Array[Number] = field(init=False, repr=False, default=None)
    _f: Array[float] = field(init=False, repr=False, default=None)

    PARAMS = {
        405: {
            Detector.BLPP369M1: (4.9173, 0, 1.0000),  # (!) bad approximation
            Detector.BLPP2000: (17.9677, 0, 0.4186),
            Detector.BLPP4000: (3.5592, 0, 0.4578),
        },  # aperture's shape params at 405 nm
    }

    def __post_init__(self):
        detector = self.detector
        wavelength = self.wavelength
        w, a, r = self.PARAMS[wavelength][detector]

        #
        x = np.linspace(-self.rx, +self.rx, 2*int(self.rx/self.dx) + 1)

        f1 = lambda x: rectangular(x, x0=0, w=1)
        f2 = lambda x: pvoigt(x, 0, w=w, a=a, r=r)
        f = signal.convolve(f1(x), f2(x), mode='same') * (x[-1] - x[0])/len(x)

        self._x = x
        self._f = f

    def __call__(self, x: Number | Array[Number], n: Number) -> float | Array[float]:
        F = interpolate.interp1d(
            self._x,
            self._f,
            kind='linear',
            bounds_error=False,
            fill_value=0,
        )
        f = F(x - n)

        return f


ApertureShape = RectangularApertureShape | RoundedRectangularApertureShape | ApproximatedApertureShape


# --------        aperture interface        --------
@dataclass(frozen=True)
class Aperture:
    """
    Interface for any detectors's aperture profile.

    Author: Vaschenko Pavel
     Email: vaschenko@vmk.ru
      Date: 2014.03.24
    """
    detector: Detector
    shape: ApertureShape

    @property
    def step(self) -> MicroMeter:
        return self.detector.config.width

    # --------        handlers        --------
    def show(self, rx: MicroMeter = 100, dx: MicroMeter = .01, xscale: Number | MicroMeter = Number) -> None:
        n_steps = rx // self.step + 1

        #
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

        xvalues = np.linspace(0, rx, int(rx/dx) + 1)
        integral = np.zeros(xvalues.shape)
        for n in range(n_steps):
            x = xvalues if xscale == MicroMeter else xvalues/self.step
            y = self(xvalues, n=n)
            plt.plot(
                x, y,
                color=COLOR['blue'],
                label='$S(x - x_{k})$' if n==0 else None,
            )

            integral += y

        x = xvalues if xscale == MicroMeter else xvalues/self.step
        y = integral
        plt.plot(
            x, y,
            color='k', linestyle=':',
            label='Integral',
        )

        plt.xlabel(r'$x$ $[\mu]$' if xscale == MicroMeter else r'$k$')
        plt.ylabel('$S(x - x_{k})$')
        plt.grid(
            color='grey', linestyle=':',
        )
        plt.legend()

        plt.show()

    # --------        private        --------
    def __call__(self, x: MicroMeter | Array[MicroMeter], n: Number = 0) -> Array[float]:
        return self.shape(x/self.step, n=n)/self.step


if __name__ == '__main__':

    # detector
    detector = Detector.BLPP2000

    # aperture
    aperture = Aperture(
        detector=detector,
        # shape=RectangularApertureShape(),
        shape=RoundedRectangularApertureShape(),
    )
    aperture.show()
