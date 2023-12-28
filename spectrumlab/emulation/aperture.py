from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, signal

from spectrumlab.alias import Array, Micro, Number
from spectrumlab.emulation.curve import rectangular, pvoigt
from spectrumlab.emulation.detector.linear_array_detector import Detector
from spectrumlab.picture.config import COLOR


# --------        shape        --------
@dataclass
class RectangularApertureShape:
    """Rectangular detector's aperture shape."""
    detector: Detector

    def __call__(self, x: Micro | Array[Micro], n: int) -> Array[float]:
        detector = self.detector
        step = detector.config.width

        f = rectangular(x, x0=step*n, w=step)

        return f


@dataclass
class RoundedRectangularApertureShape:
    """Rounded rectangular (a convolution of rectangular and pvoigt) detector's aperture shape."""
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

    def __call__(self, x: Micro | Array[Micro], n: int) -> float | Array[float]:
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
    

# --------        approximated shape        --------
@dataclass
class ApproximatedApertureShape:
    """Approximated detector's aperture shape."""
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
        },  # aperture's shape params at 405 nm
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

    def __call__(self, x: Micro | Array[Micro], n: int) -> float | Array[float]:
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


ApertureShape = RectangularApertureShape | RoundedRectangularApertureShape | ApproximatedApertureShape


# --------        aperture        --------
@dataclass
class Aperture:
    """
    Interface for any aperture's shape.

    Author: Vaschenko Pavel
     Email: vaschenko@vmk.ru
      Date: 2014.03.24
    """
    shape: ApertureShape

    def __call__(self, x: Micro | Array[Micro], n: int) -> Array[float]:
        return self.shape(x, n=n)

    @property
    def detector(self) -> Detector:
        return self.shape.detector

    # --------        handlers        --------
    def show(self, rx: Micro = 100, dx: Micro = .01, xscale: Number | Micro = Number) -> None:
        step = self.detector.config.width
        n_steps = rx // step + 1

        #
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

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

        plt.xlabel(r'$x$ $[\mu]$' if xscale == Micro else r'$k$')
        plt.ylabel('$S(x - x_{k})$')
        plt.grid(
            color='grey', linestyle=':',
        )
        plt.legend()

        plt.show()

    # --------        fabric        --------
    @classmethod
    def from_shape(cls, shape: ApertureShape) -> 'Aperture':
        return cls(shape=shape)


if __name__ == '__main__':

    # detector
    detector = Detector.BLPP2000

    # aperture
    aperture = Aperture(
        shape=RectangularApertureShape(detector=detector),
        # shape=VoigtApertureShape(detector=detector, width=1.6, asymmetry=0, ratio=.1),
    )
    aperture.show()
