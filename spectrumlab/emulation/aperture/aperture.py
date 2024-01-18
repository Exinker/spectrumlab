import os
from abc import ABC, abstractproperty
from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, signal

from spectrumlab.alias import Array, MicroMeter, Number
from spectrumlab.core.grid import Grid
from spectrumlab.emulation.curve import rectangular, pvoigt
from spectrumlab.emulation.detector import Detector
from spectrumlab.picture.config import COLOR


# --------        shapes        --------
class BaseApertureShape(ABC):
    dx: Number = 1e-2  # шаг построения интерполяции
    rx: Number = 10  # границы построения интерполяции

    @property
    def x(self) -> Array[Number]:
        return np.linspace(-self.rx, +self.rx, 2*int(self.rx/self.dx) + 1)

    @abstractproperty
    def f(self, x: Array[Number]) -> Callable[[Number], Array[float]]:
        pass

    def __call__(self, x: Number | Array[Number], n: Number) -> float | Array[float]:
        return self.f(x - n)


class RectangularApertureShape(BaseApertureShape):
    """Rectangular aperture's profile shape."""

    def __init__(self):
        super().__init__()

        self._f = partial(rectangular, x0=0, w=1)

    @property
    def f(self) -> Callable[[Number], Array[float]]:
        return self._f


class RoundedRectangularApertureShape(BaseApertureShape):
    """Rounded rectangular (a convolution of rectangular and pvoigt) aperture's profile shape."""

    def __init__(self, width: Number = .2):
        super().__init__()
        self.width = width

        x = self.x
        dx = self.dx
        self._f = interpolate.interp1d(
            x,
            signal.convolve(
                rectangular(x, x0=0, w=1),
                pvoigt(x, 0, w=self.width, a=0, r=0),
                mode='same',
            ) * dx,
            kind='linear',
            bounds_error=False,
            fill_value=0,
        )

    @property
    def f(self) -> Callable[[Number], Array[float]]:
        return self._f


class VoightApertureShape(BaseApertureShape):
    """Voight aperture's profile shape."""

    def __init__(self, width: Number, asymmetry: float, ratio: float):
        super().__init__()
        self.width = width
        self.asymmetry = asymmetry
        self.ratio = ratio

        x = self.x
        dx = self.dx
        self._f = interpolate.interp1d(
            x,
            signal.convolve(
                rectangular(x, x0=0, w=1),
                pvoigt(x, 0, w=self.width, a=self.asymmetry, r=self.ratio),
                mode='same',
            ) * dx,
            kind='linear',
            bounds_error=False,
            fill_value=0,
        )

    @property
    def f(self) -> Callable[[Number], Array[float]]:
        return self._f

    # --------        fabric        --------
    @classmethod
    def from_ini(cls, detector: Detector, kind: Literal[405] = 405) -> 'VoightApertureShape':
        PARAMS = {
            Detector.BLPP369M1: {
                405: (4.9173 / detector.config.width, 0, 1.0000),  # (!) bad approximation
            },
            Detector.BLPP2000: {
                405: (17.9677 / detector.config.width, 0, 0.4186),
            },
            Detector.BLPP4000: {
                405: (3.5592 / detector.config.width, 0, 0.4578),
            },
        }

        return cls(*PARAMS[detector][kind])


class MeasuredApertureShape(BaseApertureShape):
    """Voight aperture's profile shape."""

    def __init__(self, grid: Grid):
        super().__init__()

        self._f = interpolate.interp1d(
            grid.x,
            grid.y,
            kind='linear',
            bounds_error=False,
            fill_value=0,
        )

    @property
    def f(self) -> Callable[[Number], Array[float]]:
        return self._f

    # --------        fabric        --------
    @classmethod
    def from_datasheet(cls, detector: Detector) -> 'MeasuredApertureShape':

        filepath = os.path.join(os.path.dirname(__file__), 'datasheet', f'{detector.name}.csv')
        datasheet = np.genfromtxt(
            filepath,
            delimiter=',',
            skip_header=1,
        )

        return cls(
            grid=Grid(
                x=datasheet[:,0]/detector.config.width,
                y=datasheet[:,1],
            ),
        )



ApertureShape = RectangularApertureShape | RoundedRectangularApertureShape | VoightApertureShape


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
        plt.legend(loc='upper right')

        plt.show()

    # --------        private        --------
    def __call__(self, x: MicroMeter | Array[MicroMeter], n: Number = 0) -> Array[float]:
        return self.shape(x/self.step, n=n)/self.step


if __name__ == '__main__':

    # detector
    detector = Detector.BLPP4000

    # aperture
    aperture = Aperture(
        detector=detector,
        # shape=RectangularApertureShape(),
        # shape=RoundedRectangularApertureShape(),
        # shape=VoightApertureShape.from_ini(detector=detector),
        shape=MeasuredApertureShape.from_datasheet(detector=detector),
    )
    aperture.show()
