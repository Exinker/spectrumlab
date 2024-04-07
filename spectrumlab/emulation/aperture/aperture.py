import os
from abc import ABC, abstractproperty
from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, signal

from spectrumlab.grid import Grid
from spectrumlab.emulation.curve import pvoigt, rectangular
from spectrumlab.emulation.detector import Detector
from spectrumlab.picture.config import COLOR
from spectrumlab.typing import Array, MicroMeter, Number


DATASHEET_DIRECTORY = os.path.join(os.path.dirname(__file__), 'datasheet')


# --------        aperture shapes        --------
class AbstractApertureShape(ABC):
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


class RectangularApertureShape(AbstractApertureShape):
    """Rectangular aperture's profile shape."""

    def __init__(self):
        super().__init__()

        self._f = partial(rectangular, x0=0, w=1)

    @property
    def f(self) -> Callable[[Number], Array[float]]:
        return self._f


class RoundedRectangularApertureShape(AbstractApertureShape):
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


class VoigtApertureShape(AbstractApertureShape):
    """Voigt aperture's profile shape."""

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
    def from_ini(cls, detector: Detector, kind: Literal[405] = 405) -> 'VoigtApertureShape':
        params = {
            Detector.BLPP369M1: {
                405: (4.9173/detector.pitch, 0, 1.0000),  # (!) bad approximation
            },
            Detector.BLPP2000: {
                405: (17.9677/detector.pitch, 0, 0.4186),
            },
            Detector.BLPP4000: {
                405: (3.5592/detector.pitch, 0, 0.4578),
            },
        }

        return cls(*params[detector][kind])


class MeasuredApertureShape(AbstractApertureShape):
    """Voigt aperture's profile shape."""

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

        filepath = os.path.join(DATASHEET_DIRECTORY, f'{detector.name}.csv')
        datasheet = np.genfromtxt(
            filepath,
            delimiter=',',
            skip_header=1,
        )

        return cls(
            grid=Grid(
                x=datasheet[:, 0],
                y=datasheet[:, 1],
            ).rescale(
                detector.pitch, units=Number,
            ),
        )


ApertureShape = RectangularApertureShape | RoundedRectangularApertureShape | VoigtApertureShape | MeasuredApertureShape


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
    def pitch(self) -> MicroMeter:
        return self.detector.pitch

    # --------        handlers        --------
    def show(self, rx: MicroMeter = 100, dx: MicroMeter = .01, units: Number | MicroMeter = Number) -> None:
        scale = {
            Number: self.pitch,
            MicroMeter: 1,
        }.get(units)

        #
        x = np.linspace(-rx/2, +rx/2, int(rx/dx) + 1)
        grid = Grid(
            x=x/scale,
            y=self(x, n=0)*scale,
            units=units,
        )

        #
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

        plt.plot(
            grid.x, grid.y,
            color=COLOR['blue'],
            label='$S(x - x_{{0}})$',
        )
        plt.xlabel(grid.xlabel)
        plt.ylabel(r'$S(x - x_{k})$')
        plt.grid(color='grey', linestyle=':')
        plt.legend(loc='upper right')

        plt.show()

        # integral
        n_pixels = int(rx // self.pitch) + 1
        x = np.linspace(0, rx, int(rx/dx) + 1)

        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

        integral = np.zeros(x.shape)
        for n in range(n_pixels):
            grid = Grid(
                x=x/scale,
                y=self(x, n=n)*scale,
                units=units,
            )
            plt.plot(
                grid.x, grid.y,
                color=COLOR['blue'],
                label=r'$S(x - x_{k})$' if n == 0 else None,
            )
            integral += grid.y

        plt.plot(
            grid.x, integral,
            color='k', linestyle=':',
            label='Integral',
        )

        plt.xlabel(grid.xlabel)
        plt.ylabel(r'$S(x - x_{k})$')
        plt.grid(color='grey', linestyle=':')
        plt.legend(loc='upper right')

        plt.show()

    # --------        private        --------
    def __call__(self, x: MicroMeter | Array[MicroMeter], n: Number = 0) -> Array[float]:
        return self.shape(x/self.pitch, n=n)/self.pitch


if __name__ == '__main__':

    # detector
    detector = Detector.BLPP369M1

    # aperture
    aperture = Aperture(
        detector=detector,
        # shape=RectangularApertureShape(),
        # shape=RoundedRectangularApertureShape(),
        # shape=VoigtApertureShape.from_ini(detector=detector),
        shape=MeasuredApertureShape.from_datasheet(detector=detector),
    )
    aperture.show(units=MicroMeter)
