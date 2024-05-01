from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from scipy import interpolate, signal

from spectrumlab.emulation.curve import gauss, rectangular
from spectrumlab.types import Array, File, Meter, NanoMeter


class AbstractCharacteristic(ABC):
    """Abstract type for any characteristic."""

    # --------        private        --------
    @abstractmethod
    def __call__(self, x: Array[Meter], fill_value: float = np.nan) -> Array[float]:
        raise NotImplementedError


@dataclass(frozen=True)
class ConstantCharacteristic(AbstractCharacteristic):
    value: float

    # --------        private        --------
    def __call__(self, x: float | Array[Meter], fill_value: float = np.nan) -> Array[float]:

        if isinstance(x, float):  # TODO: don't remove! It's for integrate.quad functions!
            return self.value
        if isinstance(x, Array):
            return np.full(x.shape, self.value)

        raise ValueError


@dataclass(frozen=True)
class WindowCharacteristic(AbstractCharacteristic):
    span: tuple[NanoMeter, NanoMeter]
    smooth: float | None  # smoothing rectangular edges by gauss
    wavelength_bounds: tuple[float]
    wavelength_step: float
    _x: Array[Meter] = field(init=False, repr=False)
    _y: Array[float] = field(init=False, repr=False)

    # --------        private        --------
    def __post_init__(self):
        lb, ub = self.wavelength_bounds
        step = self.wavelength_step
        position = np.sum(self.span) / 2
        width = self.span[-1] - self.span[0]

        if self.smooth is None:
            x = np.arange(lb, ub+1, 1)
            y = width * rectangular(x, position, w=width)

        else:
            grid_x = np.arange(-10 * width, +10 * width, 1)
            R = lambda x: width * rectangular(x, 0, w=width)
            G = lambda x: gauss(x, 0, w=self.smooth)
            grid_f = signal.convolve(R(grid_x), G(grid_x), mode='same') * (grid_x[-1] - grid_x[0])/len(grid_x)

            x = np.linspace(lb, ub, int((ub - lb)/step) + 1)
            y = interpolate.interp1d(
                grid_x,
                grid_f,
                kind='linear',
                bounds_error=False,
                fill_value=0,
            )(x - position)

        self._x = x / 1e+9  # transform x values to meter units
        self._y = y

    def __call__(self, x: float | Array[Meter], fill_value: float = np.nan) -> Array[float]:
        return interpolate.interp1d(
            self._x, self._y,
            kind='linear', bounds_error=False, fill_value=fill_value,
        )(x)


@dataclass(frozen=True)
class DatasheetCharacteristic(AbstractCharacteristic):
    path: File
    xscale: float = field(default=1)  # transform x values to meter units
    yscale: float = field(default=1)  # normalize y values to scale
    delimiter: str = field(default=',')

    _x: Array[NanoMeter] = field(init=False, repr=False)
    _y: Array[float] = field(init=False, repr=False)

    # --------        private        --------
    def __post_init__(self):
        dat = np.genfromtxt(self.path, delimiter=self.delimiter, dtype=np.float32)

        self._x = dat[:, 0] / self.xscale
        self._y = dat[:, 1] / self.yscale

    def __call__(self, x: Meter | Array[Meter], fill_value: float = np.nan) -> Array[float]:
        return interpolate.interp1d(
            self._x, self._y,
            kind='linear', bounds_error=False, fill_value=fill_value,
        )(x)
