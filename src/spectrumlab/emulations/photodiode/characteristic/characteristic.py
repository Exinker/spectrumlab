from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from scipy import interpolate, signal

from spectrumlab.emulations.curves import gauss, rectangular
from spectrumlab.types import Array, FilePath, Meter, NanoMeter


class AbstractCharacteristic(ABC):
    """Abstract type for any characteristic."""

    @abstractmethod
    def __call__(self, x: Array[Meter], fill_value: float = np.nan) -> Array[float]:
        raise NotImplementedError


@dataclass(frozen=True)
class ConstantCharacteristic(AbstractCharacteristic):
    value: float

    def __call__(self, x: float | Array[Meter], fill_value: float = np.nan) -> Array[float]:

        if isinstance(x, float):  # TODO: don't remove! It's for integrate.quad functions!
            return self.value
        if isinstance(x, Array):
            return np.full(x.shape, self.value)

        raise ValueError


@dataclass(frozen=True)
class WindowCharacteristic(AbstractCharacteristic):

    x: Array[Meter] = field(init=False, repr=False)
    y: Array[float] = field(init=False, repr=False)

    @classmethod
    def create(
        cls,
        span: tuple[NanoMeter, NanoMeter],
        smooth: float | None,  # smoothing rectangular edges by gauss
        wavelength_bounds: tuple[float],
        wavelength_step: float,
    ) -> 'WindowCharacteristic':

        lb, ub = wavelength_bounds
        step = wavelength_step
        position = np.sum(span) / 2
        width = span[-1] - span[0]

        if smooth is None:
            x = np.arange(lb, ub+1, 1)
            y = width * rectangular(x, position, w=width)

            return cls(
                x=1e-9*x,  # transform x values to meter units
                y=y,
            )

        grid_x = np.arange(-10 * width, +10 * width, 1)
        grid_f = signal.convolve(
            (lambda x: width * rectangular(x, 0, w=width))(grid_x),
            (lambda x: gauss(x, 0, w=smooth))(grid_x),
            mode='same',
        ) * (grid_x[-1] - grid_x[0])/len(grid_x)

        x = np.linspace(lb, ub, int((ub - lb)/step) + 1)
        y = interpolate.interp1d(
            grid_x,
            grid_f,
            kind='linear',
            bounds_error=False,
            fill_value=0,
        )(x - position)

        return cls(
            x=1e-9*x,  # transform x values to meter units
            y=y,
        )

    def __call__(self, x: float | Array[Meter], fill_value: float = np.nan) -> Array[float]:
        return interpolate.interp1d(
            self.x, self.y,
            kind='linear', bounds_error=False, fill_value=fill_value,
        )(x)


@dataclass(frozen=True)
class DatasheetCharacteristic(AbstractCharacteristic):

    x: Array[NanoMeter] = field(repr=False)
    y: Array[float] = field(repr=False)

    @classmethod
    def create(
        cls,
        path: FilePath,
        delimiter: str = ',',
        xscale: float = 1,  # transform x values to meter units
        yscale: float = 1,  # normalize y values to scale
    ) -> 'DatasheetCharacteristic':

        dat = np.genfromtxt(path, delimiter=delimiter, dtype=np.float32)

        return cls(
            x=dat[:, 0]/xscale,
            y=dat[:, 1]/yscale,
        )

    def __call__(self, x: Meter | Array[Meter], fill_value: float = np.nan) -> Array[float]:
        return interpolate.interp1d(
            self.x, self.y,
            kind='linear', bounds_error=False, fill_value=fill_value,
        )(x)
