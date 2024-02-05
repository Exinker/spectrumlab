from collections.abc import Iterator
from typing import Callable, TypeVar
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, interpolate

from spectrumlab.alias import Array, MicroMeter, NanoMeter, Number, PicoMeter


T = TypeVar('T', Number, MicroMeter, NanoMeter, PicoMeter)


class _GridIterator:

    def __init__(self, x: Array[T], y: Array[float]):
        self.x = x
        self.y = y

        self._index = -1

    # --------        private        --------
    def __iter__(self) -> Iterator:
        return self

    def __next__(self) -> tuple[float, float]:

        try:
            self._index += 1
            return self.x[self._index], self.y[self._index]

        except IndexError:
            raise StopIteration


class Grid:

    def __init__(self, x: Array[T], y: Array[float] | None = None, units: T | None = None):
        assert len(x) == len(y)

        #
        self._x = x
        self._y = y
        self._units = units

    @property
    def x(self) -> Array[T]:
        return self._x

    @property
    def y(self) -> Array[float]:
        return self._y

    @property
    def units(self) -> T | None:
        return self._units

    @property
    def xlabel(self) -> str:
        return '{label} {units}'.format(
            label={
                Number: r'$number$',
                MicroMeter: r'$x$',
                PicoMeter: r'$x$',
            }.get(self.units, ''),
            units=self.xunits,
        )

    @property
    def xunits(self) -> str:
        return {
            Number: r'',
            MicroMeter: r'[$\mu m$]',
            PicoMeter: r'[$pm$]',
        }.get(self.units, '')

    @property
    def interpolate(self) -> Callable[[Array[T]], Array[float]]:
        """Interpolate `grid` by linear interpolate."""

        return interpolate.interp1d(
            self.x, self.y,
            kind='linear',
            bounds_error=False,
            fill_value=0,
        )

    # --------        handlers        --------
    def space(self, n_points: int = 1000) -> Array[T]:
        return np.linspace(min(self.x), max(self.x), n_points)

    def shift(self, value: T) -> 'Grid':
        """Shift `grid` by the `value`."""

        return Grid(
            x=self.x - value,
            y=self.y,
            units=self.units,
        )

    def rescale(self, value: float, units: T) -> 'Grid':
        """Rescale `grid` by the `value`. It is used to change `units`!"""

        return Grid(
            x=self.x/value,
            y=self.y*value,
            units=units,
        )

    def normalize(self, value: float | None = None) -> 'Grid':
        """Normalize `grid`."""
        value = value or 1/integrate.quad(self.interpolate, a=min(self.x), b=max(self.x))[0]

        return Grid(
            x=self.x,
            y=self.y*value,
            units=self.units,
        )

    def show(self) -> None:
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

        x, y = self.x, self.y
        plt.plot(
            x, y,
            color='red', linestyle='none', marker='s', markersize=3,
            alpha=1,
        )

        plt.xlabel(self.xlabel)
        plt.ylabel(r'$f$')
        plt.grid(color='grey', linestyle=':')

        plt.show()

    # --------        private        --------
    def __len__(self) -> int:
        return len(self.x)

    def __iter__(self) -> Iterator:
        warn(
            message='Iteration on the `grid` by points will be removed in the future!',
            category=DeprecationWarning,
            stacklevel=1,
        )

        return _GridIterator(
            x=self.x,
            y=self.y,
        )

    def __str__(self) -> str:
        cls = self.__class__

        return f'{cls.__name__}({self.units})'

    def __add__(self, other: float | Array[float]) -> 'Grid':
        cls = self.__class__

        return cls(
            x=self.x,
            y=self.y + other,
            units=self.units,
        )

    def __iadd__(self, other: float | Array[float]) -> 'Grid':
        return self + other

    def __radd__(self, other: float | Array[float]) -> 'Grid':
        return self + other

    def __sub__(self, other: float | Array[float]) -> 'Grid':
        cls = self.__class__

        return cls(
            x=self.x,
            y=self.y - other,
            units=self.units,
        )

    def __isub__(self, other: float | Array[float]) -> 'Grid':
        return self - other

    def __rsub__(self, other: float | Array[float]) -> 'Grid':
        return self - other
