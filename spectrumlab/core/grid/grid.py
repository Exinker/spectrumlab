from collections.abc import Iterator
from typing import Callable, TypeVar

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, interpolate

from spectrumlab.alias import Array, Number, MicroMeter, NanoMeter, PicoMeter


T = TypeVar('T', Number, MicroMeter, NanoMeter, PicoMeter)


class GridIterator:

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

    def __init__(self, x: Array[T], y: Array[float]):
        assert len(x) == len(y)

        self._x = x
        self._y = y

    @property
    def x(self) -> Array[T]:
        return self._x

    @property
    def y(self) -> Array[float]:
        return self._y

    @property
    def n_points(self) -> int:
        return len(self.x)

    @property
    def interpolation(self) -> Callable[[Array[T]], Array[float]]:
        """Interpolate `grid` by linear interpolation."""

        return interpolate.interp1d(
            self.x, self.y,
            kind='linear',
            bounds_error=False,
            fill_value=0,
        )

    # --------        handlers        --------
    def space(self, n_points: int = 1000) -> Array[T]:
        return np.linspace(min(self.x), max(self.x), n_points)

    def xscale(self, scale: T | None = None, bias: T | None = None) -> 'Grid':
        """Scale `x` values of the `grid`."""
        if scale is None: scale = 1
        if bias is None: bias = 0

        return Grid(
            x=scale*self.x - bias,
            y=self.y,
        )

    def yscale(self, scale: float | None = None) -> 'Grid':
        """Scale `y` values of the `grid`."""
        if scale is None: scale = 1/integrate.quad(self.interpolation, a=min(self.x), b=max(self.x))[0]

        return Grid(
            x=self.x,
            y=scale*self.y,
        )

    def show(self) -> None:
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

        x, y = self.x, self.y
        plt.plot(
            x, y,
            color='red', linestyle='none', marker='s', markersize=3,
            alpha=1,
        )

        plt.xlabel(r'$number$')
        plt.ylabel(r'$f$')
        plt.grid(color='grey', linestyle=':')

        plt.show()

    # --------        private        --------
    def __iter__(self) -> Iterator:
        return GridIterator(
            x=self.x,
            y=self.y,
        )

    def __repr__(self) -> str:
        cls = self.__class__

        return f'{cls.__name__}(n_points={self.n_points})'
