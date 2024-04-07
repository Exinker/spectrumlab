from abc import ABC
from functools import partial
from typing import Callable, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, optimize

from spectrumlab.grid import Grid, T
from spectrumlab.grid.shape import VoigtGridShape
from spectrumlab.typing import Array, MicroMeter


class AbstractGridFilter(ABC):

    def __init__(self, grid: Grid):
        self._grid = grid

    @property
    def grid(self) -> Grid:
        return self._grid

    @property
    def f(self) -> Callable[[Array[T]], Array[float]]:
        return self._f

    # --------        handlers        --------
    def show(self, bias: T = 0):
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

        x, y = self.grid.x, self.grid.y
        plt.plot(
            x - bias, y,
            color='red', linestyle='none', marker='s', markersize=3,
            alpha=1,
        )

        x = self.grid.space()
        y_hat = self.f(x)
        plt.plot(
            x - bias, y_hat,
            color='black', linestyle='-', linewidth=1,
            alpha=1,
        )

        x, y = self.grid.x, self.grid.y
        y_hat = self.f(x)
        plt.plot(
            x - bias, y - y_hat,
            color='black', linestyle='none', marker='s', markersize=0.5,
            alpha=1,
        )

        plt.xlabel(self.grid.xlabel)
        plt.ylabel(r'$f(x)$')
        plt.grid(color='grey', linestyle=':')

        plt.show()

    # --------        private        --------
    def __call__(self, x: Array) -> Array:
        return self.f(x)


class LinearInterpolationGridFilter(AbstractGridFilter):

    def __init__(self, grid: Grid, show: bool = False):
        super().__init__(grid=grid)

        #
        self._f = interpolate.interp1d(
            grid.x, grid.y,
            kind='linear',
            bounds_error=False,
            fill_value=0,
        )

        # show
        if show:
            self.show()


class VoigtGridShapeFilter(AbstractGridFilter):

    def __init__(self, grid: Grid, pitch: T, show: bool = False):
        super().__init__(grid=grid)

        def _loss(grid: Grid, pitch: T, position: MicroMeter, width: MicroMeter, asymmetry: float, ratio: float, intensity: float) -> float:
            shape = VoigtGridShape(
                width=width,
                asymmetry=asymmetry,
                ratio=ratio,
                pitch=pitch,
            )

            f = partial(shape, position=position, intensity=intensity)
            return np.sum(
                (grid.y - f(grid.x))**2
            )

        # shape
        x0 = grid.x[np.argmax(grid.y)]
        position, width, asymmetry, ratio, intensity = optimize.minimize(
            lambda x: _loss(grid, pitch, *x),
            x0=[x0, pitch, 0, .1, np.sum(grid.y) / pitch],
            bounds=[(x0-pitch/2, x0+pitch/2), (pitch/2, 100), (-1, 1), (0, 1), (0, np.inf)],
        )['x']

        shape = VoigtGridShape(
            width=width,
            asymmetry=asymmetry,
            ratio=ratio,
            pitch=pitch,
        )

        # f
        self._f = partial(shape, position=position, intensity=intensity)

        # show
        if show:
            self.show()
