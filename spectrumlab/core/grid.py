from collections.abc import Iterator

import numpy as np
import matplotlib.pyplot as plt

from spectrumlab.alias import Array, Number, MicroMeter


class GridIterator:

    def __init__(self, x: Array[Number], y: Array[float]):
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

    def __init__(self, x: Array[Number], y: Array[float]):
        assert len(x) == len(y)

        self._x = x
        self._y = y

    @property
    def x(self) -> Array[Number]:
        return self._x

    @property
    def y(self) -> Array[float]:
        return self._y

    @property
    def n_points(self) -> int:
        return len(self.x)

    # --------        handlers        --------
    def space(self, n_points: int = 1000) -> Array[Number]:
        return np.linspace(min(self.x), max(self.x), n_points)

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
