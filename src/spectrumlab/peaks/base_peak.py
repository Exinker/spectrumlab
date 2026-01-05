from dataclasses import dataclass, field

import numpy as np

from spectrumlab.types import Array, Number


@dataclass(slots=True)
class PeakABC:

    minima: tuple[Number, Number]  # spectrum's internal index of the minima
    maxima: tuple[Number] | tuple[Number, Number] | tuple[Number, ...]  # spectrum's internal index of the maximum (or indices, if line has a self-absorption)  # noqa: E501

    except_edges: bool = field(default=False)

    def __repr__(self) -> str:
        cls = self.__class__

        content = '; '.join([
            f'minima: {self.minima}',
            f'maxima: {self.maxima}',
        ])
        return f'{cls.__name__}({content})'

    @property
    def n_numbers(self) -> int:
        left, right = self.minima

        return right - left + 1

    @property
    def index(self) -> Array[int]:
        """Internal index of all peak."""

        if self.except_edges:
            return np.arange(1, self.n_numbers-1)

        return np.arange(self.n_numbers)

    @property
    def tail(self) -> Array[int]:
        """Internal index of peak's tail."""
        maxima = self.maxima
        index = self.index

        # пустой пик (нет отсчетов)
        if len(maxima) == 0:
            return index

        # обычный пик
        if len(maxima) == 1:
            return index

        # пик с провалом (с зашкалом (FIXME: fix it) или с самопоглощением)
        if len(maxima) == 2:
            left, right = maxima
            return index[(index < left) | (index > right)]

        # пик с несколькими провалами (с зашкалом?)
        left, *_, right = maxima
        return index[(index < left) | (index > right)]

    @property
    def number(self) -> Array[Number]:
        """External index of all peak."""
        index = self.index
        left, right = self.minima

        number = np.arange(left, right+1)

        return number[index]

    def include(self, n: Number) -> bool:
        """Is `n` included in a range of the peak's number?"""
        return n in self.number
