from dataclasses import dataclass, field

import numpy as np

from spectrumlab.types import Array, Number


@dataclass(slots=True)
class PeakABC:

    minima: tuple[Number, Number]
    maxima: tuple[Number] | tuple[Number, Number] | tuple[Number, ...]

    except_edges: bool = field(default=False, repr=False)

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

        if len(self.maxima) <= 1:
            return self.index

        left, *_, right = self.maxima
        return self.index[(self.number < left) | (self.number > right)]

    @property
    def number(self) -> Array[Number]:
        """External index of all peak."""

        left, right = self.minima
        number = np.arange(left, right+1)

        return number[self.index]

    def include(self, n: Number) -> bool:
        """Check if `n` is included in the peak's number."""

        return n in self.number
