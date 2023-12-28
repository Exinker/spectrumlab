from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from spectrumlab.alias import Number
from spectrumlab.peak.shape.grid import Grid


@dataclass
class Variable:
    name: str
    initial: float
    bounds: tuple[float, float]
    value: float | None


class BaseVariables(Mapping):
    """Base variables type."""

    def __init__(self, __items: Sequence[Variable]) -> None:
        self._items = {
            item.name: item
            for item in __items
        }

    @property
    def initial(self) -> tuple[float]:
        return tuple(self._items[key].initial for key in self.keys())

    @property
    def bounds(self) -> tuple[tuple[float, float]]:
        return tuple(self._items[key].bounds for key in self.keys())

    @property
    def value(self) -> tuple[float] | tuple[None]:
        return tuple(self._items[key].value for key in self.keys())

    # --------        private        --------
    def __repr__(self) -> str:
        cls = self.__class__

        content = '\n\t'.join([
            f'{self._items[key]},'
            for key in self.keys()
        ])
        return f'{cls.__name__}(\n\t{content}\n)'

    def __getitem__(self, __key: str) -> float | None:
        item = self._items[__key]
        return item.value

    def __iter__(self) -> Iterator:
        return iter(key for key in self._items.keys())

    def __len__(self) -> int:
        return len(self._items)


class ScopeVariables(BaseVariables):

    def __init__(self, grid: Grid, position: Number | None = None, intensity: float | None = None, background: float | None = None):
        super().__init__([
            self._init_position(grid, position=position),
            self._init_intensity(grid, intensity=intensity),
            self._init_background(grid, background=background),
        ])

        self.name = 'scope'

    # --------        private        --------
    def _init_position(self, grid: Grid, position: Number | None = None) -> Variable:
        initial = grid.xvalues[np.argmax(grid.yvalues)] if position is None else position
        bounds = (initial-2, initial+2) if position is None else (initial-1e-10, initial+1e-10)
        final = position

        return Variable('position', initial, bounds, final)

    def _init_intensity(self, grid: Grid, intensity: float | None = None) -> Variable:
        initial = max(grid.yvalues) if intensity is None else intensity
        bounds = (0, +np.inf) if intensity is None else (intensity - 1e-10, intensity + 1e-10)
        final = intensity

        return Variable('intensity', initial, bounds, final)

    def _init_background(self, grid: Grid, background: float | None = None) -> Variable:
        initial = min(grid.yvalues) if background is None else background
        bounds = (min(grid.yvalues), max(grid.yvalues)) if background is None else (background - 1e-10, background + 1e-10)
        final = background

        return Variable('background', initial, bounds, final)


class VoightVariables(BaseVariables):

    def __init__(self, width: Number | None = None, asymmetry: float | None = None, ratio: float | None = None):
        super().__init__([
            Variable('width', 2.0, (0.1, 20), width),
            Variable('asymmetry', 0.0, (-0.5, +0.5), asymmetry),
            Variable('ratio', 0.1, (0, 1), ratio),
        ])

        self.name = 'shape'


if __name__ == '__main__':

    vars = BaseVariables(
        [
            Variable(f'test{i}', i, (i-1, i+1), None)
            for i in range(10)
        ]
    )
    print(vars)
