from dataclasses import dataclass
from typing import Callable

import numpy as np

from spectrumlab.alias import Array


@dataclass
class SavitzkyGolayConfig:
    width: int
    degree: int

    def __post_init__(self):
        if self.width % 2 == 0:
            raise ValueError('SavitzkyGolay width: {width} is not valid! Use odd value only.')  # FIXME: add custom exception


def filter_savitzky_golay(y: Array[float], mask: Array[bool], config: SavitzkyGolayConfig) -> Callable:
    """Savitzky-Gloay filter with mask."""

    n = len(y)
    hw = (config.width - 1) / 2

    def inner(i: int) -> float:
        index = np.arange(i-hw, i+hw+1, dtype=int)
        index = index[(index >= 0) & (index < n)]
        index = index[mask[index]]

        p = np.polyfit(index, y[index], config.degree)
        return np.polyval(p, i)

    return inner


def approximate_savitzky_golay(y: Array[float], mask: Array[bool], config: SavitzkyGolayConfig) -> Array[float]:
    """Approximate y values with Savitzky-Gloay filtration."""
    filter = filter_savitzky_golay(y, mask, config=config)

    return np.array([
        filter(i)
        for i, _ in enumerate(y)
    ])
