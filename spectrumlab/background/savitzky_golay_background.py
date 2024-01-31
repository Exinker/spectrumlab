from dataclasses import dataclass
from typing import Callable

import numpy as np

from spectrumlab.alias import Array
from spectrumlab.spectrum import Spectrum
from spectrumlab.background.base_background import BaseBackground, BaseBackgroundConfig


@dataclass
class SavitzkyGolayBackgroundConfig(BaseBackgroundConfig):
    width: int
    degree: int

    def __post_init__(self):
        if self.width % 2 == 0:
            raise ValueError('SavitzkyGolay width: {width} is not valid! Use odd value only.')  # FIXME: add custom exception


class SavitzkyGolayBackground(BaseBackground):

    def __init__(self, config: SavitzkyGolayBackgroundConfig):
        super().__init__(config)

    def fit(self, spectrum: Spectrum) -> Array:
        return super().fit(spectrum)


def filter_savitzky_golay(y: Array[float], mask: Array[bool], config: SavitzkyGolayBackgroundConfig) -> Callable:
    """Savitzky-Gloay filter with mask."""

    n = len(y)
    hw = (config.width - 1) / 2

    def inner(i: int) -> float:
        index = np.arange(i-hw, i+hw+1, dtype=int)
        index = index[(index >= 0) & (index < n)]
        index = index[mask[index]]

        #
        if index.size == 0:
            return np.nan

        return np.polyval(
            p=np.polyfit(index, y[index], config.degree),
            x=i,
        )

    return inner


def approximate_savitzky_golay(y: Array[float], mask: Array[bool], config: SavitzkyGolayBackgroundConfig) -> Array[float]:
    """Approximate y values with Savitzky-Gloay filtration."""
    filter = filter_savitzky_golay(y, mask, config=config)

    return np.array([
        filter(i)
        for i, _ in enumerate(y)
    ])

