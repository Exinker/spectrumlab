"""
Abstract type for any spectrum's background.

Author: Vaschenko Pavel
 Email: vaschenko@vmk.ru
  Date: 2024.01.31
"""
from abc import ABC, abstractmethod

from spectrumlab.emulation.noise import Noise
from spectrumlab.spectrum import Spectrum
from spectrumlab.typing import Array


class AbstractBackgroundConfig(ABC):
    """Abstract type for any background's config."""

    pass


class AbstractBackground(ABC):
    """Abstract type for any spectrum's background."""

    def __init__(self, config: AbstractBackgroundConfig):
        self.config = config

    # --------        handlers        --------
    @abstractmethod
    def fit(self, spectrum: Spectrum, noise: Noise, show: bool = False) -> Array[float]:
        raise NotImplementedError
