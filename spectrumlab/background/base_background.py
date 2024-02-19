"""
Base type for any spectrum's background.

Author: Vaschenko Pavel
 Email: vaschenko@vmk.ru
  Date: 2024.01.31
"""
from abc import ABC, abstractmethod

from spectrumlab.emulation.noise import Noise
from spectrumlab.spectrum import Spectrum
from spectrumlab.typing import Array


class BaseBackgroundConfig(ABC):
    """Base type for any background's config."""

    pass


class BaseBackground(ABC):
    """Base type for any spectrum's background."""

    def __init__(self, config: BaseBackgroundConfig):
        self.config = config

    # --------        handlers        --------
    @abstractmethod
    def fit(self, spectrum: Spectrum, noise: Noise, show: bool = False) -> Array[float]:
        raise NotImplementedError
