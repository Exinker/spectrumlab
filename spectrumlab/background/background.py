"""
Abstract type for any spectrum's background.

Author: Vaschenko Pavel
 Email: vaschenko@vmk.ru
  Date: 2024.01.31
"""
import abc

from spectrumlab.emulations.noises import Noise
from spectrumlab.spectrum import Spectrum
from spectrumlab.types import Array


class AbstractBackgroundConfig(abc.ABC):
    """Abstract type for any background's config."""

    pass


class AbstractBackground(abc.ABC):
    """Abstract type for any spectrum's background."""

    def __init__(self, config: AbstractBackgroundConfig):
        self.config = config

    # --------        handlers        --------
    @abc.abstractmethod
    def fit(self, spectrum: Spectrum, noise: Noise, show: bool = False) -> Array[float]:
        raise NotImplementedError
