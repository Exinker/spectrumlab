"""
Base type for any spectrum's background.

Author: Vaschenko Pavel
 Email: vaschenko@vmk.ru
  Date: 2024.01.31
"""
from abc import ABC, abstractmethod

from spectrumlab.noises import Noise
from spectrumlab.spectra import Spectrum
from spectrumlab.types import Array


class BackgroundConfigABC(ABC):

    pass


class BackgroundABC(ABC):

    def __init__(self, config: BackgroundConfigABC):
        self.config = config

    @abstractmethod
    def fit(self, spectrum: Spectrum, noise: Noise, show: bool = False) -> Array[float]:
        raise NotImplementedError
