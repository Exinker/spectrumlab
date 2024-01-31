"""
Base type for any spectrum's background.

Author: Vaschenko Pavel
 Email: vaschenko@vmk.ru
  Date: 2024.01.31
"""
from abc import ABC, abstractmethod

from spectrumlab.alias import Array
from spectrumlab.spectrum import Spectrum


class BaseBackgroundConfig(ABC):
    """Base type for any background's config."""

    pass


class BaseBackground(ABC):
    """Base type for any spectrum's background."""

    def __init__(self, config: BaseBackgroundConfig):
        self.config = config

    # --------        handlers        --------
    @abstractmethod
    def fit(self, spectrum: Spectrum) -> Array[float]:
        raise NotImplementedError
