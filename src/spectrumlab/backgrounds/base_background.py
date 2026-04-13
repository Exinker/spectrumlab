"""
Base type for any spectrum's background.

Author: Vaschenko Pavel
 Email: vaschenko@vmk.ru
  Date: 2024.01.31
"""
from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict

from spectrumlab.noises import Noise
from spectrumlab.spectra import Spectrum


class BackgroundConfigABC(BaseModel):

    model_config = ConfigDict(
        extra='forbid',
    )


class BackgroundModelABC(ABC):

    def __init__(self, config: BackgroundConfigABC):

        self._config = config

    @property
    def config(self) -> BackgroundConfigABC:
        return self._config

    @abstractmethod
    def fit(self, spectrum: Spectrum, noise: Noise, show: bool = False) -> Spectrum:
        raise NotImplementedError
