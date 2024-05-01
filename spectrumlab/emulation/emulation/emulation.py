"""
Emitted and absorbed spectrum emulation.

    Author: Vaschenko Pavel
    Email: vaschenko@vmk.ru
    Date: 2013.04.12

"""
from abc import ABC, abstractmethod

from spectrumlab.emulation.noise import Noise
from spectrumlab.emulation.spectrum import Spectrum
from spectrumlab.types import Array, Number, Percent


class AbstractEmulation(ABC):
    """Abstract type to any spectrum emulations."""

    @property
    @abstractmethod
    def noise(self) -> Noise:
        raise NotImplementedError

    @property
    @abstractmethod
    def number(self) -> Array[Number]:
        raise NotImplementedError

    @property
    @abstractmethod
    def intensity(self) -> Array[Percent]:
        raise NotImplementedError

    # --------        handlers        --------
    @abstractmethod
    def setup(self, number: Array[Number], position: Number, concentration: float) -> 'AbstractEmulation':
        """Setup emulation of spectrum."""
        raise NotImplementedError

    @abstractmethod
    def run(self, show: bool = False, random_state: int | None = None) -> Spectrum:
        """Run emulation."""
        raise NotImplementedError
