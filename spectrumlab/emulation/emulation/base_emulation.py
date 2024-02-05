"""
Emitted and absorbed spectrum emulation.

    Author: Vaschenko Pavel
    Email: vaschenko@vmk.ru
    Date: 2013.04.12

"""
from abc import ABC, abstractmethod

from spectrumlab.alias import Array, Percent, Number
from spectrumlab.emulation.noise import Noise
from spectrumlab.emulation.spectrum import Spectrum


class EmulationInterface(ABC):
    """Interface to emulate spectrum."""

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
    def setup(self, number: Array[Number], position: Number, concentration: float) -> 'EmulationInterface':
        """Setup emulation of spectrum."""
        raise NotImplementedError

    @abstractmethod
    def run(self, show: bool = False, random_state: int | None = None) -> Spectrum:
        """Run emulation."""
        raise NotImplementedError
