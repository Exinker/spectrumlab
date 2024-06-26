"""
Emitted and absorbed spectrum emulation.

    Author: Vaschenko Pavel
    Email: vaschenko@vmk.ru
    Date: 2013.04.12

"""
import abc

from spectrumlab.emulation.noise import Noise
from spectrumlab.emulation.spectrum import Spectrum
from spectrumlab.types import Array, Number, Percent


class AbstractEmulation(abc.ABC):
    """Abstract type to any spectrum emulations."""

    @property
    @abc.abstractmethod
    def noise(self) -> Noise:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def number(self) -> Array[Number]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def intensity(self) -> Array[Percent]:
        raise NotImplementedError

    # --------        handlers        --------
    @abc.abstractmethod
    def setup(self, number: Array[Number], position: Number, concentration: float) -> 'AbstractEmulation':
        """Setup emulation of spectrum."""
        raise NotImplementedError

    @abc.abstractmethod
    def run(self, show: bool = False, random_state: int | None = None) -> Spectrum:
        """Run emulation."""
        raise NotImplementedError
