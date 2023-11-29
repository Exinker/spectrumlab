
from abc import ABC, abstractmethod

from spectrumlab.alias import Array, Number


class BaseProfile(ABC):
    """Base peak's profile type."""

    @abstractmethod
    def __call__(self, x: Array[Number], position: Number, intensity: float, background: float) -> Array[float]:
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError
