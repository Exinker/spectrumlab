from abc import ABC, abstractmethod

from spectrumlab.peaks.units import R
from spectrumlab.types import Array, Number


class AbstractPeakShape(ABC):
    """Abstract peak's shape."""

    @abstractmethod
    def __call__(self, x: Array[Number], position: Number, intensity: float, background: float) -> Array[R]:
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError
