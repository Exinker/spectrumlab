from abc import ABC, abstractmethod

from spectrumlab.typing import Array, Number


class BasePeakShape(ABC):
    """Base peak's shape."""

    @abstractmethod
    def __call__(self, x: Array[Number], position: Number, intensity: float, background: float) -> Array[float]:
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError
