import abc

from spectrumlab.peak.units import U
from spectrumlab.types import Array, Number


class AbstractPeakShape(abc.ABC):
    """Abstract peak's shape."""

    @abc.abstractmethod
    def __call__(self, x: Array[Number], position: Number, intensity: float, background: float) -> Array[U]:
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError
