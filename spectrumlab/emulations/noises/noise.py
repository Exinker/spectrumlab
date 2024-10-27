import abc

from spectrumlab.types import Array, Electron, Percent


class AbstractNoise(abc.ABC):
    """Abstract noise dependence type."""

    @abc.abstractmethod
    def __call__(self, value: Percent | Array[Percent] | Electron | Array[Electron]) -> Percent | Array[Percent] | Electron | Array[Electron]:
        pass
