from abc import ABC, abstractmethod

from spectrumlab.typing import Array, Electron, Percent


class AbstractNoise(ABC):
    """Abstract noise dependence type."""

    @abstractmethod
    def __call__(self, value: Percent | Array[Percent] | Electron | Array[Electron]) -> Percent | Array[Percent] | Electron | Array[Electron]:
        pass
