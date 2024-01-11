from abc import ABC, abstractmethod

from spectrumlab.alias import Array, Electron, Percent


class BaseNoise(ABC):
    """Base noise dependence type."""

    @abstractmethod
    def __call__(self, value: Percent | Array[Percent] | Electron | Array[Electron]) -> Percent | Array[Percent] | Electron | Array[Electron]:
        pass
