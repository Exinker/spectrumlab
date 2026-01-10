from abc import ABC, abstractmethod

from spectrumlab.types import Array, Electron, Percent


class NoiseABC(ABC):

    @abstractmethod
    def __call__(
        self,
        value: Percent | Array[Percent] | Electron | Array[Electron],
    ) -> Percent | Array[Percent] | Electron | Array[Electron]:
        pass
