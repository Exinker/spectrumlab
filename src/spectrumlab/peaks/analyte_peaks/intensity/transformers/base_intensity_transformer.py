from abc import ABC, abstractmethod

from spectrumlab.types import R


class IntensityTransformerABC(ABC):

    @abstractmethod
    def __call__(self, __value: R, *args, **kwargs) -> R:
        raise NotImplementedError
