from abc import ABC, abstractmethod

import numpy as np

from spectrumlab.types import Array, R


class IntensityTransformerABC(ABC):

    def predict(self, value: Array[R]) -> Array[R]:
        value = np.array(value, dtype=np.float64, copy=True)

        return np.array(list(map(self, value)), dtype=np.float64)

    @abstractmethod
    def __call__(self, __value: R) -> R:
        raise NotImplementedError
