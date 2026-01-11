from dataclasses import dataclass, field
from typing import overload

import numpy as np

from spectrumlab.noises.base_noise import NoiseABC
from spectrumlab.types import Array, Electron, Percent


@dataclass(frozen=True)
class ConstantNoise(NoiseABC):
    """Constant noise dependence."""
    noise_level: int = field(default=1)

    @overload
    def __call__(self, value: Percent) -> Percent: ...
    @overload
    def __call__(self, value: Array[Percent]) -> Array[Percent]: ...
    @overload
    def __call__(self, value: Electron) -> Electron: ...
    @overload
    def __call__(self, value: Array[Electron]) -> Array[Electron]: ...
    def __call__(self, value):
        if isinstance(value, (int, float)):
            return self.noise_level
        if isinstance(value, np.ndarray):
            return np.full(value.shape, self.noise_level)
