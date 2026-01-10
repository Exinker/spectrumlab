
from dataclasses import dataclass, field
from typing import overload

from spectrumlab.detectors import Detector
from spectrumlab.noises.base_noise import NoiseABC
from spectrumlab.types import Array, Percent


@dataclass(frozen=True)
class MixedSpectrumNoise(NoiseABC):
    """Detector's noise dependence for any microwave or ICP spectra."""
    detector: Detector
    n_frames: int = field(default=1)
    units: Percent = field(default=Percent)

    @overload
    def __call__(self, value: Percent) -> Percent: ...
    @overload
    def __call__(self, value: Array[Percent]) -> Array[Percent]: ...
    def __call__(self, value):
        raise NotImplementedError
