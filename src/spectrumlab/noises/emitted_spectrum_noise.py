from dataclasses import dataclass, field
from typing import overload

import numpy as np

from spectrumlab.detectors import Detector
from spectrumlab.noises.base_noise import NoiseABC
from spectrumlab.types import Array, Electron, Percent


@dataclass(frozen=True)
class EmittedSpectrumNoise(NoiseABC):
    """Detector's noise dependence for any emitted spectra."""
    detector: Detector
    n_frames: int
    units: Percent | Electron = field(default=Percent)

    @overload
    def __call__(self, value: Percent) -> Percent: ...
    @overload
    def __call__(self, value: Array[Percent]) -> Array[Percent]: ...
    @overload
    def __call__(self, value: Electron) -> Electron: ...
    @overload
    def __call__(self, value: Array[Electron]) -> Array[Electron]: ...
    def __call__(self, value):
        detector = self.detector
        n_frames = self.n_frames

        if self.units == Percent:
            read_noise = detector.config.read_noise  # [e]
            kc = detector.config.capacity / 100

            return (1/kc) * np.sqrt(
                read_noise**2 + value*kc  # noqa: C815
            ) / np.sqrt(n_frames)

        if self.units == Electron:
            read_noise = detector.config.read_noise

            return np.sqrt(
                read_noise**2 + value  # noqa: C815
            ) / np.sqrt(n_frames)

        raise TypeError(f'{self.units} units is not supported!')
