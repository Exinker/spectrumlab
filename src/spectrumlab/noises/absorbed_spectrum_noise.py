from dataclasses import dataclass, field
from typing import overload

import numpy as np

from spectrumlab.detectors import Detector
from spectrumlab.noises.base_noise import NoiseABC
from spectrumlab.noises.emitted_spectrum_noise import EmittedSpectrumNoise
from spectrumlab.types import Absorbance, Array, Electron, Percent


@dataclass(frozen=True)
class AbsorbedSpectrumNoise(NoiseABC):
    """Detector's noise dependence for any absorbtion spectra."""
    detector: Detector
    n_frames: int
    base_level: float | Array
    base_noise: EmittedSpectrumNoise
    units: Absorbance = field(default=Absorbance)

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
        base_level = self.base_level
        base_noise = self.base_noise

        part_base = calculate_squared_relative_standard_deviation(
            value=base_level,
            noise=base_noise,
        )
        part_recorded = calculate_squared_relative_standard_deviation(
            value=base_level*10**(-value),
            noise=EmittedSpectrumNoise(
                detector=detector,
                n_frames=n_frames,
            ),
        )
        return calculate_absorbance_deviation(part_base=part_base, part_recorded=part_recorded)


@overload
def calculate_squared_relative_standard_deviation(
    value: Percent,
    noise: EmittedSpectrumNoise,
) -> float: ...
@overload
def calculate_squared_relative_standard_deviation(
    value: Array[Percent],
    noise: EmittedSpectrumNoise,
) -> Array[float]: ...
def calculate_squared_relative_standard_deviation(value, noise):
    """Calculate squared relative standard deviation."""

    return (noise(value) / value)**2


@overload
def calculate_absorbance_deviation(part_base: float, part_recorded: float) -> float: ...
@overload
def calculate_absorbance_deviation(part_base: Array[float], part_recorded: Array[float]) -> Array[float]: ...
def calculate_absorbance_deviation(part_base, part_recorded):
    """Calculate absorbance deviation."""

    return (1/np.log(10)) * np.sqrt(part_base + part_recorded)
