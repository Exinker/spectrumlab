"""
Detector noise for any emulation.

Author: Vaschenko Pavel
 Email: vaschenko@vmk.ru
  Date: 2021.11.06
"""
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TypeAlias, overload

import numpy as np
import matplotlib.pyplot as plt

from spectrumlab.alias import Array, Electron, Percent, Absorbance
from spectrumlab.emulation.detector.linear_array_detector import Detector


# --------        noise        --------
@dataclass
class NoiseConfig:
    detector: Detector = field(default=Detector.BLPP2000)
    n_frames: int = field(default=1)


class BaseNoise(ABC):
    """Base noise dependence type."""

    @abstractmethod
    def __call__(self, value: Percent | Array[Percent]) -> Percent | Array[Percent]:
        pass


@dataclass(frozen=True)
class ConstantNoise(BaseNoise):
    """Constant noise dependence."""
    noise_level: int = field(default=1)

    @overload
    def __call__(self, value: Percent) -> Percent: ...
    @overload
    def __call__(self, value: Array[Percent]) -> Array[Percent]: ...
    def __call__(self, value):

        if isinstance(value, (int, float)):
            return self.noise_level
        
        if isinstance(value, Sequence):
            return np.full(value.shape, self.noise_level)


# --------        emitted spectrum        --------
@dataclass(frozen=True)
class EmittedSpectrumNoise(BaseNoise):
    """Detector's noise dependence for any emitted spectra."""
    detector: Detector
    n_frames: int
    units: Percent | Electron = field(default=Percent)

    @overload
    def __call__(self, value: Percent) -> Percent: ...
    @overload
    def __call__(self, value: Array[Percent]) -> Array[Percent]: ...
    def __call__(self, value):
        detector = self.detector
        n_frames = self.n_frames

        #
        if self.units == Percent:
            read_noise = detector.config.read_noise  # [e]
            kc = detector.config.capacity / 100

            return (1/kc**2) * np.sqrt(
                read_noise**2 + value*kc
            ) / np.sqrt(n_frames)

        if self.units == Electron:
            read_noise = detector.config.read_noise

            return np.sqrt(
                read_noise**2 + value
            ) / np.sqrt(n_frames)

        raise TypeError(f'{self.units} units is not supported!')


# --------        absorbed spectrum        --------
@overload
def calculate_squared_relative_standard_deviation(value: Percent, noise: EmittedSpectrumNoise) -> float: ...
@overload
def calculate_squared_relative_standard_deviation(value: Array[Percent], noise: EmittedSpectrumNoise) -> Array[float]: ...
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


@dataclass(frozen=True)
class AbsorbedSpectrumNoise(BaseNoise):
    """Detector's noise dependence for any absorbtion spectra."""

    detector: Detector
    n_frames: int
    base_level: float | Array
    base_noise: EmittedSpectrumNoise
    units: Absorbance = field(default=Absorbance)

    @overload
    def __call__(self, value: Absorbance) -> Absorbance: ...
    @overload
    def __call__(self, value: Array[Absorbance]) -> Array[Absorbance]: ...
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
            )
        )
        return calculate_absorbance_deviation(part_base=part_base, part_recorded=part_recorded)


# --------        mixed spectrum        --------
@dataclass(frozen=True)
class MixedSpectrumNoise(BaseNoise):
    """Detector's noise dependence for any microwave or ICP spectra."""
    detector: Detector
    n_frames: int = field(default=1)
    units: Percent | Electron = field(default=Percent)

    @overload
    def __call__(self, value: Percent) -> Percent: ...
    @overload
    def __call__(self, value: Array[Percent]) -> Array[Percent]: ...
    def __call__(self, value):
        raise NotImplementedError


# --------        _        --------
Noise: TypeAlias = ConstantNoise | EmittedSpectrumNoise | AbsorbedSpectrumNoise | MixedSpectrumNoise


if __name__ == '__main__':

    fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), tight_layout=True)

    # emitted spectrum
    n_frames = 1
    level = 10**(np.logspace(-3, np.log10(2), 1000))

    plt.sca(ax_left)
    for detector in Detector:
        noise = EmittedSpectrumNoise(
            detector=detector,
            n_frames=n_frames,
            units=Percent,
        )

        x, y = level, noise(level)
        plt.plot(
            x, y,
            linestyle='-',
            label=detector.config.name,
        )

    plt.xlabel('$I, \%$')
    plt.ylabel('$\sigma_{I}, \%$')

    plt.grid(
        color='grey', linestyle=':',
    )
    plt.legend()

    # absorbed spectrum
    base_level = 100
    level = 10**(np.logspace(-3, np.log10(2), 1000))

    plt.sca(ax_right)
    for detector in Detector:
        absorbance = np.log10(base_level / level)
        noise = AbsorbedSpectrumNoise(
            detector=detector,
            n_frames=1,
            base_level=base_level,
            base_noise=EmittedSpectrumNoise(
                detector=detector,
                n_frames=2000,
            ),
        )

        x, y = absorbance, noise(absorbance)
        plt.plot(
            x, y,
            linestyle='-',
            label=detector.config.name,
        )

    plt.xlabel('$A$')
    plt.ylabel('$\sigma_{A}$')

    plt.grid(
        color='grey', linestyle=':',
    )
    plt.legend()

    #
    plt.show()
