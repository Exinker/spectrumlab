from collections.abc import Sequence
from typing import Literal, TypeAlias, NewType, TYPE_CHECKING

import numpy as np

from ._estimate_by_amplitude import AmplitudeIntensityConfig, estimate_intensity_by_amplitude
from ._estimate_by_integral import IntegralIntensityConfig, estimate_intensity_by_integral
from ._estimate_by_approx import ApproxIntensityConfig, estimate_intensity_by_approx

if TYPE_CHECKING:
    from spectrumlab.peak.analyte_peak import AnalytePeak


# --------        intensity        --------
IntensityConfig: TypeAlias = AmplitudeIntensityConfig | IntegralIntensityConfig | ApproxIntensityConfig


def calculate_intensity(peak: 'AnalytePeak', config: IntensityConfig) -> float:
    """Interface to estimate analyte peak's intensity."""

    if isinstance(config, AmplitudeIntensityConfig):
        return estimate_intensity_by_amplitude(
            peak=peak,
            config=config,
        )

    if isinstance(config, IntegralIntensityConfig):
        return estimate_intensity_by_integral(
            peak=peak,
            config=config,
        )

    if isinstance(config, ApproxIntensityConfig):
        return estimate_intensity_by_approx(
            peak=peak,
            config=config,
        )

    raise TypeError(f'config: {config} is not supported!')


# --------        limits (LOD and LOQ)        --------
Units: TypeAlias = Literal['%', 'A']
Intercept = NewType('Intercept', float)
Slope = NewType('Slope', float)


class BaseLimit:

    def __init__(self, intensity: Sequence[float], units: Units, k: float, info: str):
        self.intensity = intensity
        self.units = units
        self.k = k
        self.info = info

    @property
    def value(self) -> float:
        return self.k * np.std(self.intensity, ddof=1)

    def to_concentration(self, coeff: tuple[Intercept, Slope]) -> float:
        """Convert the limit to concentration."""
        intercept, slope = coeff

        return 10**((np.log10(self.value) - intercept) / slope)

    def __repr__(self) -> str:
        return f'{self.value:.4f}'

    def __repr__(self) -> str:
        cls = self.__class__

        return '\n'.join([
            f'{cls.__name__}({self.info})',
            f'\tvalue: {self.value:.4f}$ [{self.units}]',
        ])


class LOD(BaseLimit):
    """Limit of Detection (LOD) in emission or absorption."""

    def __init__(self, intensity: Sequence[float], units: Units, k: float = 3, info: str = ''):
        super().__init__(intensity, units, k=k, info=info)


class LOQ(BaseLimit):
    """Limit of Quantity (LOQ) in emission or absorption."""

    def __init__(self, intensity: Sequence[float], units: Units, k: float = 10, info: str = ''):
        super().__init__(intensity, units, k=k, info=info)

    @classmethod
    def from_lod(cls, lod: LOD, k: float = 10) -> 'LOQ':
        return cls(
            intensity=lod.intensity,
            units=lod.units,
            k=k,
            info=lod.info,
        )
