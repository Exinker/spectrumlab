
from typing import TypeAlias, TYPE_CHECKING

from ._estimate_by_amplitude import AmplitudeIntensityConfig, estimate_intensity_by_amplitude
from ._estimate_by_integral import IntegralIntensityConfig, estimate_intensity_by_integral
from ._estimate_by_approx import ApproxIntensityConfig, estimate_intensity_by_approx

if TYPE_CHECKING:
    from spectrumlab.peak.analyte_peak import AnalytePeak


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

    raise ValueError(f'config: {config} is not supported!')
