from typing import TypeAlias

from .amplitude_intensity_estimator import AmplitudeIntensityEstimator
from .approx_intensity_estimator import ApproxIntensityEstimator
from .integral_intensity_estimator import IntegralIntensityEstimator


IntensityEstimator : TypeAlias = AmplitudeIntensityEstimator | ApproxIntensityEstimator | IntegralIntensityEstimator

__all__ = [
    AmplitudeIntensityEstimator,
    ApproxIntensityEstimator,
    IntegralIntensityEstimator,
    IntensityEstimator,
]
