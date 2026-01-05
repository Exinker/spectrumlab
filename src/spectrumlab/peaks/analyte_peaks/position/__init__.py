from typing import TypeAlias

from .interpolation_position_estimator import InterpolationPositionEstimator
from .parabola_position_estimator import ParabolaPositionEstimator, CORR_COEFF


PositionEstimator : TypeAlias = InterpolationPositionEstimator | ParabolaPositionEstimator


__all__ = [
    PositionEstimator,
    InterpolationPositionEstimator,
    ParabolaPositionEstimator,
    CORR_COEFF,
]
