
from typing import TypeAlias, TYPE_CHECKING

from spectrumlab.typing import Number
from ._estimate_by_interpolation import InterpolationPositionConfig, estimate_position_by_interpolation
from ._estimate_by_parabola import ParabolaPositionConfig, estimate_position_by_parabola

if TYPE_CHECKING:
    from spectrumlab.peak.analyte_peak import AnalytePeak


PositionConfig: TypeAlias = InterpolationPositionConfig | ParabolaPositionConfig


def calculate_position(peak: 'AnalytePeak', config: PositionConfig) -> Number:
    """Interface to estimate analyte peak's position."""

    if isinstance(config, InterpolationPositionConfig):
        return estimate_position_by_interpolation(
            peak=peak,
            config=config,
        )

    if isinstance(config, ParabolaPositionConfig):
        return estimate_position_by_parabola(
            peak=peak,
            config=config,
        )

    raise ValueError(f'config: {config} is not supported!')
