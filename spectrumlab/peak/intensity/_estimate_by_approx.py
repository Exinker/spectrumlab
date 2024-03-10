from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from spectrumlab.peak.shape.utils import approx_peak
from spectrumlab.picture import COLOR_INTENSITY

if TYPE_CHECKING:
    from spectrumlab.peak.analyte_peak import AnalytePeak
    from spectrumlab.peak.shape import PeakShape


@dataclass
class ApproxIntensityConfig:
    approx_shape: 'PeakShape'
    approx_params: dict = field(default_factory=dict)
    delta: float = field(default=1)  # span of peak's position
    by_tail: bool = field(default=False)  # use the tail of peak for approximation

    @property
    def color(self) -> str:
        return COLOR_INTENSITY['shape']


def estimate_intensity_by_approx(peak: 'AnalytePeak', config: ApproxIntensityConfig, verbose: bool = False, show: bool = False) -> float:
    """Estimate analyte peak's intensity by approximation."""

    # intensity
    config.approx_params = approx_peak(
        peak=peak,
        shape=config.approx_shape,
        delta=config.delta,
        by_tail=config.by_tail,
        show=show,
    )

    value = config.approx_params['intensity']

    # verbose
    if verbose:
        print(f'Peak\'s intensity: {value}')

    #
    return value
