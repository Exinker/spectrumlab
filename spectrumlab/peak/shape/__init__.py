from typing import TypeAlias

from .grid import Grid
from .voight_peak_shape import VoightPeakShape, SelfReversedVoightPeakShape, approx_grid


PeakShape: TypeAlias = VoightPeakShape | SelfReversedVoightPeakShape
