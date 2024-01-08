from typing import TypeAlias

from .grid import Grid
from .voight_peak_shape import VoightPeakShape, SelfReversedVoightPeakShape, approx_grid, restore_shape_from_grid, restore_shape_from_spectrum


PeakShape: TypeAlias = VoightPeakShape | SelfReversedVoightPeakShape
