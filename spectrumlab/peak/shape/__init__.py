from typing import TypeAlias

from .voigt_peak_shape import VoigtPeakShape, SelfReversedVoigtPeakShape, approx_grid, restore_shape_from_grid, restore_shape_from_spectrum


PeakShape: TypeAlias = VoigtPeakShape | SelfReversedVoigtPeakShape
