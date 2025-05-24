"""
Detector noise for any emulation.

Author: Vaschenko Pavel
 Email: vaschenko@vmk.ru
  Date: 2021.11.06
"""
from typing import TypeAlias

from .absorbed_spectrum_noise import (
    AbsorbedSpectrumNoise,
    calculate_absorbance_deviation,
    calculate_squared_relative_standard_deviation,
)
from .constant_noise import ConstantNoise
from .emitted_spectrum_noise import EmittedSpectrumNoise
from .mixed_spectrum_noise import MixedSpectrumNoise


Noise: TypeAlias = ConstantNoise | EmittedSpectrumNoise | AbsorbedSpectrumNoise | MixedSpectrumNoise
