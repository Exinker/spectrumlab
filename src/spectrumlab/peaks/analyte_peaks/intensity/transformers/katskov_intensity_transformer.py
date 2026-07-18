from dataclasses import dataclass

from spectrumlab.peaks.analyte_peaks.intensity.transformers.base_intensity_transformer import (
    IntensityTransformerABC,
)
from spectrumlab.types import R


@dataclass
class KatskovIntensityTransformer(IntensityTransformerABC):
    """Katskov linearization by coefficients.

    Аn introduction to multi-element atomic-absorption analysis
    Katskov D.
    DOI: 10.15826/analitika.2018.22.4.001
    """

    def __init__(self, c1: R, c2: R) -> None:

        self.c1 = c1
        self.c2 = c2

    def __call__(self, __value: R) -> R:

        if __value > self.c1:
            __value = ((__value + self.c1)**2)/(4*self.c1)
        if __value > self.c2:
            __value = ((__value + self.c2)**2)/(4*self.c2)

        return __value
