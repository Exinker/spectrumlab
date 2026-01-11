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

    coeff: tuple[float, float]

    def __call__(self, __value: R) -> R:
        c1, c2 = self.coeff

        if __value > c1:
            __value = ((__value + c1)**2)/(4*c1)
        if __value > c2:
            __value = ((__value + c2)**2)/(4*c2)

        return __value
