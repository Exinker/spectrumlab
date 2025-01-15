from abc import ABC, abstractmethod
from dataclasses import dataclass


class AbstractIntensityCorrector(ABC):

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


@dataclass
class IntensityNormalization(AbstractIntensityCorrector):
    """Normalization by value."""

    coeff: float

    def __call__(self, value: float) -> float:
        return value / self.coeff


@dataclass
class KatskovIntensityLinearization(AbstractIntensityCorrector):
    """Katskov linearization.

    Аn introduction to multi-element atomic-absorption analysis
    Katskov D.
    DOI: 10.15826/analitika.2018.22.4.001
    """

    coeff: tuple[float, float]

    def __call__(self, value: float) -> float:
        c1, c2 = self.coeff

        if value > c1:
            value = ((value + c1)**2)/(4*c1)
        if value > c2:
            value = ((value + c2)**2)/(4*c2)

        return value


def _correct_intensity(
    __value: float,
    corrector: AbstractIntensityCorrector,
) -> float:

    if corrector is None:
        return __value
    if isinstance(corrector, (
        KatskovIntensityLinearization,
        IntensityNormalization,
    )):
        return corrector(__value)

    raise ValueError(f'Correction method {corrector.__name__} is not supported!')
