from abc import ABC, abstractmethod
from dataclasses import dataclass


class AbstractIntensityTransformer(ABC):

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


@dataclass
class IntensityNormalization(AbstractIntensityTransformer):
    """Normalization by value."""

    coeff: float

    def __call__(self, value: float) -> float:
        return value / self.coeff


@dataclass
class KatskovIntensityTransformer(AbstractIntensityTransformer):
    """Katskov linearization by coefficients.

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


def transform_intensity(
    __value: float,
    transformer: AbstractIntensityTransformer,
) -> float:

    if transformer is None:
        return __value
    if isinstance(transformer, (
        KatskovIntensityTransformer,
        IntensityNormalization,
    )):
        return transformer(__value)

    raise ValueError(f'Correction method {transformer.__name__} is not supported!')
