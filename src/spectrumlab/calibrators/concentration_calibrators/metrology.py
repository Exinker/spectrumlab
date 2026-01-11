from typing import Mapping

import numpy as np
from scipy import interpolate

from spectrumlab.types import Frame, Intercept, Slope


class LimitABC:

    def __init__(
        self,
        intensity: float,
        coeff: tuple[Intercept, Slope],
        info: str,
    ) -> None:

        self._intensity = intensity
        self._coeff = coeff
        self._info = info

    @property
    def intensity(self) -> float:
        return self._intensity

    @property
    def concentration(self) -> float:
        intercept, slope = self._coeff

        return 10**((np.log10(self.intensity) - intercept) / slope)

    def __repr__(self) -> str:
        cls = self.__class__
        return f'{cls.__name__}({self.concentration})'

    def __str__(self) -> str:
        cls = self.__class__

        return '\n'.join([
            f'{cls.__name__}({self._info})',
            f'\tI: {self.intensity:.4f}$',
            f'\tC: {self.concentration:.4f}$',
        ])


class LOD(LimitABC):
    """Limit of Detection (LOD) in emission or absorption."""

    k_default = 3

    def __init__(
        self,
        intensity: float,
        coeff: tuple[Intercept, Slope],
        info: str = '',
    ) -> None:
        super().__init__(intensity, coeff=coeff, info=info)

    @staticmethod
    def calculate(mean: float, deviation: float, k: float) -> float:
        return mean + k*deviation

    @classmethod
    def from_json(cls, data: Mapping[str, float], coeff: tuple[Intercept, Slope], k: float | None = None) -> 'LOD':
        k = k or cls.k_default

        return cls(
            intensity=cls.calculate(data['mean'], data['deviation'], k=k),
            coeff=coeff,
            info=f'k: {k}',
        )

    @classmethod
    def from_blank(cls, data: Frame, coeff: tuple[Intercept, Slope], k: float | None = None) -> 'LOD':
        k = k or cls.k_default
        mean = data['intensity'].mean()
        deviation = data['intensity'].std(ddof=1)

        return cls(
            intensity=cls.calculate(mean, deviation, k=k),
            coeff=coeff,
            info=f'k: {k}',
        )


class LOQ(LimitABC):
    """Limit of Quantity (LOQ) in emission or absorption."""

    k_default = 10

    def __init__(
        self,
        intensity: float,
        coeff: tuple[Intercept, Slope],
        info: str = '',
    ) -> None:
        super().__init__(intensity, coeff=coeff, info=info)

    @staticmethod
    def calculate(
        mean: float,
        deviation: float,
        k: float,
    ) -> float:
        return mean + k*deviation

    @classmethod
    def from_json(
        cls,
        data: Mapping[str, float],
        coeff: tuple[Intercept, Slope],
        k: float | None = None,
    ) -> 'LOQ':
        k = k or cls.k_default

        return cls(
            intensity=cls.calculate(data['mean'], data['deviation'], k=k),
            coeff=coeff,
            info=f'k: {k}',
        )

    @classmethod
    def from_blank(
        cls,
        data: Frame,
        coeff: tuple[Intercept, Slope],
        k: float | None = None,
    ) -> 'LOQ':
        k = k or cls.k_default
        mean = data['intensity'].mean()
        deviation = data['intensity'].std(ddof=1)

        return cls(
            intensity=cls.calculate(mean, deviation, k=k),
            coeff=coeff,
            info=f'k: {k}',
        )


class LOL(LimitABC):
    """Limit of Linearity (LOL) in emission or absorption."""

    def __init__(
        self,
        intensity: float,
        coeff: tuple[Intercept, Slope],
        info: str = '',
    ) -> None:
        super().__init__(intensity, coeff=coeff, info=info)


def estimate_lol(
    data: Frame,
    coeff: tuple[float, float],
    threshold: float = 0.05,
) -> 'LOL':
    intercept, slope = coeff

    # concentration calibration
    x_grid = data['concentration'].apply(lambda x: np.log10(x))
    y_grid = data['intensity'].apply(lambda x: np.log10(x))

    x = np.linspace(np.min(x_grid), np.max(x_grid), 1_000_000)
    y = interpolate.interp1d(
        x_grid, y_grid,
        kind='linear',
        bounds_error=False,
        fill_value=np.nan,
    )(x)

    ref = 10**(slope*x + intercept)
    predicted = 10**(y)

    mask = (np.abs(ref - predicted) / ref) <= threshold
    value = 10**(np.max(y[mask])) if any(mask) else np.nan

    #
    return LOL(
        intensity=value,
        coeff=coeff,
    )


class DynamicRange:

    def __init__(
        self,
        intensity: tuple[float, float],
        coeff: tuple[Intercept, Slope],
        info: str = '',
    ) -> None:
        self._intensity = intensity
        self._coeff = coeff
        self._info = info

    @property
    def intensity(self) -> tuple[float, float]:
        return self._intensity

    @property
    def concentration(self) -> tuple[float, float]:
        intercept, slope = self._coeff

        return tuple([
            10**((np.log10(value) - intercept) / slope)
            for value in self.intensity
        ])

    def __repr__(self) -> str:
        cls = self.__class__

        low, high = self.concentration
        return f'{cls.__name__}({low:.4f} - {high:.4f})'

    def __str__(self) -> str:
        cls = self.__class__

        def _get_content(values: tuple[float, float]) -> str:
            low, high = values

            return '{low:.4f} - {high:.4f} ({span:.4f})'.format(
                low=low,
                high=high,
                span=np.log10(high) - np.log10(low),
            )

        return '\n'.join([
            f'{cls.__name__}({self._info})',
            f'\tI: {_get_content(values=self.intensity)}',
            f'\tC: {_get_content(values=self.concentration)}',
        ])
