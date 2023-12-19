from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias, NewType, TYPE_CHECKING

import numpy as np
from scipy import interpolate

from spectrumlab.alias import Frame
from spectrumlab.emulation import Emulation, EmittedSpectrumEmulation, AbsorbedSpectrumEmulation


Intercept = NewType('Intercept', float)
Slope = NewType('Slope', float)


# --------        limits        --------
class BaseLimit:

    def __init__(self, intensity: float, coeff: tuple[Intercept, Slope], info: str):
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


class LOD(BaseLimit):
    """Limit of Detection (LOD) in emission or absorption."""

    def __init__(self, intensity: float, coeff: tuple[Intercept, Slope], info: str = ''):
        super().__init__(intensity, coeff=coeff, info=info)

    @classmethod
    def from_deviation(cls, deviation: float, coeff: tuple[Intercept, Slope], k: float = 3) -> 'LOD':
        return cls(
            intensity=k*deviation,
            coeff=coeff,
        )


class LOQ(BaseLimit):
    """Limit of Quantity (LOQ) in emission or absorption."""

    def __init__(self, intensity: float, coeff: tuple[Intercept, Slope], info: str = ''):
        super().__init__(intensity, coeff=coeff, info=info)

    @classmethod
    def from_deviation(cls, deviation: float, coeff: tuple[Intercept, Slope], k: float = 10) -> 'LOQ':
        return cls(
            intensity=k*deviation,
            coeff=coeff,
        )


# --------        limit of linearity        --------
class LOL(BaseLimit):
    """Limit of Linearity (LOL) in emission or absorption."""

    def __init__(self, intensity: float, coeff: tuple[Intercept, Slope], info: str = ''):
        super().__init__(intensity, coeff=coeff, info=info)


def estimate_lol(data: Frame, coeff: tuple[float, float], threshold: float = 0.05) -> 'LOL':
    intercept, slope = coeff

    # calibration curve
    x_grid = data.loc[~data['mask'], 'concentration'].apply(lambda x: np.log10(x))
    y_grid = data.loc[~data['mask'], 'intensity'].apply(lambda x: np.log10(x))

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

# --------        dynamic range        --------
class DynamicRange:

    def __init__(self, intensity: tuple[float, float], coeff: tuple[Intercept, Slope], info: str = ''):
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

    # def __iter__(self):
    #     for key in ['lb', 'ub']:
    #         yield getattr(self, key)


def estimate_dynamic_range(emulation: Emulation, unicorn: Frame, coeff: tuple[float, float], loq: LOQ, k: float = 3, threshold: float = 0.05) -> DynamicRange:
    n_numbers = emulation.config.spectrum.n_numbers
    config = emulation.config

    #
    if isinstance(emulation, EmittedSpectrumEmulation):
        emulation = emulation.setup(position=n_numbers//2, concentration=1)
        B = config.background_level
        k = 3
        lb = k * (emulation.noise(B) / np.max(emulation.intensity))
        ub = 100 / (B + np.max(emulation.intensity))

        try:
            if B > 0:
                message = '\n{red}\tошибка в расчете диапазона концентраций, если есть спектральный фон!{black}\n'.format(
                    red='\033[91m',
                    black='\x1b[0m',
                )
                raise ValueError(message)
        except ValueError as error:
            print(error)

        return DynamicRange(
            emulation=emulation,
            lb=lb,
            ub=ub,
        )

    if isinstance(emulation, AbsorbedSpectrumEmulation):
        lb = loq.to_concentration(coeff=coeff)
        ub = calculate_concentration_LOL(
            unicorn=unicorn,
            coeff=coeff,
            threshold=threshold,
        )

        return DynamicRange(
            emulation=emulation,
            lb=lb,
            ub=ub if lb < ub else np.nan,
        )

    raise TypeError(f'Emulation type: {type(emulation)} is not supported yet!')
