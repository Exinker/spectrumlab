
from dataclasses import dataclass

import numpy as np
from scipy import interpolate

from spectrumlab.alias import Frame
from spectrumlab.emulation import Emulation, EmittedSpectrumEmulation, AbsorbedSpectrumEmulation
from spectrumlab.emulation.intensity import LOQ


# --------        limit of linearity (LOL)        --------
def calculate_concentration_LOL(unicorn: Frame, coeff: tuple[float, float], threshold: float) -> float:
    """Calculate Limit of Linearity in concentration."""
    intercept, slope = coeff

    # calibration curve
    x_grid = unicorn['concentration'].apply(lambda x: np.log10(x))
    y_grid = unicorn['intensity'].apply(lambda x: np.log10(x))

    x = np.linspace(np.min(x_grid), np.max(x_grid), 1_000_000)
    y = interpolate.interp1d(
        x_grid, y_grid,
        kind='linear',
        bounds_error=False,
        fill_value=np.nan,
    )(x)

    ref = 10**(slope*x + intercept)
    predicted = 10**(y)

    mask = (100*np.abs(ref - predicted) / ref) <= threshold
    limit = 10**(np.max(x[mask])) if any(mask) else np.nan

    #
    return limit


# --------        dynamic range        --------
@dataclass(frozen=True)
class DynamicRange:
    emulation: Emulation
    lb: float  # limit of quantity (LOQ)
    ub: float  # limit of linearity (LOL)

    @property
    def value(self) -> float:
        return np.log10(self.ub) - np.log10(self.lb)

    def __repr__(self):
        emulation = self.emulation
        config = emulation.config
        detector = emulation.detector

        return '\n'.join([
            f'Dynamic range ({detector.name}, n_frames: {config.spectrum.n_frames})',
            f'\trange: {self.lb:.4f} - {self.ub:.4f} ({self.value:.4f})',
        ])

    def __iter__(self):
        for key in ['lb', 'ub']:
            yield getattr(self, key)


def calculate_dynamic_range(emulation: Emulation, unicorn: Frame, coeff: tuple[float, float], loq: LOQ, k: float = 3, threshold: float = 0.05) -> DynamicRange:
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
