
from dataclasses import dataclass

import numpy as np
from scipy import interpolate

from ...alias import Frame
from ..emulation import Emulation, EmittedSpectrumEmulation, AbsorbedSpectrumEmulation
from .intensity_limits import IntensityLOD


# --------        limit of detection (LoD)        --------
def calculate_concentration_LOD(intensity_lod: IntensityLOD, coeff: tuple[float, float]) -> float:
    intercept, slope = coeff

    return 10**((np.log10(intensity_lod.value) - intercept) / slope)


# --------        limit of linearity (LoD)        --------
def calculate_concentration_LOL(unicorn: Frame, coeff: tuple[float, float], threshold) -> float:

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

    return limit


# --------        dynamic range        --------
@dataclass(frozen=True)
class DynamicRange:
    emulation: Emulation
    lod: float  # limit of detection
    lol: float  # limit of linearity

    @property
    def value(self) -> float:
        return np.log10(self.lol) - np.log10(self.loq)

    @property
    def loq(self) -> float:
        return 10 * self.lod / 3

    def __repr__(self):
        emulation = self.emulation
        config = emulation.config
        detector = emulation.detector

        return '\n'.join([
            f'Dynamic range ({detector.name}, n_frames: {config.spectrum.n_frames})',
            f'\trange: {self.lod:.4f} - {self.lol:.4f} ({self.value:.4f})',
        ])

    def __iter__(self):
        for key in ['lod', 'lol']:
            yield getattr(self, key)

def calculate_dynamic_range(emulation: Emulation, unicorn: Frame, coeff: tuple[float, float], intensity_lod: IntensityLOD, threshold: float = 0.05) -> DynamicRange:
    n_numbers = emulation.config.spectrum.n_numbers
    config = emulation.config

    #
    if isinstance(emulation, EmittedSpectrumEmulation):

        # setup emulation
        emulation = emulation.setup(position=n_numbers//2, concentration=1)

        #
        B = config.background_level

        try:
            lod = (3 * emulation.noise(B)) / np.max(emulation.intensity)
            lol = 100 / (B + np.max(emulation.intensity))

            if B > 0:  # FIXME:
                message = '\n{red}\tошибка в расчете диапазона концентраций, если есть спектральный фон!{black}\n'.format(
                    red='\033[91m',
                    black='\x1b[0m',
                )
                raise ValueError(message)

        except ValueError as error:
            print(error)

        # return domain
        return DynamicRange(
            emulation=emulation,
            lod=lod,
            lol=lol,
        )

    if isinstance(emulation, AbsorbedSpectrumEmulation):

        # lod
        lod = calculate_concentration_LOD(intensity_lod, coeff=coeff)

        # lol
        lol = calculate_concentration_LOL(
            unicorn=unicorn,
            coeff=coeff,
            threshold=threshold,
        )
        lol = lol if lod < lol else np.nan

        # return domain
        return DynamicRange(
            emulation=emulation,
            lod=lod,
            lol=lol,
        )

    raise TypeError
