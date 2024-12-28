from typing import Literal, TypeAlias

import numpy as np

from spectrumlab.concentration_calibration import DynamicRange, Intercept, LOL, LOQ, Slope
from spectrumlab.emulations.emulators import AbsorbedSpectrumEmulator, EmittedSpectrumEmulator, Emulation
from spectrumlab.emulations.intensity import calculate_intensity
from spectrumlab.peak.intensity import AbstractIntensityCalculator


# --------        deviation        --------
EstimateDeviationKind: TypeAlias = Literal['theoretical', 'emulational']


def estimate_blank_mean(
    emulation: Emulation,
    calculator: AbstractIntensityCalculator,
    n_parallels: int = 20,
    kind: EstimateDeviationKind = 'theoretical',
) -> float:
    n_numbers = emulation.config.spectrum.n_numbers
    background_level = emulation.config.background_level
    emulation = emulation.setup(position=n_numbers//2, concentration=0)

    match kind:
        case 'theoretical':
            return 0

        case 'emulational':
            intensity = []
            for _ in range(n_parallels):
                value = calculate_intensity(
                    spectrum=emulation.run(),
                    background=background_level,
                    position=n_numbers//2,
                    calculator=calculator,
                )
                intensity.append(value)
            return np.mean(intensity).item()

    raise TypeError(f'LimitsKind: {kind} is not supported yet!')


def estimate_blank_deviation(
    emulation: Emulation,
    calculator: AbstractIntensityCalculator,
    n_parallels: int = 20,
    kind: EstimateDeviationKind = 'theoretical',
) -> float:
    """Calculate intensity deviation."""
    n_numbers = emulation.config.spectrum.n_numbers
    background_level = emulation.config.background_level
    emulation = emulation.setup(position=n_numbers//2, concentration=0)

    match kind:
        case 'theoretical':
            return np.sqrt(calculator.interval) * emulation.noise(background_level)

        case 'emulational':
            intensity = []
            for _ in range(n_parallels):
                value = calculate_intensity(
                    spectrum=emulation.run(),
                    background=background_level,
                    position=n_numbers//2,
                    calculator=calculator,
                )
                intensity.append(value)
            return np.std(intensity, ddof=1).item()

    raise TypeError(f'LimitsKind: {kind} is not supported yet!')


# --------        dynamic range        --------
def estimate_dynamic_range(emulation: Emulation, coeff: tuple[Intercept, Slope], loq: LOQ, lol: LOL, k: float = 3) -> DynamicRange:

    if isinstance(emulation, EmittedSpectrumEmulator):
        # n_numbers = emulation.config.spectrum.n_numbers
        # config = emulation.config
        # emulation = emulation.setup(position=n_numbers//2, concentration=1)
        # B = config.background_level
        # lb = k * (emulation.noise(B) / np.max(emulation.intensity))
        # ub = 100 / (B + np.max(emulation.intensity))

        # try:
        #     if B > 0:
        #         message = '\n{red}\tошибка в расчете диапазона концентраций, если есть спектральный фон!{black}\n'.format(
        #             red='\033[91m',
        #             black='\x1b[0m',
        #         )
        #         raise ValueError(message)
        # except ValueError as error:
        #     print(error)

        lb = loq.intensity
        ub = lol.intensity

        return DynamicRange(
            intensity=(lb, ub),
            coeff=coeff,
        )

    if isinstance(emulation, AbsorbedSpectrumEmulator):
        lb = loq.intensity
        ub = lol.intensity

        return DynamicRange(
            intensity=(lb, ub if lb < ub else np.nan),
            coeff=coeff,
        )

    raise TypeError(f'Emulation type: {type(emulation)} is not supported yet!')
