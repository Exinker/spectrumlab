
from dataclasses import dataclass

import numpy as np
from scipy import interpolate

from spectrumlab.alias import Frame
from spectrumlab.emulation import Emulation, EmittedSpectrumEmulation, AbsorbedSpectrumEmulation
from spectrumlab.emulation.intensity import calculate_intensity

from dataclasses import dataclass
from string import Template
from typing import Literal, TypeAlias

import numpy as np
import matplotlib.pyplot as plt

from spectrumlab.alias import Array, Number
from spectrumlab.calibration_curve import Intercept, Slope, LOD, LOQ, LOL, DynamicRange
from spectrumlab.emulation import Emulation, EmittedSpectrumEmulation, AbsorbedSpectrumEmulation

from spectrumlab.emulation.spectrum import Spectrum, EmittedSpectrum, AbsorbedSpectrum
from spectrumlab.peak.intensity import IntensityConfig, AmplitudeIntensityConfig, IntegralIntensityConfig, ApproxIntensityConfig
from spectrumlab.peak.intensity import InterpolationKind, integrate_grid, interpolate_grid


# --------        limits (LOD and LOQ)        --------
EstimationKind: TypeAlias = Literal['theoretical', 'emulational']


def estimate_deviation(emulation: Emulation, config: IntegralIntensityConfig, n_parallels: int = 10, kind: EstimationKind = 'theoretical') -> float:
    """Calculate intensity deviation."""
    n_numbers = emulation.config.spectrum.n_numbers
    background_level = emulation.config.background_level
    emulation = emulation.setup(position=n_numbers//2, concentration=0)

    match kind:
        case 'theoretical':
            return np.sqrt(config.interval) * emulation.noise(background_level)

        case 'emulational':
            intensity = []
            for _ in range(n_parallels):
                value = calculate_intensity(
                    spectrum=emulation.run(),
                    background=background_level,
                    position=n_numbers//2,
                    config=config,
                )
                intensity.append(value)
            return np.std(intensity, ddof=1).item()

    raise TypeError(f'LimitsKind: {kind} is not supported yet!')


def estimate_dynamic_range(emulation: Emulation, coeff: tuple[Intercept, Slope], loq: LOQ, lol: LOL, k: float = 3) -> DynamicRange:
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
            intensity=(lb, ub),
            coeff=coeff,
        )

    if isinstance(emulation, AbsorbedSpectrumEmulation):
        lb = loq.concentration
        ub = lol.concentration

        return DynamicRange(
            intensity=(lb, ub if lb < ub else np.nan),
            coeff=coeff,
        )

    raise TypeError(f'Emulation type: {type(emulation)} is not supported yet!')
