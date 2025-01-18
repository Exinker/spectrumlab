from typing import Literal

import numpy as np

from spectrumlab.emulations.emulators import Emulation
from spectrumlab.emulations.intensity import calculate_intensity
from spectrumlab.peak.intensity import AbstractIntensityCalculator


EstimateDeviationKind = Literal['theoretical', 'emulational']


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
