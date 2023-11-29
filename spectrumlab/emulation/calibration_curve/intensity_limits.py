
from dataclasses import dataclass
from string import Template
from typing import Literal, TypeAlias

import numpy as np

from spectrumlab.emulation import Emulation, EmittedSpectrumEmulation, AbsorbedSpectrumEmulation
from spectrumlab.emulation.intensity import IntegralIntensityConfig, calculate_intensity, InterpolationKind


# --------        limit of detection (LoD)        --------
LimitKind: TypeAlias = Literal['theoretical', 'emulational']

@dataclass
class IntensityLOD:
    """Limit of detection (LoD) of an intensity (in intensity or absorbance)."""
    emulation: Emulation
    value: float
    kind: LimitKind

    def __repr__(self) -> str:
        emulation = self.emulation
        detector = emulation.detector

        template = Template(
            '\n'.join([
                f'Intensity limit of detection ({detector.name})',
                f'\t{self.kind} value: {self.value:.4f}$units',
            ])
        )
        if isinstance(emulation, EmittedSpectrumEmulation):
            return template.substitute(units='%')
        if isinstance(emulation, AbsorbedSpectrumEmulation):
            return template.substitute(units='A')

        raise TypeError


def calculate_intensity_LOD(emulation: Emulation, config: IntegralIntensityConfig = IntegralIntensityConfig(), n_parallels: int = 10, kind: LimitKind = 'theoretical') -> IntensityLOD:
    n_numbers = emulation.config.spectrum.n_numbers
    background_level = emulation.config.background_level

    match kind:
        case 'theoretical':

            # deviation
            deviation = np.sqrt(config.interval) * emulation.noise(background_level)

            #
            return IntensityLOD(
                emulation=emulation,
                value=3*deviation,
                kind=kind,
            )

        case 'emulational':

            # deviation
            emulation = emulation.setup(position=n_numbers//2, concentration=0)

            intensity = []
            for _ in range(n_parallels):
                value = calculate_intensity(
                    spectrum=emulation.run(),
                    background=background_level,
                    position=n_numbers//2,
                    config=config,
                )
                intensity.append(value)
            deviation = np.std(intensity, ddof=1).item()

            #
            return IntensityLOD(
                emulation=emulation,
                value=3*deviation,
                kind=kind,
            )

    raise ValueError


# --------        limit of linearity (LoL)        --------

