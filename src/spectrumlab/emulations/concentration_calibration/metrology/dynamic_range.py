import numpy as np

from spectrumlab.concentration_calibration import (
    DynamicRange,
    Intercept,
    # LOD,
    # LOL,
    # LOQ,
    Slope,
)
from spectrumlab.emulations.emulators import (
    AbsorbedSpectrumEmulator,
    EmittedSpectrumEmulator,
    Emulation,
)


def estimate_dynamic_range(
    emulation: Emulation,
    coeff: tuple[Intercept, Slope],
    lb: float,
    ub: float,
    k: float = 3,
) -> DynamicRange:

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

        return DynamicRange(
            intensity=(lb, ub),
            coeff=coeff,
        )

    if isinstance(emulation, AbsorbedSpectrumEmulator):
        return DynamicRange(
            intensity=(lb, ub if lb < ub else np.nan),
            coeff=coeff,
        )

    raise TypeError(f'Emulation type: {type(emulation)} is not supported yet!')
