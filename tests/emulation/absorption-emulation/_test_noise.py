
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import pytest

# from spectrumlab.emulation.c
from spectrumlab.emulation.emulation import emulate_emitted_spectrum, emulate_absorbed_spectrum, EmulationConfig, EmittedSpectrumEmulationConfig, AbsorbedSpectrumEmulationConfig
from spectrumlab.emulation.noise import EmittedSpectrumNoise, AbsorbedSpectrumNoise

N_TIMES = 5_000
N_NUMBERS = 100


@pytest.mark.parametrize()
def test_absorbed_noise_vs_value(config: AbsorbedSpectrumEmulationConfig, ylim: Optional[tuple[float, float]] = None) -> None:

    # config = ``

    number = np.arange(N_NUMBERS)
    values = np.logspace(-3, np.log10(2), N_NUMBERS)

    # theoretical noise
    theoretical_noise = AbsorbedSpectrumNoise(
        detector=config.detector,
        n_frames=config.spectrum.n_frames,
        base_level=config.spectrum_base.level,
        base_noise=EmittedSpectrumNoise(
            detector=config.detector,
            n_frames=config.spectrum_base.n_frames,
        ),
    )(values)

    # emulated noise
    spectrum = emulate_absorbed_spectrum(
        intensity=np.array([config.spectrum_base.level * 10**(-values)]*N_TIMES),
        number=number,
        noise=EmittedSpectrumNoise(
            detector=config.detector,
            n_frames=config.spectrum.n_frames,
        ),
        base_level=config.spectrum_base.level,
        base_noise=EmittedSpectrumNoise(
            detector=config.detector,
            n_frames=config.spectrum_base.n_frames,
        ),
        detector=config.detector,
    )

    emulated_noise = np.std(spectrum.intensity, ddof=1, axis=0)

    #
    print()

