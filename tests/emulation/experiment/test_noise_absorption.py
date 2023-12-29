import os
import pytest

import numpy as np

from spectrumlab.emulation.emulation import emulate_absorbed_spectrum, AbsorbedSpectrumEmulationConfig, SpectrumBaseConfig, SpectrumConfig
from spectrumlab.emulation.experiment import AbsorbedExperimentConfig as ExperimentConfig
from spectrumlab.emulation.noise import EmittedSpectrumNoise, AbsorbedSpectrumNoise


@pytest.fixture(scope='module')
def config() -> AbsorbedSpectrumEmulationConfig:
    config = ExperimentConfig.from_ini(
        filedir=os.path.join('.', 'tests', 'emulation', 'ini'),
        filename='config_absorption_Ag338.289_I.ini',
    )

    return AbsorbedSpectrumEmulationConfig(
        device=config.device,
        detector=config.detector,

        line_shape=config.line_shape,
        apparatus=config.apparatus,
        aperture=config.aperture,

        spectrum_base=SpectrumBaseConfig(
            level=config.base_level,
            n_frames=config.base_n_frames,
        ),
        spectrum=SpectrumConfig(
            n_numbers=config.n_numbers,
            n_frames=config.n_frames,
        ),
        concentration_ratio=config.concentration_ratio,
        background_level=config.background_level,
        scattering_ratio=config.scattering_ratio,

        info=f'{100*config.scattering_ratio:.2f}; {config.background_level:.2f}',
    )


@pytest.mark.parametrize(
    ['n_times', 'n_numbers'],
    [
        (5000, 2048),
    ],
)
def test_absorbed_noise_vs_value(n_times: int, n_numbers: int, config: AbsorbedSpectrumEmulationConfig) -> None:
    tolerance = 0.1
    number = np.arange(n_numbers)
    value = np.logspace(-3, np.log10(2), n_numbers)

    # theoretical noise
    theoretical_noise = AbsorbedSpectrumNoise(
        detector=config.detector,
        n_frames=config.spectrum.n_frames,
        base_level=config.spectrum_base.level,
        base_noise=EmittedSpectrumNoise(
            detector=config.detector,
            n_frames=config.spectrum_base.n_frames,
        ),
    )(value)

    # emulated noise
    spectrum = emulate_absorbed_spectrum(
        intensity=np.array([config.spectrum_base.level * 10**(-value)]*n_times),
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
    assert np.all((theoretical_noise - emulated_noise) / theoretical_noise <= tolerance)
