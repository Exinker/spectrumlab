import os

import pytest

from spectrumlab.emulations.concentration_calibration import EmittedExperimentConfigNaive as ExperimentConfig
from spectrumlab.emulations.emulators import EmittedSpectrumEmulator, EmittedSpectrumEmulationConfig, SpectrumConfig
from spectrumlab.emulations.emulators import fetch_emulation


@pytest.fixture(scope='module')
def config() -> ExperimentConfig:
    return ExperimentConfig.from_ini(
        filedir=os.path.join(os.path.dirname(__file__), 'ini'),
        filename='config_emission_naive.ini',
    )


@pytest.fixture(scope='module')
def emulation(config: ExperimentConfig) -> EmittedSpectrumEmulator:
    return fetch_emulation(
        config=EmittedSpectrumEmulationConfig(
            device=config.device,
            detector=config.detector,

            line=None,
            apparatus=config.apparatus,
            aperture=config.aperture,

            spectrum=SpectrumConfig(
                n_numbers=config.n_numbers,
                n_frames=config.n_frames,
            ),
            concentration_ratio=config.concentration_ratio,
            background_level=config.background_level,

            rx=400,
        ),
    )
