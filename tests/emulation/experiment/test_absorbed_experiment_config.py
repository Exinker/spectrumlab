import os
import pytest

from spectrumlab.emulation.emulation import calculate_absorbance
from spectrumlab.emulation.concentration_calibration import AbsorbedExperimentConfig as ExperimentConfig


@pytest.fixture(scope='module')
def config() -> ExperimentConfig:
    return ExperimentConfig.from_ini(
        filedir=os.path.join('.', 'tests', 'emulation', 'ini'),
        filename='config_absorption_Ag338.289_I.ini',
    )


class TestConfig:

    def test_config_base_level(self, config: ExperimentConfig):
        assert 0 < config.base_level <= 100, f'base level have to be in (0, 100]!'

    def test_config_scattering_ratio(self, config: ExperimentConfig):
        assert 0 <= config.scattering_ratio < 1, f'scattering ratio have to be in [0, 1)!'

    def test_config_max_background_level(self, config: ExperimentConfig):
        if config.scattering_ratio > 0:
            max_background_level = calculate_absorbance(config.scattering_ratio, 1)
            assert config.background_level < max_background_level, f'maximum background level have to be less then {max_background_level:.4f}A at the given scattering level!'
