
import os

from spectrumlab.emulation.emulation import calculate_absorbance

from config import Config


FILE_DIR = os.path.join(os.path.dirname(__file__), 'dat')  # TODO: parameterize it!
FILE_NAME = 'GRAND2_II.ini'  # TODO: parameterize it!


def test_config_base_level():
    config = Config(filedir=FILE_DIR, filename=FILE_NAME)

    assert 0 < config.base_level <= 100, f'base level have to be in (0, 100]!'


def test_config_scattering_ratio():
    config = Config(filedir=FILE_DIR, filename=FILE_NAME)

    assert 0 <= config.scattering_ratio < 1, f'scattering ratio have to be in [0, 1)!'


def test_config_max_background_level():
    config = Config(filedir=FILE_DIR, filename=FILE_NAME)

    if config.scattering_ratio > 0:
        max_background_level = calculate_absorbance(config.scattering_ratio, 1)
        assert config.background_level < max_background_level, f'maximum background level have to be less then {max_background_level:.4f}A at the given scattering level!'
