import os

import numpy as np
import pytest

from spectrumlab.emulation.concentration_calibration import ConcentrationCalibration, ConcentrationCalibrationConfig, EmittedExperimentConfigNaive as ExperimentConfig
from spectrumlab.emulation.detector import Detector
from spectrumlab.emulation.emulation import EmittedSpectrumEmulationConfig, Emulation, SpectrumConfig
from spectrumlab.emulation.emulation import fetch_emulation
from spectrumlab.emulation.intensity import AmplitudeIntensityConfig, IntegralIntensityConfig, IntensityConfig


@pytest.fixture(scope='module')
def config() -> ExperimentConfig:
    return ExperimentConfig.from_ini(
        filedir=os.path.join(os.path.dirname(__file__), 'ini'),
        filename='config_emission_naive.ini',
    )


@pytest.fixture(scope='module')
def emulation(config: ExperimentConfig) -> Emulation:
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

            # info='',
        ),
    )


@pytest.fixture(scope='module')
def concentration_calibration(config: ExperimentConfig, emulation: Emulation) -> ConcentrationCalibration:
    concentration_calibration = ConcentrationCalibration(
        emulation=emulation,
        config=ConcentrationCalibrationConfig(
            intensity_config=config.intensity,
            n_probes=config.n_probes,
            n_parallels=config.n_parallels,
        ),
    )
    concentration_calibration = concentration_calibration.setup(
        position=config.position,
        concentrations=config.concentrations,
    )
    concentration_calibration = concentration_calibration.run(
        verbose=False,
        show=False,
        write=False,
    )

    return concentration_calibration


class TestConcentrationCalibration:
    tolerance = 1e-9

    # --------        coeff        --------
    def test_concentration_calibration_coeff(self, config: ExperimentConfig, concentration_calibration: ConcentrationCalibration):
        intercept, slope = concentration_calibration.coeff

        assert np.abs(slope - 1) <= self.tolerance

    # --------        LOD        --------
    def test_concentration_calibration_lod(self, config: ExperimentConfig, concentration_calibration: ConcentrationCalibration):
        lod = self.calcualte_lod(detector=config.detector, config=config.intensity)

        assert np.abs(concentration_calibration.lod.intensity - lod) < self.tolerance

    @staticmethod
    def calcualte_lod(detector: Detector, config: IntensityConfig, k: float = 3) -> float:
        """Calculate theoretical LOD value."""

        if isinstance(config, AmplitudeIntensityConfig):
            read_noise = 100 * detector.config.read_noise / detector.config.capacity
            deviation = read_noise

            return k * deviation

        if isinstance(config, IntegralIntensityConfig):
            read_noise = 100 * detector.config.read_noise / detector.config.capacity
            deviation = np.sqrt(config.interval) * read_noise

            return k * deviation

        raise ValueError(f'config: {config} is not supported yet!')
