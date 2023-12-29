import os
import pytest

import numpy as np

from spectrumlab.emulation.calibration_curve.calibration_curve import CalibrationCurve, CalibrationCurveConfig
from spectrumlab.emulation.detector.linear_array_detector import Detector
from spectrumlab.emulation.emulation import fetch_emulation, Emulation, SpectrumConfig, EmittedSpectrumEmulationConfig
from spectrumlab.emulation.experiment import EmittedExperimentConfigNaive as ExperimentConfig
from spectrumlab.emulation.intensity import IntensityConfig, IntegralIntensityConfig, AmplitudeIntensityConfig


@pytest.fixture(scope='module')
def config() -> ExperimentConfig:
    return ExperimentConfig.from_ini(
        filedir=os.path.join('.', 'tests', 'emulation', 'ini'),
        filename='config_emission_naive.ini',
    )


@pytest.fixture(scope='module')
def emulation(config: ExperimentConfig) -> Emulation:
    return fetch_emulation(
        config=EmittedSpectrumEmulationConfig(
            device=config.device,
            detector=config.detector,

            line_shape=None,
            apparatus_shape=config.apparatus_shape,
            aperture=config.aperture,

            spectrum=SpectrumConfig(
                n_numbers=config.n_numbers,
                n_frames=config.n_frames,
            ),
            concentration_ratio=config.concentration_ratio,
            background_level=config.background_level,

            # info='',
        )
    )


@pytest.fixture(scope='module')
def calibration_curve(config: ExperimentConfig, emulation: Emulation) -> CalibrationCurve:
    calibration_curve = CalibrationCurve(
        emulation=emulation,
        config=CalibrationCurveConfig(
            intensity_config=config.intensity,
            n_probes=config.n_probes,
            n_parallels=config.n_parallels,
        ),
    )
    calibration_curve = calibration_curve.setup(
        position=config.position,
        concentrations=config.concentrations,
    )
    calibration_curve = calibration_curve.run(
        verbose=False,
        show=False,
        write=False,
    )

    return calibration_curve


class TestCalibrationCurve:
    tolerance = 1e-9

    # --------        coeff        --------
    def test_calibration_curve_coeff(self, config: ExperimentConfig, calibration_curve: CalibrationCurve):
        intercept, slope = calibration_curve.coeff

        assert np.abs(slope - 1) <= self.tolerance

    # --------        LOD        --------
    def test_calibration_curve_lod(self, config: ExperimentConfig, calibration_curve: CalibrationCurve):
        lod = self.calcualte_lod(detector=config.detector, config=config.intensity)

        assert np.abs(calibration_curve.lod.intensity - lod) < self.tolerance

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
