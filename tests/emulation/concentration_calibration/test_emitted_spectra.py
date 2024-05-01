import os
from functools import partial

import numpy as np
import pandas as pd
import pytest
from tqdm import tqdm

from spectrumlab.concentration_calibration import calibrate
from spectrumlab.emulation.concentration_calibration import EmittedExperimentConfigNaive as ExperimentConfig
from spectrumlab.emulation.emulation import EmittedSpectrumEmulation, EmittedSpectrumEmulationConfig, SpectrumConfig
from spectrumlab.emulation.emulation import fetch_emulation
from spectrumlab.emulation.noise import EmittedSpectrumNoise
from spectrumlab.emulation.spectrum import Spectrum
from spectrumlab.grid import InterpolationKind
from spectrumlab.line import Line
from spectrumlab.peak.analyte_peak import AnalytePeak, FactoryAnalytePeak
from spectrumlab.peak.intensity import AmplitudeIntensityCalculator, ApproxIntensityCalculator, IntegralIntensityCalculator
from spectrumlab.peak.position import InterpolationPositionCalculator
from spectrumlab.peak.shape import VoigtPeakShape
from spectrumlab.types import Frame


@pytest.fixture(scope='module')
def config() -> ExperimentConfig:
    return ExperimentConfig.from_ini(
        filedir=os.path.join(os.path.dirname(__file__), 'ini'),
        filename='config_emission_naive.ini',
    )


@pytest.fixture(scope='module')
def emulation(config: ExperimentConfig) -> EmittedSpectrumEmulation:
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


@pytest.fixture(scope='module')
def spectra(config: ExperimentConfig, emulation: EmittedSpectrumEmulation) -> Frame:

    spectra = pd.DataFrame(
        data={'spectrum': None, 'concentration': None},
        columns=['spectrum', 'concentration'],
        index=pd.MultiIndex.from_product([list(range(config.n_probes)), list(range(config.n_parallels))], names=['probe', 'parallel']),
    )

    # blank
    concentration = config.concentration_blank
    emulation = emulation.setup(position=config.position, concentration=concentration)
    for j in range(200):
        spectra.loc[('blank', j), 'spectrum'] = emulation.run(is_noised=True, is_clipped=True)
        spectra.loc[('blank', j), 'concentration'] = concentration

    # concentrations
    for i, concentration in enumerate(tqdm(config.concentrations, leave=True)):
        emulation = emulation.setup(position=config.position, concentration=concentration)

        for j in range(config.n_parallels):
            spectra.loc[(i, j), 'spectrum'] = emulation.run(is_noised=True, is_clipped=True)
            spectra.loc[(i, j), 'concentration'] = concentration

    #
    return spectra


def calculate_intensity(factory: FactoryAnalytePeak, *args, **kwargs):
    create = partial(factory.create, *args, **kwargs)

    def inner(spectrum: Spectrum) -> float:
        peak = create(spectrum=spectrum)

        return peak.intensity

    return inner


class TestConcentrationCalibration:
    tolerance = 1e-2

    @pytest.mark.filterwarnings
    def test_calculate_intensity_by_amplitude(self, config: ExperimentConfig, spectra: Frame):

        # calibrate
        concentration_calibration = calibrate(
            spectra=spectra,
            handler=calculate_intensity(
                line=Line(
                    symbol='NA',
                    wavelength=config.position,
                ),
                noise=EmittedSpectrumNoise(
                    detector=config.detector,
                    n_frames=config.n_frames,
                ),
                factory=AnalytePeak.factory(
                    noise_level=5,
                    position_calculator=InterpolationPositionCalculator(),
                    intensity_calculator=AmplitudeIntensityCalculator(),
                ),
            ),
        )

        #
        intercept, slope = concentration_calibration.coeff
        assert np.abs(slope - 1) < self.tolerance

    @pytest.mark.filterwarnings
    def test_calculate_intensity_by_integral(self, config: ExperimentConfig, spectra: Frame):

        # calibrate
        concentration_calibration = calibrate(
            spectra=spectra,
            handler=calculate_intensity(
                line=Line(
                    symbol='NA',
                    wavelength=config.position,
                ),
                noise=EmittedSpectrumNoise(
                    detector=config.detector,
                    n_frames=config.n_frames,
                ),
                factory=AnalytePeak.factory(
                    noise_level=5,
                    position_calculator=InterpolationPositionCalculator(),
                    intensity_calculator=IntegralIntensityCalculator(
                        interval=3,
                        kind=InterpolationKind.LINEAR,
                    ),
                ),
            ),
        )

        #
        intercept, slope = concentration_calibration.coeff
        assert np.abs(slope - 1) < self.tolerance

    @pytest.mark.filterwarnings
    def test_calculate_intensity_by_approx(self, config: ExperimentConfig, spectra: Frame):
        detector = config.detector
        apparatus = config.apparatus

        # calibrate
        concentration_calibration = calibrate(
            spectra=spectra,
            handler=calculate_intensity(
                line=Line(
                    symbol='NA',
                    wavelength=25,
                ),
                noise=EmittedSpectrumNoise(
                    detector=config.detector,
                    n_frames=config.n_frames,
                ),
                factory=AnalytePeak.factory(
                    noise_level=5,
                    position_calculator=InterpolationPositionCalculator(),
                    intensity_calculator=ApproxIntensityCalculator(
                        shape=VoigtPeakShape(
                            width=apparatus.shape.width/detector.pitch,
                            asymmetry=apparatus.shape.asymmetry,
                            ratio=apparatus.shape.ratio,
                            rx=25,
                        ),
                        delta=0,
                    ),
                ),
            ),
        )

        #
        intercept, slope = concentration_calibration.coeff
        assert np.abs(slope - 1) < self.tolerance
