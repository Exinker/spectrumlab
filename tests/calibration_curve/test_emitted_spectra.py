import os
import pytest
from functools import partial

import pandas as pd
from tqdm import tqdm

from spectrumlab.alias import Frame
from spectrumlab.calibration_curve import calibrate_spectra
from spectrumlab.emulation.emulation import fetch_emulation, EmittedSpectrumEmulation, SpectrumConfig, EmittedSpectrumEmulationConfig
from spectrumlab.emulation.experiment import EmittedExperimentConfigNaive as ExperimentConfig
from spectrumlab.emulation.noise import EmittedSpectrumNoise
from spectrumlab.emulation.spectrum import Spectrum
from spectrumlab.line.line import Line
from spectrumlab.peak.analyte_peak import GatherAnalytePeakConfig, gather_analyte_peak
from spectrumlab.peak.intensity import IntegralIntensityConfig, InterpolationKind, ApproxIntensityConfig
from spectrumlab.peak.position import InterpolationPositionConfig
from spectrumlab.peak.shape import VoightPeakShape


@pytest.fixture(scope='module')
def config() -> ExperimentConfig:
    return ExperimentConfig.from_ini(
        filedir=os.path.join('.', 'tests', 'emulation', 'ini'),
        filename='config_emission_naive.ini',
    )


@pytest.fixture(scope='module')
def emulation(config: ExperimentConfig) -> EmittedSpectrumEmulation:
    return fetch_emulation(
        config=EmittedSpectrumEmulationConfig(
            device=config.device,
            detector=config.detector,

            line_shape=None,
            apparatus_shape=config.apparatus_shape,
            aperture_shape=config.aperture_shape(
                detector=config.detector,
            ),

            spectrum=SpectrumConfig(
                n_numbers=config.n_numbers,
                n_frames=config.n_frames,
            ),
            concentration_ratio=config.concentration_ratio,
            background_level=config.background_level,

            rx=400,
        )
    )


@pytest.fixture(scope='module')
def spectra(config: ExperimentConfig, emulation: EmittedSpectrumEmulation) -> Frame:

    spectra = pd.DataFrame(
        data={'spectrum': None, 'concentration': None},
        columns=['spectrum', 'concentration'],
        index=pd.MultiIndex.from_product([list(range(config.n_probes)), list(range(config.n_parallels))], names=['probe', 'parallel'])
    )

    # blank
    concentration = config.concentration_blank
    emulation = emulation.setup(position=config.position, concentration=concentration)
    for j in range(200):
        spectra.loc[('blank',j), 'spectrum'] = emulation.run(is_noised=True, is_clipped=True)
        spectra.loc[('blank',j), 'concentration'] = concentration

    # concentrations
    for i, concentration in enumerate(tqdm(config.concentrations, leave=True)):
        emulation = emulation.setup(position=config.position, concentration=concentration)

        for j in range(config.n_parallels):
            spectra.loc[(i,j), 'spectrum'] = emulation.run(is_noised=True, is_clipped=True)
            spectra.loc[(i,j), 'concentration'] = concentration

    #
    return spectra


def calculate_intensity(*args, **kwargs):
    handler = partial(gather_analyte_peak, *args, **kwargs)

    def inner(spectrum: Spectrum) -> float:
        peak = handler(spectrum=spectrum)

        return peak.intensity

    return inner


@pytest.mark.filterwarnings
def test_calculate_intensity_by_integral(config: ExperimentConfig, spectra: Frame):

    # calibrate
    calibration_curve = calibrate_spectra(
        spectra=spectra,
        handler=calculate_intensity(
            line=Line(
                id=0,
                symbol='NA',
                wavelength=config.position,
            ),
            noise=EmittedSpectrumNoise(
                detector=config.detector,
                n_frames=config.n_frames,
            ),
            config=GatherAnalytePeakConfig(
                noise_level=5,
                position=InterpolationPositionConfig(),
                intensity=IntegralIntensityConfig(
                    interval=3,
                    kind=InterpolationKind.LINEAR,
                ),
            ),
        ),
    )

    assert calibration_curve.coeff[1]


@pytest.mark.filterwarnings
def test_calculate_intensity_by_approx(config: ExperimentConfig, spectra: Frame):
    detector = config.detector
    apparatus_shape = config.apparatus_shape

    # calibrate
    calibration_curve = calibrate_spectra(
        spectra=spectra,
        handler=calculate_intensity(
            line=Line(
                id=0,
                symbol='',
                wavelength=25,
            ),
            noise=EmittedSpectrumNoise(
                detector=config.detector,
                n_frames=config.n_frames,
            ),
            config=GatherAnalytePeakConfig(
                noise_level=5,
                position=InterpolationPositionConfig(),
                intensity=ApproxIntensityConfig(
                    approx_shape=VoightPeakShape(
                        width=apparatus_shape.width/detector.config.width,
                        asymmetry=apparatus_shape.asymmetry,
                        ratio=apparatus_shape.ratio,
                        rx=25,
                    ),
                    delta=0,
                ),
            ),
        ),
    )

    assert True
