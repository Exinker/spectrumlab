import os

import numpy as np
import matplotlib.pyplot as plt
import pytest

from spectrumlab.alias import Electron, Percent
from spectrumlab.emulation.concentration_calibration import EmittedExperimentConfigNaive as ExperimentConfig
from spectrumlab.emulation.detector.linear_array_detector import Detector
from spectrumlab.emulation.emulation import emulate_emitted_spectrum, emulate_absorbed_spectrum
from spectrumlab.emulation.noise import EmittedSpectrumNoise, AbsorbedSpectrumNoise


@pytest.mark.parametrize(
    'detector',
    [detector for detector in Detector],
)
def test_percent_units(detector: Detector):
    tolerance = 1e-9
    kc = 100 / detector.config.capacity

    value = np.arange(0, detector.config.capacity+1, 1)
    noise = EmittedSpectrumNoise(
        detector=detector,
        n_frames=1,
        units=Electron,
    )(value)  # in electron

    noise_hat = EmittedSpectrumNoise(
        detector=detector,
        n_frames=1,
        units=Percent,
    )(value*kc)/kc  # in electron

    assert np.all(np.abs(noise_hat - noise) < tolerance)


@pytest.mark.parametrize(
    'detector',
    [detector for detector in Detector],
)
def test_n_frames(detector: Detector):
    tolerance = 1e-9

    value = np.arange(0, detector.config.capacity+1, 1)
    noise = EmittedSpectrumNoise(
        detector=detector,
        n_frames=1,
        units=Electron,
    )(value)

    for n_frames in np.logspace(0, 6, 10):
        noise_hat = EmittedSpectrumNoise(
            detector=detector,
            n_frames=n_frames,
            units=Electron,
        )(value) * np.sqrt(n_frames)

        assert np.all(np.abs(noise_hat - noise) < tolerance)


@pytest.mark.parametrize(
    'detector',
    [detector for detector in Detector],
)
def test_emulate_emitted_spectrum_noise(detector: Detector):
    tolerance = 1e-1
    n_times = 10_000
    n_numbers = detector.config.n_pixels

    number = np.arange(n_numbers)
    intensity = np.linspace(0, 100, n_numbers)

    theoretical_noise = EmittedSpectrumNoise(
        detector=detector,
        n_frames=1,
    )(intensity)

    spectrum = emulate_emitted_spectrum(
        number=number,
        intensity=np.array([intensity]*n_times),
        noise=EmittedSpectrumNoise(
            detector=detector,
            n_frames=1,
        ),
        detector=detector,

        is_clipped=False,  # TODO: don't remove! It's for correct estimate a noise about 100[%]!
    )
    emulated_noise = np.std(spectrum.intensity, ddof=1, axis=0)

    assert np.all((emulated_noise - theoretical_noise)/theoretical_noise < tolerance)  # relative deviation
