import numpy as np
import pytest

from spectrumlab.typing import Electron, Percent
from spectrumlab.emulation.detector import Detector
from spectrumlab.emulation.noise import EmittedSpectrumNoise


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
