import os

import numpy as np
import matplotlib.pyplot as plt
import pytest

from spectrumlab.alias import Electron, Percent
from spectrumlab.emulation.concentration_calibration import EmittedExperimentConfigNaive as ExperimentConfig
from spectrumlab.emulation.detector.linear_array_detector import Detector
from spectrumlab.emulation.noise import EmittedSpectrumNoise, AbsorbedSpectrumNoise


@pytest.fixture(scope='module')
def config() -> ExperimentConfig:
    return ExperimentConfig.from_ini(
        filedir=os.path.join('.', 'tests', 'emulation', 'ini'),
        filename='config_emission_naive.ini',
    )


@pytest.mark.parametrize(
    ['detector', ],
    [
        (Detector.BLPP2000, ),
        (Detector.BLPP4000, ),
    ],
)
def test_formula_conversion(detector: Detector):
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
    ['detector', ],
    [
        (Detector.BLPP2000, ),
        (Detector.BLPP4000, ),
    ],
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


# @pytest.mark.parametrize(
#     ['detector', ],
#     [
#         (Detector.BLPP2000, ),
#         (Detector.BLPP4000, ),
#     ],
# )
# def test_emulated_noise(detector: Detector):
#     tolerance = 1e-9



if __name__ == '__main__':
    config = ExperimentConfig.from_ini(
        filedir=os.path.join('.', 'tests', 'emulation', 'ini'),
        filename='config_emission_naive.ini',
    )
    detector = config.detector

    n_numbers = detector.config.n_pixels
    lims = (0, detector.config.capacity)

    number = np.arange(n_numbers)
    value = np.linspace(lims, n_numbers)


    theoretical_noise = EmittedSpectrumNoise(
        detector=detector,
        n_frames=config.n_frames,
        units=Electron,
    )(value)

    plt.plot(value, theoretical_noise)
    plt.show()


    # 

    # # theoretical noise

    # # emulated noise
    # spectrum = emulate_emitted_spectrum(
    #     intensity=np.array([values]*N_TIMES),
    #     number=number,
    #     noise=EmittedSpectrumNoise(
    #         detector=config.detector,
    #         n_frames=config.spectrum.n_frames,
    #     ),
    #     detector=config.detector,
    # )

    # emulated_noise = np.std(spectrum.intensity, ddof=1, axis=0)

    # detection_limit = values[np.argmin(np.abs(values - 3*theoretical_noise))]

    # #
    # plt.figure(
    #     figsize=(12, 4),
    # )

    # # sigma vs. intenisty
    # ax = plt.subplot(1, 2, 1)

    # plt.plot(
    #     values, theoretical_noise,
    #     color='black', label='theoretical',
    # )
    # plt.scatter(
    #     values, emulated_noise,
    #     marker='.', color='red', label=f'emulated',
    # )
    # plt.axvline(
    #     detection_limit,
    #     linestyle=':',
    #     color='red',
    #     label=r'$I_{min}$',
    # )

    # ax.set_xscale('log')

    # plt.xlabel('$I, \%$')
    # plt.ylabel(r'$\sigma_{I}, \%$')

    # plt.grid()
    # plt.legend()

    # # ОСКО vs. intenisty
    # ax = plt.subplot(1, 2, 2)

    # plt.plot(
    #     values, 100 * theoretical_noise / values,
    #     color='black', label='theoretical',
    # )
    # plt.scatter(
    #     values, 100 * emulated_noise / values,
    #     marker='.', color='red', label=f'emulated',
    # )
    # plt.axvline(
    #     x=detection_limit,
    #     linestyle=':',
    #     color='red',
    #     label=r'$I_{min}$',
    # )

    # ax.set_xscale('log')
    # # ax.set_yscale('log')

    # if ylim:
    #     plt.ylim(ylim)

    # plt.xlabel('$I, \%$')
    # plt.ylabel(r'$\sigma_{I}/I, \%$')

    # plt.grid()
    # plt.legend()

    # #
    # plt.tight_layout()
    # plt.show()