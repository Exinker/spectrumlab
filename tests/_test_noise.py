from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from spectrumlab.emulation.emulation import emulate_emitted_spectrum, emulate_absorbed_spectrum, EmulationConfig, EmittedSpectrumEmulationConfig, AbsorbedSpectrumEmulationConfig
from spectrumlab.emulation.noise import EmittedSpectrumNoise, AbsorbedSpectrumNoise


N_NUMBERS, N_TIMES = 100, 50


def test_emitted_noise_vs_value(config: EmittedSpectrumEmulationConfig, lims=(-3, 2), ylim: Optional[tuple[float, float]] = None) -> None:
    number = np.arange(N_NUMBERS)
    values = np.logspace(*lims, N_NUMBERS)

    # theoretical noise
    theoretical_noise = EmittedSpectrumNoise(
        detector=config.detector,
        n_frames=config.spectrum.n_frames,
    )(values)

    # emulated noise
    spectrum = emulate_emitted_spectrum(
        intensity=np.array([values]*N_TIMES),
        number=number,
        noise=EmittedSpectrumNoise(
            detector=config.detector,
            n_frames=config.spectrum.n_frames,
        ),
        detector=config.detector,
    )

    emulated_noise = np.std(spectrum.intensity, ddof=1, axis=0)

    detection_limit = values[np.argmin(np.abs(values - 3*theoretical_noise))]

    #
    plt.figure(
        figsize=(12, 4),
    )

    # sigma vs. intenisty
    ax = plt.subplot(1, 2, 1)

    plt.plot(
        values, theoretical_noise,
        color='black', label='theoretical',
    )
    plt.scatter(
        values, emulated_noise,
        marker='.', color='red', label=f'emulated',
    )
    plt.axvline(
        detection_limit,
        linestyle=':',
        color='red',
        label=r'$I_{min}$',
    )

    ax.set_xscale('log')

    plt.xlabel('$I, \%$')
    plt.ylabel(r'$\sigma_{I}, \%$')

    plt.grid()
    plt.legend()

    # ОСКО vs. intenisty
    ax = plt.subplot(1, 2, 2)

    plt.plot(
        values, 100 * theoretical_noise / values,
        color='black', label='theoretical',
    )
    plt.scatter(
        values, 100 * emulated_noise / values,
        marker='.', color='red', label=f'emulated',
    )
    plt.axvline(
        x=detection_limit,
        linestyle=':',
        color='red',
        label=r'$I_{min}$',
    )

    ax.set_xscale('log')
    # ax.set_yscale('log')

    if ylim:
        plt.ylim(ylim)

    plt.xlabel('$I, \%$')
    plt.ylabel(r'$\sigma_{I}/I, \%$')

    plt.grid()
    plt.legend()

    #
    plt.tight_layout()
    plt.show()


def test_absorbed_noise_vs_value(config: AbsorbedSpectrumEmulationConfig, lims=(-3, np.log10(2)), ylim: Optional[tuple[float, float]] = None) -> None:
    number = np.arange(N_NUMBERS)
    values = np.logspace(*lims, N_NUMBERS)

    # theoretical noise
    theoretical_noise = AbsorbedSpectrumNoise(
        detector=config.detector,
        n_frames=config.spectrum.n_frames,
        base_level=config.spectrum_base.level,
        base_noise=EmittedSpectrumNoise(
            detector=config.detector,
            n_frames=config.spectrum_base.n_frames,
        ),
    )(values)

    # emulated noise
    scattering_level = config.scattering_ratio * config.spectrum_base.level

    spectrum = emulate_absorbed_spectrum(
        intensity=np.array([config.spectrum_base.level * 10**(-values)]*N_TIMES),
        number=number,
        noise=EmittedSpectrumNoise(
            detector=config.detector,
            n_frames=config.spectrum.n_frames,
        ),
        base_level=config.spectrum_base.level,
        base_noise=EmittedSpectrumNoise(
            detector=config.detector,
            n_frames=config.spectrum_base.n_frames,
        ),
        detector=config.detector,
    )

    emulated_noise = np.std(spectrum.intensity, ddof=1, axis=0)

    detection_limit = values[np.argmin(np.abs(values - 3*theoretical_noise))]

    #
    plt.figure(
        figsize=(12, 4),
    )

    # sigma vs. intenisty
    ax = plt.subplot(1, 2, 1)

    plt.plot(values, theoretical_noise, color='black', label='theoretical')
    plt.scatter(values, emulated_noise, marker='.', color='red', label=f'emulated')

    plt.xlabel('A')
    plt.ylabel(r'$\sigma_{A}$')

    plt.grid()
    plt.legend()

    # ОСКО vs. intenisty
    ax = plt.subplot(1, 2, 2)

    plt.plot(values, 100 * theoretical_noise / values, color='black', label='theoretical')
    plt.scatter(values, 100 * emulated_noise / values, marker='.', color='red', label=f'emulated')

    if ylim:
        plt.ylim(ylim)

    plt.xlabel(r'$A$')
    plt.ylabel(r'$\sigma_{A}/A, \%$')

    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()


def test_absorbed_noise_vs_base_value(config: AbsorbedSpectrumEmulationConfig, ylim: Optional[tuple[float, float]] = None) -> None:
    number = np.arange(N_NUMBERS)
    base_levels = np.logspace(np.log10(config.scattering_ratio + 1), np.log10(100), N_NUMBERS)

    # theoretical noise
    theoretical_noise = []
    for base_level in base_levels:
        I0 = base_level
        S0 = config.scattering_ratio

        value = AbsorbedSpectrumNoise(
            detector=config.detector,
            n_frames=config.spectrum.n_frames,
            base_level=(I0 - S0),
            base_noise=EmittedSpectrumNoise(
                detector=config.detector,
                n_frames=config.spectrum_base.n_frames,
            ),
        )(0)

        theoretical_noise.append(value)

    theoretical_noise = np.array(theoretical_noise)

    # emulated noise
    emulated_noise = []
    for base_level in base_levels:
        I0 = base_level
        S0 = config.scattering_ratio

        spectrum = emulate_absorbed_spectrum(
            intensity=(I0 - S0) * 10**(-np.full((N_NUMBERS,), 0)),
            number=number,
            noise=EmittedSpectrumNoise(
                detector=config.detector,
                n_frames=config.spectrum.n_frames,
            ),
            base_level=(I0 - S0),
            base_noise=EmittedSpectrumNoise(
                detector=config.detector,
                n_frames=config.spectrum_base.n_frames,
            ),
            detector=config.detector,
        )

        value = np.std(spectrum.intensity, ddof=1, axis=0)
        emulated_noise.append(value)

    emulated_noise = np.array(emulated_noise)

    #
    plt.figure(
        figsize=(12, 4),
    )

    # sigma vs. intenisty
    ax = plt.subplot(1, 2, 1)

    plt.plot(base_levels, theoretical_noise, color='black', label='theoretical')
    plt.scatter(base_levels, emulated_noise, marker='.', color='red', label=f'emulated')

    ax.set_xscale('log')

    plt.xlabel('$I_{0}, \%$')
    plt.ylabel(r'$\sigma_{A}$')

    plt.grid()
    plt.legend()

    # sigma vs. ОСКО
    ax = plt.subplot(1, 2, 2)

    plt.plot(base_levels, 100 * theoretical_noise / base_levels, color='black', label='theoretical')
    plt.scatter(base_levels, 100 * emulated_noise / base_levels, marker='.', color='red', label=f'emulated')

    ax.set_xscale('log')

    if ylim:
        plt.ylim(ylim)

    plt.xlabel('$I_{0}, \%$')
    plt.ylabel(r'$\sigma_{A}/I_{0}, \%$')

    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()


# --------        handlers        --------
def test_noise_vs_value(config: EmulationConfig, *args, **kwargs):

    if isinstance(config, EmittedSpectrumEmulationConfig):
        return test_emitted_noise_vs_value(config=config, *args, **kwargs)
    if isinstance(config, AbsorbedSpectrumEmulationConfig):
        return test_absorbed_noise_vs_value(config=config, *args, **kwargs)

    raise TypeError()


def test_noise_vs_base_value(config: EmulationConfig, *args, **kwargs):

    if isinstance(config, EmittedSpectrumEmulationConfig):
        raise TypeError()
    if isinstance(config, AbsorbedSpectrumEmulationConfig):
        return test_absorbed_noise_vs_base_value(config=config, *args, **kwargs)

    raise TypeError()
