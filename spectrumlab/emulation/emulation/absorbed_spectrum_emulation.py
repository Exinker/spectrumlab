from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, interpolate, signal

from spectrumlab.emulation.aperture import Aperture
from spectrumlab.emulation.apparatus import Apparatus
from spectrumlab.emulation.detector import Detector
from spectrumlab.emulation.device import Device
from spectrumlab.emulation.emulation import AbstractEmulation, SpectrumConfig, emulate_emitted_spectrum
from spectrumlab.emulation.line import Line
from spectrumlab.emulation.noise import EmittedSpectrumNoise
from spectrumlab.emulation.noise.absorbed_spectrum_noise import AbsorbedSpectrumNoise, calculate_absorbance_deviation, calculate_squared_relative_standard_deviation
from spectrumlab.emulation.spectrum import AbsorbedSpectrum, EmittedSpectrum
from spectrumlab.picture.config import COLOR
from spectrumlab.typing import Absorbance, Array, MicroMeter, Number, Percent


@dataclass
class SpectrumBaseConfig:
    level: Percent
    n_frames: int = field(default=200)


@dataclass
class AbsorbedSpectrumEmulationConfig:
    device: Device
    detector: Detector

    line: Line
    apparatus: Apparatus
    aperture: Aperture
    spectrum_base: SpectrumBaseConfig
    spectrum: SpectrumConfig

    concentration_ratio: float
    background_level: Absorbance
    scattering_ratio: float

    info: str | None = field(default=None)  # дополнительное текстовое описание (для указания в имени файлов)

    rx: MicroMeter = field(default=100)  # границы построения интерполяции
    dx: MicroMeter = field(default=.01)  # шаг сетки интерполяции


class AbsorbedSpectrumEmulation(AbstractEmulation):
    """Absorbed spectrum emulation."""

    def __init__(self, config: AbsorbedSpectrumEmulationConfig):
        self.config = config

        self.detector = config.detector
        self.line = config.line
        self.apparatus = config.apparatus
        self.aperture = config.aperture

        self.position = None
        self.concentration = None

        self._noise = None
        self._number = None
        self._intensity = None

    # --------        noise        --------
    @property
    def noise(self) -> AbsorbedSpectrumNoise:
        if self._noise is None:
            self._noise = self._get_noise()

        return self._noise

    def _get_noise(self) -> AbsorbedSpectrumNoise:
        config = self.config

        base_level = config.spectrum_base.level
        base_noise = EmittedSpectrumNoise(
            detector=self.detector,
            n_frames=config.spectrum_base.n_frames,
        )

        return AbsorbedSpectrumNoise(
            detector=self.detector,
            n_frames=config.spectrum.n_frames,
            base_level=base_level,
            base_noise=base_noise,
        )

    # --------        number        --------
    @property
    def number(self) -> Array:
        if self._number is None:
            self._number = self._get_number()

        return self._number

    def _get_number(self) -> Array:
        config = self.config

        n_numbers = config.spectrum.n_numbers
        number = np.arange(n_numbers)

        return number

    # --------        intensity        --------
    @property
    def intensity(self) -> Array:
        if self._intensity is None:
            raise Exception('setup the emulation before!')

        return self._intensity

    def _get_intensity(self, number: Array, position: Number, concentration: float, show: bool = False, ylim: tuple[float, float] | None = None) -> Array:
        config = self.config
        number = self.number
        line = self.line
        apparatus = self.apparatus
        aperture = self.aperture

        device = config.device
        detector = config.detector
        rx = config.rx
        dx = config.dx

        I0 = config.spectrum_base.level  # base level, in percent
        S0 = config.spectrum_base.level * config.scattering_ratio  # scattering level, in percent

        B = calculate_absorbance(I0*10**(-config.background_level), I0, config.scattering_ratio)  # true background (recalculate to true absorbance), in A
        B0 = (I0 - S0)*10**(-B)  # true background, in percent

        L = config.concentration_ratio  # path length (concentration coefficient to agreement between theory and emulation)
        D = device.config.dispersion

        # physical line shape
        I = L/D*concentration  # it's divided by D because an amplitude of absorption is independent of dispersion!

        physical_line = lambda x: (I0 - S0)*10**(
            -(B + line(x, position=0, intensity=I))
        )

        # apparatus line function
        span = 10*rx

        x = np.linspace(-span, +span, 2*int(span/dx) + 1)        
        apparatus_line = interpolate.interp1d(
            x,
            B0 + signal.convolve(apparatus(x, 0), physical_line(x) - B0, mode='same') * 2*span/len(x),
            kind='linear',
            bounds_error=False,
            fill_value=np.nan,
        )

        # intensity (detector's output signal)
        x0 = position*detector.pitch

        intensity = np.zeros(number.shape)
        for n in number:
            intensity[n] = integrate.quad(
                lambda x: apparatus_line(x - x0) * aperture(x, n),
                n*detector.pitch - rx,
                n*detector.pitch + rx,
            )[0]

        intensity += S0  # add scattering radiation
        intensity += I0 * (1 - integrate.quad(lambda x: aperture(x, 0), -rx, +rx)[0])  # approximated aperture characteristics correction

        # show
        if show:
            x = np.linspace(min(number)*detector.pitch, max(number)*detector.pitch, 1000)  # in MicroMeter

            #
            plt.figure(figsize=(12, 4))

            # title = '\n'.join([
            #     f'dispersion: {device.config.dispersion:.4f}, nm/mm',
            #     f'detector: {detector.config.name}',
            # ])
            # plt.suptitle(title)

            # in emission units
            plt.subplot(1, 2, 1)

            plt.plot(x/detector.pitch, S0 + physical_line(x - x0), label=r'$I(\lambda)$')  # f'physical line'
            plt.plot(x/detector.pitch, S0 + apparatus_line(x - x0), label=r'$I^{F}(\lambda)$')  # f'apparatus line'
            plt.fill_between(
                number,
                y1=S0 + np.full(number.shape, B0),
                y2=intensity,
                step='mid', alpha=0.2, facecolor=COLOR['pink'], edgecolor='k', label=r'$s_{k}$',  # f'spectrum'
            )

            plt.xlabel(r'number')
            plt.ylabel(r'$I$ $[\%]$')

            plt.grid(color='grey', linestyle=':')
            plt.legend(loc='lower right')

            # in absorption units
            plt.subplot(1, 2, 2)

            plt.plot(x/detector.pitch, calculate_absorbance(S0 + physical_line(x - x0), I0), label=r'$A(\lambda)$')  # f'physical line'
            plt.plot(x/detector.pitch, calculate_absorbance(S0 + apparatus_line(x - x0), I0), label=r'$A^{F}(\lambda)$')  # f'apparatus line'
            plt.fill_between(
                number,
                y1=np.full(number.shape, config.background_level),
                y2=calculate_absorbance(intensity, I0),
                step='mid', alpha=0.2, facecolor=COLOR['pink'], edgecolor='k', label=r'$a_{k}$',  # f'spectrum'
            )

            if ylim is None:
                absorbance = calculate_absorbance(intensity, I0)
                spam = (max(absorbance) - min(absorbance))
                plt.ylim((min(absorbance) - spam/10, max(absorbance) + spam/2))
            else:
                plt.ylim(ylim)

            plt.xlabel(r'number')
            plt.ylabel(r'$A$')

            plt.grid(color='grey', linestyle=':')
            plt.legend(loc='upper right')

            plt.tight_layout()
            plt.show()

        # return intensity
        return intensity

    # --------        handlers        --------
    def setup(
            self,
            position: Number,
            concentration: float,
            show: bool = False,
            ylim: tuple[float, float] | None = None,
            ) -> 'AbstractEmulation':
        """Setup emulation of absorbed spectrum."""
        self.position = position
        self.concentration = concentration

        #
        self._intensity = self._get_intensity(number=self.number, position=position, concentration=concentration, show=show, ylim=ylim)

        #
        return self

    def run(self, is_noised: bool = True, is_clipped: bool = True, show: bool = False, random_state: int | None = None) -> AbsorbedSpectrum:
        """Run emulation."""
        config = self.config

        # set random state
        if random_state is not None:
            np.random.seed(random_state)

        # init spectrum
        spectrum = emulate_absorbed_spectrum(
            number=self.number,
            intensity=self.intensity,
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
            is_noised=is_noised,
            is_clipped=is_clipped,
        )

        # show spectrum
        if show:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), tight_layout=True)

            plt.fill_between(
                spectrum.number,
                y1=config.background_level,
                y2=spectrum.intensity,
                step='mid', alpha=0.2, facecolor=COLOR['pink'], edgecolor='k', label='paek',
            )
            plt.xlabel(r'number')
            plt.ylabel({
                EmittedSpectrum: r'$I$ [$\%$]',
                AbsorbedSpectrum: r'$A$',
            }.get(type(spectrum)))
            plt.grid(color='grey', linestyle=':')
            plt.show()

        # return spectrum
        return spectrum


# --------        handlers        --------
def calculate_absorbance(level, base_level, scattering_ratio=0):
    scattering_level = scattering_ratio * base_level

    return np.log10(
        (base_level - scattering_level) / (level - scattering_level)  # (base_level - scattering_level) / (level - scattering_level)
    )


def emulate_absorbed_spectrum(
        number: Array,
        intensity: Array,
        noise: EmittedSpectrumNoise,
        base_level: float,
        base_noise: EmittedSpectrumNoise,
        detector: Detector,
        is_noised: bool = True,
        is_clipped: bool = True,
        ) -> AbsorbedSpectrum:

    # init base spectrum
    spectrum_base = emulate_emitted_spectrum(
        number=number,
        intensity=np.full(number.shape, base_level),  # массив intensity - одномерный (так как получен путем n_frames накоплений)
        noise=base_noise,
        detector=detector,
        is_noised=is_noised,
        is_clipped=is_clipped,
    )

    # init recorded spectrum
    spectrum_recorded = emulate_emitted_spectrum(
        number=number,
        intensity=intensity,
        noise=noise,
        detector=detector,
        is_noised=is_noised,
        is_clipped=is_clipped,
    )

    # init spectrum
    spectrum = AbsorbedSpectrum(
        intensity=calculate_absorbance(
            spectrum_recorded.intensity,
            spectrum_base.intensity,
        ),
        deviation=calculate_absorbance_deviation(
            part_base=calculate_squared_relative_standard_deviation(
                value=base_level,
                noise=base_noise,
            ),
            part_recorded=calculate_squared_relative_standard_deviation(
                value=intensity,
                noise=noise,
            ),
        ),
        base=spectrum_base.intensity,
        detector=detector,
        number=number,
        clipped=(spectrum_base.clipped | spectrum_recorded.clipped),
    )

    # return spectrum
    return spectrum
