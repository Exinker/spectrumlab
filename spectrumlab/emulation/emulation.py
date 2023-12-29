"""
Emitted and absorbed spectrum emulation.

    Author: Vaschenko Pavel
    Email: vaschenko@vmk.ru
    Date: 2013.04.12

"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeAlias, overload

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, interpolate, signal

from spectrumlab.alias import Array, Absorbance, MilliSecond, Micro, Percent, Number
from spectrumlab.picture.config import COLOR
from spectrumlab.emulation.detector.linear_array_detector import Detector
from spectrumlab.emulation.apparatus import Apparatus
from spectrumlab.emulation.aperture import Aperture
from spectrumlab.emulation.device import Device
from spectrumlab.emulation.line import Line
from spectrumlab.emulation.noise import Noise, EmittedSpectrumNoise, AbsorbedSpectrumNoise, calculate_squared_relative_standard_deviation, calculate_absorbance_deviation
from spectrumlab.emulation.spectrum import Spectrum, EmittedSpectrum, AbsorbedSpectrum, HighDynamicRangeEmittedSpectrum


class EmulationInterface(ABC):
    """Interface to emulate spectrum."""

    @property
    @abstractmethod
    def noise(self) -> Noise:
        raise NotImplementedError

    @property
    @abstractmethod
    def number(self) -> Array[Number]:
        raise NotImplementedError

    @property
    @abstractmethod
    def intensity(self) -> Array[Percent]:
        raise NotImplementedError

    # --------        handlers        --------
    @abstractmethod
    def setup(self, number: Array[Number], position: Number, concentration: float) -> 'EmulationInterface':
        """Setup emulation of spectrum."""
        raise NotImplementedError

    @abstractmethod
    def run(self, show: bool = False, random_state: int | None = None) -> Spectrum:
        """Run emulation."""
        raise NotImplementedError


# --------        emission emulation        --------
@dataclass
class SpectrumConfig:
    n_numbers: int = field(default=50)
    n_frames: int = field(default=1)


@dataclass
class EmittedSpectrumEmulationConfig:
    device: Device
    detector: Detector

    line: Line | None
    apparatus: Apparatus
    aperture: Aperture
    spectrum: SpectrumConfig

    concentration_ratio: float
    background_level: Percent

    info: str | None = field(default=None)  # дополнительное текстовое описание (для указания в имени файлов)

    rx: Micro = field(default=100)  # границы построения интерполяции
    dx: Micro = field(default=.01)  # шаг сетки интерполяции


def emulate_emitted_spectrum(number: Array[Number], intensity: Array[Percent], noise: EmittedSpectrumNoise, detector: Detector, is_noised: bool = True, is_clipped: bool = True) -> EmittedSpectrum:
    """Fabric to emulate emitted spectrum."""

    # add noise
    if is_noised:
        values = noise(intensity)*np.random.randn(*intensity.shape)
        intensity = intensity + values

    # add clipping
    clipped = np.full(intensity.shape, False)
    if is_clipped:
        cond = intensity >= 100
        clipped[cond] = True
        intensity[cond] = 100

    # return spectrum
    return EmittedSpectrum(
        number=number,
        deviation=noise(intensity),
        intensity=intensity,
        clipped=clipped,
        detector=detector,
    )


class EmittedSpectrumEmulation(EmulationInterface):
    """Emitted spectrum emulation."""

    def __init__(self, config: EmittedSpectrumEmulationConfig):
        self.config = config

        self.detector = config.detector
        self.line = config.line
        self.apparatus = config.apparatus
        self.aperture = config.aperture

        self.position = None
        self.concentration = None

        self._physical_line = None
        self._apparatus_line = None

        self._x_grid = None
        self._y_grid = None

        self._noise = None
        self._number = None
        self._intensity = None

    # --------        noise        --------
    @property
    def noise(self) -> EmittedSpectrumNoise:
        if self._noise is None:
            self._noise = self._get_noise()

        return self._noise

    def _get_noise(self) -> EmittedSpectrumNoise:
        config = self.config

        return EmittedSpectrumNoise(
            detector=config.detector,
            n_frames=config.spectrum.n_frames,
        )

    # --------        number        --------
    @property
    def number(self) -> Array[Number]:
        if self._number is None:
            self._number = self._get_number()

        return self._number

    def _get_number(self) -> Array[Number]:
        config = self.config

        n_numbers = config.spectrum.n_numbers
        number = np.arange(n_numbers)

        return number

    # --------        intensity        --------
    @property
    def x_grid(self) -> Array[Micro]:
        if self._x_grid is None:
            config = self.config

            rx = config.rx
            dx = config.dx
            # x_grid = np.arange(-rx, rx+dx, dx)
            x_grid = np.linspace(-rx, +rx, 2*rx*int(1/dx) + 1)

            self._x_grid = x_grid

        return self._x_grid

    @property
    def y_grid(self) -> Array[Percent]:

        if self._y_grid is None:
            config = self.config
            line = self.line
            apparatus = self.apparatus
            aperture = self.aperture

            x_grid = self.x_grid
            rx = config.rx

            # physical line function
            if line is None:
                self._physical_line = None
            else:
                self._physical_line = lambda x: line(x, 0, 1)

            # apparatus line function
            if line is None:
                self._apparatus_line = lambda x: apparatus(x, 0)

            else:
                self._apparatus_line = interpolate.interp1d(
                    x_grid,
                    signal.convolve(self._physical_line(x_grid), apparatus(x_grid, 0), mode='same') * 2*rx/len(x_grid),
                    kind='linear',
                    bounds_error=False,
                    fill_value=np.nan,
                )

            # peak shape
            self._y_grid = signal.convolve(self._apparatus_line(x_grid), aperture(x_grid, 0), mode='same') * 2*rx/len(x_grid)

        return self._y_grid

    def _get_intensity(self, number: Array, position: Number, concentration: float, show: bool = False, ylim: tuple[float, float] | None = None) -> Array[Percent]:
        config = self.config
        detector = config.detector

        L = config.concentration_ratio  # path length (concentration coefficient to agreement between theory and emulation)

        # number
        number = self.number

        # I
        I = L * concentration * (detector.config.width * detector.config.height)
        I *= (100/detector.config.capacity)  # to percent

        # intensity (detector's output signal)
        f = interpolate.interp1d(
            self.x_grid,
            self.y_grid,
            kind='linear',
            bounds_error=False,
            fill_value=0,
        )

        intensity = I*f((number - position)*detector.config.width)

        # background
        background = config.background_level

        intensity += background

        # show
        if show:
            plt.figure(figsize=(6, 4))

            x = self._x_grid

            if self._physical_line is not None:
                plt.plot(
                    x/detector.config.width + position, background + I*self._physical_line(x),
                    label=r'$I(\lambda)$',
                )
            plt.plot(
                x/detector.config.width + position, background + I*self._apparatus_line(x),
                label=r'$I^{F}(\lambda)$',
            )
            plt.fill_between(
                number,
                y1=background,
                y2=intensity,
                step='mid', facecolor=COLOR['pink'], edgecolor='k',
                label='$s_{k}$',
                alpha=0.2, 
            )

            if ylim is None:
                pass
            else:
                plt.ylim(ylim)

            plt.xlabel(r'number')
            plt.ylabel(r'$I$ $[\%]$')

            plt.grid(color='grey', linestyle=':')
            plt.legend()

            plt.show()

        # return intensity
        return intensity

    @property
    def intensity(self) -> Array[Percent]:
        if self._intensity is None:
            raise Exception('setup the emulation before!')

        return self._intensity

    # --------        handlers        --------
    def setup(self, position: Number | Sequence[Number], concentration: float, environment: Array[Percent] | None = None, show: bool = False, ylim: tuple[float, float] | None = None) -> 'EmulationInterface':
        """Setup emulation of emitted spectrum."""
        self.position = position
        self.concentration = concentration

        # setup intensity
        if isinstance(position, (int, float)):
            self._intensity = self._get_intensity(number=self.number, position=position, concentration=concentration, show=show, ylim=ylim)

        if isinstance(position, Sequence):
            self._intensity = np.array([
                self._get_intensity(number=self.number, position=x, concentration=concentration, show=show, ylim=ylim)
                for x in position
            ])

        # add spectral environment
        if isinstance(environment, Sequence):
            assert (environment.ndim == 1) and (environment.shape[-1] == self.config.spectrum.n_numbers)
            self._intensity += environment

        #
        return self    

    def run(self, is_noised: bool = True, is_clipped: bool = True, show: bool = False, random_state: int | None = None) -> EmittedSpectrum:
        """Run emulation."""
        config = self.config
        detector = config.detector

        # set random state
        if random_state is not None:
            np.random.seed(random_state)

        # init spectrum
        spectrum = emulate_emitted_spectrum(
            number=self.number,
            intensity=self.intensity,
            noise=self.noise,
            detector=detector,
            is_noised=is_noised,
            is_clipped=is_clipped,
        )

        # show spectrum
        if show:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), tight_layout=True)

            if spectrum.intensity.ndim == 1:
                y2 = spectrum.intensity
            else:
                y2 = spectrum.intensity[0]

            plt.fill_between(
                spectrum.number,
                y1=config.background_level,
                y2=y2,
                step='mid', alpha=0.2, facecolor=COLOR['pink'], edgecolor='k', label=f'paek',
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


# --------        HDR emission emulation        --------
@dataclass(frozen=True, slots=True)
class _HighDynamicRangeMode:
    total: MilliSecond  # total exposure time
    n_frames: tuple[int, ...]  # tuple of n_frames of the each exposure (tau)
    method: Literal['naive', 'weighted'] = 'weighted'
    base: int = 10  # base of the each exposure

    def __post_init__(self):
        assert self._validate(), f'{self} is not valid!'

    def _validate(self, tol=1e-9) -> bool:
        """Validate mode to equal total exposure time and expected."""
        total = sum([n_frames * tau for n_frames, tau in self.items()])

        return abs(total - self.total) <= tol

    def items(self) -> tuple[int, MilliSecond]:
        """Generate tuples of n_frames and tau."""

        for degree, n_frames in enumerate(self.n_frames):
            if n_frames > 0:
                tau = self.base ** (-degree)

                yield n_frames, tau


@dataclass(frozen=True, slots=True)
class HighDynamicRangeMode:
    total: MilliSecond  # total exposure time
    n_frames: tuple[int, ...]  # tuple of n_frames of the each exposure
    tau: tuple[MilliSecond, ...]  # tuple of exposures
    method: Literal['naive', 'weighted'] = 'weighted'

    # --------        handlers        --------
    def items(self) -> tuple[int, MilliSecond]:
        """Generate tuples of n_frames and tau."""

        for n_frames, tau in zip(self.n_frames, self.tau):
                yield n_frames, tau

    # --------        private        --------
    def __post_init__(self):
        assert self._validate(), f'{self} is not valid!'

    def _validate(self, tol=1e-9) -> bool:
        """Validate mode to equal total exposure time and expected."""
        total = sum([n_frames * tau for n_frames, tau in self.items()])

        return abs(total - self.total) <= tol


def emulate_hdr_emitted_spectrum(mode: HighDynamicRangeMode, number: Array, intensity: Array, detector: Detector, is_noised: bool = True, is_clipped: bool = True) -> HighDynamicRangeEmittedSpectrum:

    shorts = {}
    for n_frames, tau in mode.items():
        spe = emulate_emitted_spectrum(
            number=number,
            intensity=tau*intensity,
            noise=EmittedSpectrumNoise(
                detector=detector,
                n_frames=n_frames,
            ),
            detector=detector,
            is_noised=is_noised,
            is_clipped=is_clipped,
        )
        shorts[tau] = spe

    # 
    spectrum = HighDynamicRangeEmittedSpectrum(
        number=number,
        shorts=shorts,
        method=mode.method,
        detector=detector,
    )

    # return spectrum
    return spectrum


class HighDynamicRangeEmittedSpectrumEmulation(EmittedSpectrumEmulation):
    """High dynamic range (HDR) emitted spectrum emulation."""

    def __init__(self, config: EmittedSpectrumEmulationConfig, mode: HighDynamicRangeMode):
        super().__init__(config=config)

        self.mode = mode

    # --------        handlers        --------
    def run(self, is_noised: bool = True, is_clipped: bool = True, show: bool = False, random_state: int | None = None) -> HighDynamicRangeEmittedSpectrum:
        """Run emulation."""
        config = self.config
        detector = config.detector

        # set random state
        if random_state is not None:
            np.random.seed(random_state)

        # init spectrum
        spectrum = emulate_hdr_emitted_spectrum(
            mode=self.mode,
            number=self.number,
            intensity=self.intensity,
            detector=detector,
            is_noised=is_noised,
            is_clipped=is_clipped,
        )

        # show spectrum
        if show:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), tight_layout=True)

            if spectrum.intensity.ndim == 1:
                y2 = spectrum.intensity
            else:
                y2 = spectrum.intensity[0]
            plt.fill_between(
                spectrum.number,
                y1=config.background_level,
                y2=y2,
                step='mid', alpha=0.2, facecolor=COLOR['pink'], edgecolor='k', label=f'paek',
            )
            plt.xlim([spectrum.number.min()-1, spectrum.number.max()+1])

            plt.xlabel(r'number')
            plt.ylabel({
                EmittedSpectrum: r'$I$ [$\%$]',
                AbsorbedSpectrum: r'$A$',
            }.get(type(spectrum)))
            plt.grid(color='grey', linestyle=':')
            plt.show()

        # return spectrum
        return spectrum


# --------        absorption emulation        --------
def calculate_absorbance(level, base_level, scattering_ratio=0):
    scattering_level = scattering_ratio * base_level

    return np.log10(
        (base_level - scattering_level) / (level - scattering_level)  # # (base_level - scattering_level) / (level - scattering_level)
    )


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

    rx: Micro = field(default=100)  # границы построения интерполяции
    dx: Micro = field(default=.01)  # шаг сетки интерполяции


def emulate_absorbed_spectrum(number: Array, intensity: Array, noise: EmittedSpectrumNoise, base_level: float, base_noise: EmittedSpectrumNoise, detector: Detector, is_noised: bool = True, is_clipped: bool = True) -> AbsorbedSpectrum:

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


class AbsorbedSpectrumEmulation(EmulationInterface):
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
        x = np.arange(-span, span+dx, dx)

        apparatus_line = interpolate.interp1d(
            x,
            B0 + signal.convolve(apparatus(x, 0), physical_line(x) - B0, mode='same') * 2*span/len(x),
            kind='linear',
            bounds_error=False,
            fill_value=np.nan,
        )

        # intensity (detector's output signal)
        x0 = position*detector.config.width

        intensity = np.zeros(number.shape)
        for n in number:
            intensity[n] = integrate.quad(
                lambda x: apparatus_line(x - x0) * aperture(x, n),
                n*detector.config.width - rx,
                n*detector.config.width + rx,
            )[0]

        intensity += S0  # add scattering radiation
        intensity += I0 * (1 - integrate.quad(lambda x: aperture(x, 0), -rx, +rx)[0])  # approximated aperture characteristics correction

        # show
        if show:
            x = np.linspace(min(number)*detector.config.width, max(number)*detector.config.width, 1000)  # in Micro

            #
            plt.figure(figsize=(12, 4))

            # title = '\n'.join([
            #     f'dispersion: {device.config.dispersion:.4f}, nm/mm',
            #     f'detector: {detector.config.name}',
            # ])
            # plt.suptitle(title)

            # in emission units
            ax = plt.subplot(1, 2, 1)

            plt.plot(x/detector.config.width, S0 + physical_line(x - x0), label=r'$I(\lambda)$')  # f'physical line'
            plt.plot(x/detector.config.width, S0 + apparatus_line(x - x0), label=r'$I^{F}(\lambda)$')  # f'apparatus line'
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
            ax = plt.subplot(1, 2, 2)

            plt.plot(x/detector.config.width, calculate_absorbance(S0 + physical_line(x - x0), I0), label=r'$A(\lambda)$')  # f'physical line'
            plt.plot(x/detector.config.width, calculate_absorbance(S0 + apparatus_line(x - x0), I0), label=r'$A^{F}(\lambda)$')  # f'apparatus line'
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
    def setup(self, position: Number, concentration: float, show: bool = False, ylim: tuple[float, float] | None = None) -> 'EmulationInterface':
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
                step='mid', alpha=0.2, facecolor=COLOR['pink'], edgecolor='k', label=f'paek',
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


# --------        aliases        --------
EmulationConfig: TypeAlias = EmittedSpectrumEmulationConfig | AbsorbedSpectrumEmulationConfig
Emulation: TypeAlias = EmittedSpectrumEmulation | AbsorbedSpectrumEmulation


# --------        handlers        --------
@overload
def fetch_emulation(config: EmittedSpectrumEmulationConfig) -> EmittedSpectrumEmulation: ...
@overload
def fetch_emulation(config: EmittedSpectrumEmulationConfig, mode: HighDynamicRangeMode) -> HighDynamicRangeEmittedSpectrumEmulation: ...
@overload
def fetch_emulation(config: AbsorbedSpectrumEmulationConfig) -> AbsorbedSpectrumEmulation: ...
def fetch_emulation(config, mode=None):

    if isinstance(config, EmittedSpectrumEmulationConfig):
        if isinstance(mode, type(None)):
            return EmittedSpectrumEmulation(config=config)

        if isinstance(mode, HighDynamicRangeMode):
            return HighDynamicRangeEmittedSpectrumEmulation(config=config, mode=mode)

        raise TypeError()

    if isinstance(config, AbsorbedSpectrumEmulationConfig):
        return AbsorbedSpectrumEmulation(config=config)

    raise TypeError()


# --------        main        --------
if __name__ == '__main__':
    from spectrumlab.emulation.aperture import RectangularApertureShape
    from spectrumlab.emulation.apparatus import VoigtApparatusShape

    line_width = 1.68  # in pm

    # device
    device = Device.COLIBRI2
    dispersion = device.config.dispersion

    # detector
    detector = Detector.BLPP2000

    # emulation
    emulation = fetch_emulation(
        config=EmittedSpectrumEmulationConfig(
            device=device,
            detector=detector,

            line=None,
            apparatus=Apparatus(
                shape=VoigtApparatusShape(
                    width=25,
                    asymmetry=0,
                    ratio=0.1,
                ),
            ),
            aperture=Aperture(
                detector=detector,
                shape=RectangularApertureShape(),
            ),

            spectrum=SpectrumConfig(
                n_numbers=20,
            ),
            concentration_ratio=1,
            background_level=0,
        ),
    )
    emulation = emulation.setup(
        position=10,
        concentration=100,
        show=True,
    )

    # spectrum
    spectrum = emulation.run(
        random_state=42,
        show=True,
    )
