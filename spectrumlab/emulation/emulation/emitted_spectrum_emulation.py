from dataclasses import dataclass, field
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, signal

from spectrumlab.emulation.aperture import Aperture
from spectrumlab.emulation.apparatus import Apparatus
from spectrumlab.emulation.detector import Detector
from spectrumlab.emulation.device import Device
from spectrumlab.emulation.emulation import AbstractEmulation
from spectrumlab.emulation.line import Line
from spectrumlab.emulation.noise import EmittedSpectrumNoise
from spectrumlab.emulation.spectrum import EmittedSpectrum
from spectrumlab.picture.config import COLOR
from spectrumlab.typing import Array, MicroMeter, Number, Percent


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

    rx: MicroMeter = field(default=100)  # границы построения интерполяции
    dx: MicroMeter = field(default=.01)  # шаг сетки интерполяции


class EmittedSpectrumEmulation(AbstractEmulation):
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
    def x_grid(self) -> Array[MicroMeter]:
        if self._x_grid is None:
            config = self.config

            rx = config.rx
            dx = config.dx
            x_grid = np.linspace(-rx, +rx, 2*int(rx/dx) + 1)

            self._x_grid = x_grid

        return self._x_grid

    @property
    def y_grid(self) -> Array[Percent]:

        if self._y_grid is None:
            config = self.config
            detector = self.detector
            line = self.line
            apparatus = self.apparatus
            aperture = self.aperture

            x_grid = self.x_grid
            dx = config.dx

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
                    signal.convolve(self._physical_line(x_grid), apparatus(x_grid, 0), mode='same') * dx,
                    kind='linear',
                    bounds_error=False,
                    fill_value=np.nan,
                )

            # peak shape
            self._y_grid = convolve(
                x_grid/detector.pitch,
                self._apparatus_line,
                aperture,
                pitch=detector.pitch,
            )(x_grid/detector.pitch)

        return self._y_grid

    def _get_intensity(self, number: Array, position: Number, concentration: float, show: bool = False, ylim: tuple[float, float] | None = None) -> Array[Percent]:
        config = self.config
        detector = config.detector

        # number
        number = self.number

        # I
        I = concentration
        I *= config.concentration_ratio  # normalize to path length (concentration coefficient to agreement between theory and emulation)
        I *= (detector.pitch * detector.config.height)  # normalize to detector's square
        I *= (100/detector.config.capacity)  # normalize to detector's capacity

        # intensity (detector's output signal)
        f = interpolate.interp1d(
            self.x_grid,
            self.y_grid,
            kind='linear',
            bounds_error=False,
            fill_value=0,
        )

        intensity = I*f((number - position)*detector.pitch)

        # background
        background = config.background_level

        intensity += background

        # show
        if show:
            plt.figure(figsize=(6, 4))

            x = self._x_grid

            if self._physical_line is not None:
                plt.plot(
                    x/detector.pitch + position, detector.pitch*I*(background + self._physical_line(x)),
                    label=r'$I(\lambda)$',
                )
            plt.plot(
                x/detector.pitch + position, detector.pitch*I*(background + self._apparatus_line(x)),
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
    def setup(
            self,
            position: Number | Array[Number],
            concentration: float,
            environment: Array[Percent] | None = None,
            show: bool = False,
            ylim: tuple[float, float] | None = None,
            ) -> 'AbstractEmulation':
        """Setup emulation of emitted spectrum."""
        self.position = position
        self.concentration = concentration

        # setup intensity
        if isinstance(position, (int, float)):
            self._intensity = self._get_intensity(number=self.number, position=position, concentration=concentration, show=show, ylim=ylim)

        if isinstance(position, np.ndarray):
            self._intensity = np.sum([
                self._get_intensity(number=self.number, position=x, concentration=concentration, show=show, ylim=ylim)
                for x in position
            ], axis=0)

        # add spectral environment
        if isinstance(environment, np.ndarray):
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
                step='mid', alpha=0.2, facecolor=COLOR['pink'], edgecolor='k', label='paek',
            )

            plt.xlabel(r'number')
            plt.ylabel(r'$I$ [$\%$]')
            plt.grid(color='grey', linestyle=':')
            plt.show()

        # return spectrum
        return spectrum


# --------        handlers        --------
def convolve(
        x: Array[Number],
        apparatus: Callable[[Array[MicroMeter]], Array[float]],
        aperture: Callable[[Array[MicroMeter]], Array[float]],
        pitch: MicroMeter,
        ) -> Callable[[Array[Number]], Array[float]]:
    return interpolate.interp1d(
        x,
        signal.convolve(pitch*apparatus(x*pitch), pitch*aperture(x*pitch), mode='same') * (x[-1] - x[0])/(len(x) + 1),
        kind='linear',
        bounds_error=False,
        fill_value=0,
    )


def emulate_emitted_spectrum(
        number: Array[Number],
        intensity: Array[Percent],
        noise: EmittedSpectrumNoise,
        detector: Detector,
        is_noised: bool = True,
        is_clipped: bool = True,
        ) -> EmittedSpectrum:
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


if __name__ == '__main__':
    from spectrumlab.emulation.aperture import RectangularApertureShape
    from spectrumlab.emulation.apparatus import VoigtApparatusShape

    # device
    device = Device.COLIBRI2
    dispersion = device.config.dispersion

    # detector
    detector = Detector.BLPP2000

    # emulation
    emulation = EmittedSpectrumEmulation(
        config=EmittedSpectrumEmulationConfig(
            device=device,
            detector=detector,

            line=None,
            apparatus=Apparatus(
                detector=detector,
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
        position=np.array([10, 12]),
        concentration=1,
        show=True,
    )

    # spectrum
    spectrum = emulation.run(
        random_state=42,
        show=True,
    )
