from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from spectrumlab.calibrators.concentration_calibrators.concentration_calibrators import AbstractConcentrationCalibrator
from spectrumlab.grids import InterpolationKind
from spectrumlab.lines.line import Line
from spectrumlab.noises import Noise
from spectrumlab.peaks.analyte_peaks.intensity import (
    AbstractIntensityCalculator,
    ApproxIntensityCalculator,
    IntegralIntensityCalculator,
)
from spectrumlab.peaks.analyte_peaks.position import AbstractPositionCalculator, InterpolationPositionCalculator
from spectrumlab.peaks.blink_peaks.draft_blinks import DraftBlinksConfig, draft_blinks
from spectrumlab.peaks.peak import AbstractPeak
from spectrumlab.pictures.color import COLOR
from spectrumlab.spectra.spectrum import Spectrum
from spectrumlab.types import Array, Inch, NanoMeter, Number, R


@dataclass
class FactoryAnalytePeak:

    noise_level: float = field(default=5)
    position_calculator: AbstractPositionCalculator = field(default_factory=InterpolationPositionCalculator)
    intensity_calculator: AbstractIntensityCalculator = field(default_factory=IntegralIntensityCalculator)

    except_edges: bool = field(default=False)
    autocalculate: bool = field(default=False)
    # window: int | None = field(default=None)  # TODO: добавить размер вырезаемого участка спектра

    def create(
        self,
        line: Line,
        spectrum: Spectrum,
        verbose: bool = False,
        show: bool = False,
    ) -> 'AnalytePeak':
        assert spectrum.n_times == 1, 'time resolved spectra are not supported yet!'

        cursor = abs(spectrum.wavelength - line.wavelength).argmin()
        minima = [0, spectrum.n_numbers-1]
        maxima = []

        # mask
        blinks = draft_blinks(
            spectrum=spectrum,
            config=DraftBlinksConfig(
                except_clipped_peak=False,
                except_sloped_peak=False,
                except_edges=False,

                noise_level=self.noise_level,
            ),
        )

        is_found = False
        mask = np.full(spectrum.shape, True)
        for blink in blinks:
            left, right = blink.minima

            is_far = (np.abs(left - cursor) > 2) and (np.abs(right - cursor) > 2)  # расстояние от cursor до blink больше 2х отсчетов  # noqa: E501
            if is_far:
                if blink.include(cursor):  # cursor находится внутри этого blink (линия без самопоглощения)
                    is_found = True
                    maxima.append(blink.maxima[0])

                else:
                    mask[blink.number] = False

            else:
                if blink.amplitude > 100 * blink.deviation:  # этот blink достаточную амплитуду, чтобы считаться частью линии (линия с самопоглощением)  # noqa: E501
                    is_found = True
                    maxima.append(blink.maxima[0])

                else:
                    mask[blink.number] = False

        if not is_found:
            mask = np.full(spectrum.shape, True)

        # mask clipped counts
        mask[spectrum.clipped] = False

        # gather peak
        peak = AnalytePeak(
            minima=tuple(minima),
            maxima=tuple(maxima),

            spectrum=spectrum,
            mask=mask,
            config=AnalytePeakConfig(
                line=line,
                position_calculator=self.position_calculator,
                intensity_calculator=self.intensity_calculator,
            ),

            except_edges=self.except_edges,
            autocalculate=self.autocalculate,
        )

        if verbose:
            print(peak)

        if show:
            peak.show(verbose=True)

        return peak


@dataclass
class AnalytePeakConfig:
    line: Line

    position_calculator: AbstractPositionCalculator
    intensity_calculator: AbstractIntensityCalculator
    concentration_calibrator: AbstractConcentrationCalibrator | None = field(default=None)


class AnalytePeak(AbstractPeak):

    factory = FactoryAnalytePeak

    def __init__(
        self,
        minima: tuple[Number, Number],
        maxima: tuple[Number] | tuple[Number, Number] | tuple[Number, ...],
        spectrum: Spectrum,
        mask: Array,
        config: AnalytePeakConfig,
        except_edges: bool = False,
        autocalculate: bool = True,
    ) -> None:
        super().__init__(minima=minima, maxima=maxima, except_edges=except_edges)

        self.spectrum = spectrum
        self.config = config

        self._mask = mask
        self._cursor = None
        self._position = None
        self._intensity = None
        self._concentration = None

        if autocalculate:
            self.position
            self.intensity
            self.concentration

    @property
    def wavelength(self):
        return self.spectrum.wavelength[self.index]

    @property
    def value(self):
        return self.spectrum.intensity[self.index]

    @property
    def clipped(self):
        return self.spectrum.clipped[self.index]

    @property
    def mask(self):
        mask = self._mask

        return mask[self.index]

    # --------            position            --------
    @property
    def position(self) -> Number:
        if self._position is None:
            self._position = self.calculate_position()

        return self._position

    def calculate_position(self, calculator: AbstractPositionCalculator | None = None) -> Number:
        """Calculate peak's position."""
        calculator = calculator or self.config.position_calculator

        return calculator.calculate(self)

    # --------            intensity            --------
    @property
    def intensity(self) -> R:
        if self._intensity is None:
            self._intensity = self.calculate_intensity()

        return self._intensity

    def calculate_intensity(self, calculator: AbstractIntensityCalculator | None = None) -> R:
        """Calculate peak's intensity."""
        calculator = calculator or self.config.intensity_calculator

        return calculator.calculate(self)

    # --------            concentration            --------
    @property
    def concentration(self) -> float:
        if self._concentration is None:
            self._concentration = self.calculate_concentration()

        return self._concentration

    def calculate_concentration(self, calibrator: AbstractConcentrationCalibrator | None = None) -> float:
        """Calculate peak's concentration."""
        calibrator = calibrator or self.config.concentration_calibrator

        if calibrator is None:
            return np.nan

        return calibrator.predict(self.intensity)

    # --------            cursor            --------
    @property
    def cursor(self) -> Number:
        """Find peak's cursor (number of amount's maximum)."""

        if self._cursor is None:
            self._cursor = self.estimate_cursor()

        return self._cursor

    def estimate_cursor(self) -> Number:
        """Estimate peak's cursor."""
        line = self.config.line

        return abs(self.wavelength - line.wavelength).argmin()

    # --------            handlers            --------
    def transform(self, number: Number | Array[Number]) -> NanoMeter | Array[NanoMeter]:
        """Transform from a number to wavelength."""

        return interpolate.interp1d(
            self.number, self.wavelength,
            kind='linear',
            bounds_error=False,
            fill_value=0,
        )(number)

    def space(self, n_points: int = 1000, units: Number | NanoMeter = Number) -> Array:
        """Transform index or wavelength to space (n_points grid)."""
        left, right = self.minima
        if units == NanoMeter:
            left, right = self.wavelength[[left, right]]

        return np.linspace(left, right, n_points)

    def show(self, ax: plt.Axes | None = None, figsize: tuple[Inch, Inch] = (6, 4), verbose: bool = False) -> None:
        is_filling = ax is not None
        ylim = None

        if not is_filling:
            fig, ax = plt.subplots(figsize=figsize, tight_layout=True)

        calculator = self.config.intensity_calculator

        # draw cursor
        ax.axvline(
            self.transform(self.position),
            color='k',
            ls=':',
        )

        # draw peak
        if isinstance(calculator, IntegralIntensityCalculator):

            # draw peak
            x = self.wavelength
            y = self.value
            if calculator.kind == InterpolationKind.NEAREST:
                ax.step(
                    x, y,
                    where='mid',
                    linestyle='-', linewidth=1, color=COLOR.get('blue', '#000000'), alpha=.75,
                    marker='.', markersize=5,
                    label='$s_{k}$',
                )
            if calculator.kind == InterpolationKind.LINEAR:
                ax.plot(
                    x, y,
                    linestyle='-', linewidth=1, color=COLOR.get('blue', '#000000'), alpha=.75,
                    marker='.', markersize=5,
                    label='$s_{k}$',
                )

            # draw intensity
            x = self.transform(
                number=np.linspace(
                    self.position - calculator.interval/2,
                    self.position + calculator.interval/2,
                    round(2*calculator.interval*101),
                ),
            )
            f = interpolate.interp1d(
                self.wavelength, self.value,
                kind={
                    InterpolationKind.NEAREST: 'nearest',
                    InterpolationKind.LINEAR: 'linear',
                }.get(calculator.kind),
                bounds_error=False,
                fill_value=0,
            )
            ax.fill_between(
                x, f(x),
                step='mid',
                alpha=0.2, facecolor=COLOR.get('blue', '#000000'), edgecolor=COLOR.get('blue', '#000000'),
                label='область\nинтегр.',
            )

        if isinstance(calculator, ApproxIntensityCalculator):

            # draw peak
            x = self.wavelength
            y = self.value
            ax.plot(
                x, y,
                c=COLOR.get('blue', '#000000'),
                alpha=.25,
            )

            x = self.wavelength[self.mask]
            y = self.value[self.mask]
            ax.scatter(
                x, y,
                marker='s', s=10, facecolors=COLOR.get('red', '#000000'), edgecolors=COLOR.get('red', '#000000'),
                alpha=.75,
            )

            # draw intensity
            x = self.space(units=Number)
            x_hat = self.space(units=NanoMeter)
            y_hat = calculator.shape(x=x, **calculator.params)
            ax.plot(
                x_hat, y_hat,
                color='black', linestyle='-', linewidth=1,
            )

            if np.any(self.spectrum.clipped):
                ylim = [-10, 110]

            # x = self.index[self.mask]
            # y = self.value[self.mask]
            # x_hat = self.wavelength[self.mask]
            # y_hat = config.shape(x=x, **config.params)
            # ax.scatter(
            #     x_hat, y - y_hat,
            #     marker='s', s=5, facecolors='#000000', edgecolors='#000000',
            #     alpha=1,
            # )

            x = self.index[self.mask]
            y = self.value[self.mask]
            x_hat = self.wavelength[self.mask]
            y_hat = calculator.shape(x=x, **calculator.params)
            ax.plot(
                x_hat, y - y_hat,
                color='black', linestyle=':',
                alpha=.75,
            )

        if not is_filling:
            if ylim is not None:
                ax.set_ylim(ylim)

        if verbose:
            ax.text(
                .95, .95,
                f'I={self.intensity:.4f} %',
                color='black',
                transform=ax.transAxes, ha='right', va='top',
            )

        # set axes
        ax.set_xlabel(r'$\lambda$ [$nm$]')
        ax.set_ylabel(r'$I$ [$\%$]')

        ax.grid(color='grey', linestyle=':')

        if not is_filling:
            plt.show()

    def __repr__(self):
        cls = self.__class__

        content = '; '.join([
            f'{self.config.line.nickname}',
            f'intensity: {self.intensity:.4f}',
        ])
        return f'{cls.__name__}({content})'
