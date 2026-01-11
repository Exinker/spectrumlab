from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from spectrumlab.calibrators.concentration_calibrators.concentration_calibrators import (
    ConcentrationCalibratorABC,
)
from spectrumlab.grids import InterpolationKind
from spectrumlab.lines.line import Line
from spectrumlab.peaks.analyte_peaks.intensity.estimators import (
    ApproxIntensityEstimator,
    IntegralIntensityEstimator,
    IntensityEstimator,
)
from spectrumlab.peaks.analyte_peaks.position.estimators import (
    InterpolationPositionEstimator,
    PositionEstimator,
)
from spectrumlab.peaks.base_peak import PeakABC
from spectrumlab.peaks.blink_peaks.draft_blinks import DraftBlinksConfig, draft_blinks
from spectrumlab.picture.colors import COLOR
from spectrumlab.spectra import Spectrum
from spectrumlab.types import Array, Inch, NanoMeter, Number, R


@dataclass(frozen=True, slots=True)
class AnalytePeakFactory:

    noise_level: float = field(default=5)
    position_estimator: PositionEstimator = field(default_factory=InterpolationPositionEstimator)
    intensity_estimator: IntensityEstimator = field(default_factory=IntegralIntensityEstimator)

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
            config=DraftBlinksConfig.model_construct(
                except_clipped_peak=False,
                except_sloped_peak=False,
                except_wide_peak=False,
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
                if blink.include(cursor):  # cursor находится внутри blink (линия без самопоглощения)
                    is_found = True
                    maxima.append(blink.maxima[0])
                else:
                    mask[blink.number] = False

            else:
                # if blink.amplitude > 100 * blink.deviation:  # этот blink достаточную амплитуду, чтобы считаться частью линии (линия с самопоглощением)  # noqa: E501
                #     is_found = True
                #     maxima.append(blink.maxima[0])
                # else:
                #     mask[blink.number] = False
                is_found = True
                maxima.append(blink.maxima[0])

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
                position_estimator=self.position_estimator,
                intensity_estimator=self.intensity_estimator,
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

    position_estimator: PositionEstimator
    intensity_estimator: IntensityEstimator
    concentration_calibrator: ConcentrationCalibratorABC | None = field(default=None)


class AnalytePeak(PeakABC):

    factory = AnalytePeakFactory

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

    @property
    def position(self) -> Number:
        if self._position is None:
            self._position = self.calculate_position()

        return self._position

    @property
    def intensity(self) -> R:
        if self._intensity is None:
            self._intensity = self.calculate_intensity()

        return self._intensity

    @property
    def concentration(self) -> float:
        if self._concentration is None:
            self._concentration = self.calculate_concentration()

        return self._concentration

    @property
    def cursor(self) -> Number:
        """Find peak's cursor (number of amount's maximum)."""

        if self._cursor is None:
            self._cursor = self.estimate_cursor()

        return self._cursor

    def calculate_position(self, estimator: PositionEstimator | None = None) -> Number:
        """Calculate peak's position."""
        estimator = estimator or self.config.position_estimator

        return estimator.calculate(self)

    def calculate_intensity(self, estimator: IntensityEstimator | None = None) -> R:
        """Calculate peak's intensity."""
        estimator = estimator or self.config.intensity_estimator

        return estimator.calculate(self)

    def calculate_concentration(self, calibrator: ConcentrationCalibratorABC | None = None) -> float:
        """Calculate peak's concentration."""
        calibrator = calibrator or self.config.concentration_calibrator

        if calibrator is None:
            return np.nan

        return calibrator.predict(self.intensity)

    def estimate_cursor(self) -> Number:
        """Estimate peak's cursor."""
        line = self.config.line

        return abs(self.wavelength - line.wavelength).argmin()

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

        estimator = self.config.intensity_estimator

        # draw cursor
        ax.axvline(
            self.transform(self.position),
            color='k',
            ls=':',
        )

        # draw peak
        if isinstance(estimator, IntegralIntensityEstimator):

            # draw peak
            x = self.wavelength
            y = self.value
            if estimator.kind == InterpolationKind.NEAREST:
                ax.step(
                    x, y,
                    where='mid',
                    linestyle='-', linewidth=1, color=COLOR.get('blue', '#000000'), alpha=.75,
                    marker='.', markersize=5,
                    label='$s_{k}$',
                )
            if estimator.kind == InterpolationKind.LINEAR:
                ax.plot(
                    x, y,
                    linestyle='-', linewidth=1, color=COLOR.get('blue', '#000000'), alpha=.75,
                    marker='.', markersize=5,
                    label='$s_{k}$',
                )

            # draw intensity
            x = self.transform(
                number=np.linspace(
                    self.position - estimator.interval/2,
                    self.position + estimator.interval/2,
                    round(2*estimator.interval*101),
                ),
            )
            f = interpolate.interp1d(
                self.wavelength, self.value,
                kind={
                    InterpolationKind.NEAREST: 'nearest',
                    InterpolationKind.LINEAR: 'linear',
                }.get(estimator.kind),
                bounds_error=False,
                fill_value=0,
            )
            ax.fill_between(
                x, f(x),
                step='mid',
                alpha=0.2, facecolor=COLOR.get('blue', '#000000'), edgecolor=COLOR.get('blue', '#000000'),
                label='область\nинтегр.',
            )

        if isinstance(estimator, ApproxIntensityEstimator):

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
            y_hat = estimator.shape(x=x, **estimator.params)
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
            y_hat = estimator.shape(x=x, **estimator.params)
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
