
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

from spectrumlab.alias import Array, Number, Percent
from spectrumlab.emulation.noise import Noise
from spectrumlab.line.line import Line
from spectrumlab.peak.base_peak import BasePeak
from spectrumlab.peak.blink_peak import BlinkPeak, DraftBlinkPeakConfig, draft_blinks
from spectrumlab.peak.concentration import ConcentrationConfig
from spectrumlab.peak.intensity import IntensityConfig, AmplitudeIntensityConfig, IntegralIntensityConfig, ApproxIntensityConfig, calculate_intensity
from spectrumlab.peak.position import PositionConfig, calculate_position
from spectrumlab.picture.config import COLOR
from spectrumlab.spectrum.mask import MaskConfig
from spectrumlab.spectrum.spectrum import Spectrum


# --------        analyte peak        --------
@dataclass
class AnalytePeakConfig:
    line: Line

    position: PositionConfig
    intensity: IntensityConfig
    concentration: ConcentrationConfig


class AnalytePeak(BasePeak):

    def __init__(self, minima: tuple[int, int], maxima: tuple | tuple[int, int] | tuple[int, ...], spectrum: Spectrum, mask: Array, config: AnalytePeakConfig, except_edges: bool = False, autocalculate: bool = True):
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

    def __repr__(self):
        cls = self.__class__

        content = '; '.join([
            f'{self.config.line.nickname}',
            f'intensity: {self.intensity:.4f} %',
        ])
        return f'{cls.__name__}({content})'

    # --------            position            --------
    def calculate_position(self, config: PositionConfig | None = None) -> Number:
        """Estimate peak's position."""
        config = self.config.position if config is None else config

        return calculate_position(self, config=config)

    @property
    def position(self) -> Number:
        if self._position is None:
            self._position = self.calculate_position()

        return self._position

    # --------            intensity            --------
    def calculate_intensity(self, config: IntensityConfig | None = None) -> float:
        """Estimate peak's intensity."""
        config = self.config.intensity if config is None else config

        return calculate_intensity(self, config=config)

    @property
    def intensity(self) -> float:
        if self._intensity is None:
            self._intensity = self.calculate_intensity()

        return self._intensity

    # --------            concentration            --------
    def estimate_concentration(self, config: ConcentrationConfig | None = None) -> float:
        # raise NotImplementedError

        return 0

    @property
    def concentration(self) -> float:
        if self._concentration is None:
            self._concentration = self.estimate_concentration()

        return self._concentration

    # --------            cursor            --------
    def estimate_cursor(self) -> Number:
        """Find peak's cursor."""
        line = self.config.line

        return abs(self.wavelength - line.wavelength).argmin()

    @property
    def cursor(self) -> Number:
        """Find peak's cursor (number of amount's maximum)."""

        if self._cursor is None:
            self._cursor = self.estimate_cursor()

        return self._cursor

    # --------            other            --------
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

    def transform(self, number: float | Array) -> float | Array:
        """transform from a number to wavelength"""

        return interpolate.interp1d(
            self.number, self.wavelength,
            kind='linear',
            bounds_error=False,
            fill_value=0,
        )(number)

    def space(self, kind: Literal['index', 'wavelength'] = 'wavelength', n_points: int = 1000) -> Array:
        left, right = self.minima

        if kind == 'wavelength':
            left, right = self.wavelength[[left, right]]

        return np.linspace(left, right, n_points)

    # --------            show            --------
    def show(self, ax: plt.Axes | None = None, figsize: tuple[float, float] = (6, 4), verbose: bool = False) -> None:
        is_filling = ax is not None

        if not is_filling:
            fig, ax = plt.subplots(figsize=figsize, tight_layout=True)

        config = self.config.intensity

        # draw cursor
        ax.axvline(
            self.transform(self.position),
            color='k',
            ls=':',
        )
        
        # draw peak
        if isinstance(config, IntegralIntensityConfig):

            # draw peak
            x = self.wavelength
            y = self.value
            if config.method == 'nearest':
                ax.step(
                    x, y,
                    where='mid',
                    linestyle='-', linewidth=1, color=COLOR.get('blue', '#000000'), alpha=.75,
                    marker='.', markersize=5,
                    label='$s_{k}$',
                )
            if config.method == 'linear':
                ax.plot(
                    x, y,
                    linestyle='-', linewidth=1, color=COLOR.get('blue', '#000000'), alpha=.75,
                    marker='.', markersize=5,
                    label='$s_{k}$',
                )

            # draw intensity
            x = self.transform(
                np.linspace(self.position - config.interval/2, self.position + config.interval/2, round(2*config.interval*101))
            )
            f = interpolate.interp1d(
                self.wavelength, self.value,
                kind=config.method,
                bounds_error=False,
                fill_value=0,
            )
            ax.fill_between(
                x, f(x),
                step='mid',
                alpha=0.2, facecolor=COLOR.get('blue', '#000000'), edgecolor=COLOR.get('blue', '#000000'),
                label='область\nинтегр.',
            )

        if isinstance(config, ApproxIntensityConfig):

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
            x = self.space(kind='index')
            x_hat = self.space(kind='wavelength')
            y_hat = config.approx_profile(x=x, **config.approx_params)
            ax.plot(
                x_hat, y_hat,
                color='red',
            )

            # x = self.index[self.mask]
            # y = self.value[self.mask]
            # x_hat = self.wavelength[self.mask]
            # y_hat = config.approx_profile(x=x, **config.approx_params)
            # ax.scatter(
            #     x_hat, y - y_hat,
            #     marker='s', s=5, facecolors='#000000', edgecolors='#000000',
            #     alpha=1,
            # )

            x = self.index[self.mask]
            y = self.value[self.mask]
            x_hat = self.wavelength[self.mask]
            y_hat = config.approx_profile(x=x, **config.approx_params)
            ax.plot(
                x_hat, y - y_hat,
                color='black', linestyle=':',
                alpha=.75,
            )

        # verbose
        if verbose:
            ax.text(
                .95, .95,
                f'I={self.intensity:.4f} %',
                color='black',
                transform=ax.transAxes, ha='right', va='top',
            )

        # set axes
        ax.set_xlabel(r'$\lambda, nm$')
        ax.set_ylabel(r'$I, \%$')

        ax.grid(color='grey', linestyle=':')

        if not is_filling:
            plt.show()


# --------        gather analyte peak        --------
@dataclass
class GatherAnalytePeakConfig:
    line: Line

    maskConfig: MaskConfig
    position: PositionConfig
    intensity: IntensityConfig
    concentration: ConcentrationConfig

    except_edges: bool | None = field(default=False)
    autocalculate: bool | None = field(default=False)
    # window: int | None = field(default=None)  # FIXME: добавить размер вырезаемого участка спектра


def gather_analyte_peak(spectrum: Spectrum, noise: Noise, config: GatherAnalytePeakConfig, show: bool = False, verbose: bool = False) -> AnalytePeak:
    """Interface to gather a peak with selected config.

    Author: Vaschenko Pavel
     Email: vaschenko@vmk.ru
      Date: 2016.04.09
    """

    assert spectrum.n_times == 1, 'kinetics spectra are not supported yet!'

    # blinks
    blinks = draft_blinks(
        spectrum=spectrum,
        noise=noise,
        config=DraftBlinkPeakConfig(
            except_clipped_peak=False,
            except_sloped_peak=False,
            except_edges=False,

            noise_level=config.maskConfig.noise_level,
        ),
    )

    # mask overlapped blinks
    minima = [0, spectrum.n_numbers-1]
    maxima = []
    mask = np.full(spectrum.shape, True)

    cursor = abs(spectrum.wavelength - config.line.wavelength).argmin()

    for blink in blinks:
        left, right = blink.minima

        # попадает ли cursor между blink.minima
        distance = map(abs, (left - cursor, right - cursor))

        condition = all(map(lambda x: x > 2, distance))
        if condition:  # если расстояние от линии до blink больше 2х отсчетов
            if blink.include(cursor):  # линия находится внутри этого blink (линия без самопоглощения)
                maxima.append(blink.maxima[0])

            else:
                mask[blink.number] = False  # маскируем мешающий blink

        else:  # если расстояние меньше 2х отсчетов, то нужно проверить не шум ли это
            if blink.amplitude > 5 * blink.deviation:  # этот blink достаточную амплитуду, чтобы считаться частью линии (линия с самопоглощением)
                maxima.append(blink.maxima[0])

            else:
                mask[blink.number] = False  # маскируем мешающий blink

    # gather peak
    peak = AnalytePeak(
        minima=tuple(minima),
        maxima=tuple(maxima),

        spectrum=spectrum,
        mask=mask,
        config=AnalytePeakConfig(
            line=config.line,
            position=config.position,
            intensity=config.intensity,
            concentration=config.concentration,
        ),

        except_edges=config.except_edges,
        autocalculate=config.autocalculate,
    )

    # verbose
    if verbose:
        print(peak)

    # show
    if show:
        peak.show(verbose=True)

    # 
    return peak
