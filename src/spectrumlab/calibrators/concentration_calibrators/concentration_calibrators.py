import os
from abc import ABC, abstractmethod
from typing import Callable, Literal, Self

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectrumlab.calibrators.concentration_calibrators.exceptions import ConcentrationCalibratorError
from spectrumlab.calibrators.concentration_calibrators.metrology import DynamicRange, LOD, LOL, LOQ, estimate_lol
from spectrumlab.picture.alphas import ALPHA, Alpha
from spectrumlab.picture.colors import COLOR, Color
from spectrumlab.spectra import Spectrum
from spectrumlab.types import Frame, Intercept, R, Series, Slope


class ConcentrationCalibratorABC(ABC):

    @property
    @abstractmethod
    def coeff(self) -> tuple[Intercept, Slope]:
        raise NotImplementedError

    @abstractmethod
    def fit(self, intensity: Series, concentration: Series) -> Self:
        raise NotImplementedError

    @abstractmethod
    def predict(self, intensity: Series) -> Series:
        raise NotImplementedError

    @abstractmethod
    def show(self, save: bool = False):
        """Show concentration calibration."""
        raise NotImplementedError

    def write(self):
        """Write concentration calibration's data to file."""

        filedir = os.path.join('.', 'txt')
        if not os.path.isdir(filedir):
            os.mkdir(filedir)
        filename = self._get_filename(extension='txt')
        filepath = os.path.join(filedir, filename)

        #
        data = self.data
        data.to_csv(
            filepath,
            decimal=',',
            sep=';',
            encoding='utf-8',
            index=True,
            columns=data.columns,
        )

    @abstractmethod
    def _get_filename(self, extension: Literal['png', 'txt']) -> str:
        raise NotImplementedError

    def _get_color(self, mask: Series, color: Color) -> list[Color]:
        mapping = {
            True: 'grey',
            False: color,
        }

        return list(map(lambda x: mapping[x], mask))

    def _get_alpha(self, mask: Series, alpha: Alpha) -> list[Alpha]:
        mapping = {
            True: ALPHA['is_not_active'],
            False: alpha,
        }

        return list(map(lambda x: mapping[x], mask))


class RegressionConcentrationCalibrator(ConcentrationCalibratorABC):

    def __init__(self) -> None:

        self._data = None
        self._blank = None
        self._coeff = None
        self._lod = None
        self._loq = None
        self._lol = None
        self._dynamic_range = None

    @property
    def data(self) -> Frame:
        return self._data

    @property
    def blank(self) -> Frame:
        if isinstance(self._blank, Frame):
            return self._blank

        #
        print('blank: blank is not found!')  # FIXME: add exception!

        index = self.data.index[self.data['concentration'] == 0]
        if index.empty:
            print('blank: min concentration of data is not zero!')  # FIXME: add exception!
            index = self.data.index[self.data['concentration'] == min(self.data['concentration'])]

        return self.data.loc[index]

    @property
    def coeff(self) -> tuple[Intercept, Slope]:
        if self._coeff is None:
            raise ConcentrationCalibratorError('Fit the concentration concentrator calibration before')

        return self._coeff

    @property
    def lod(self) -> LOD:
        if self._lod is None:
            raise ConcentrationCalibratorError('Fit the concentration concentrator calibration before!')

        return self._lod

    @property
    def loq(self) -> LOQ:
        if self._loq is None:
            raise ConcentrationCalibratorError('Fit the concentration concentrator calibration before!')

        return self._loq

    @property
    def lol(self) -> LOL:
        if self._lol is None:
            raise ConcentrationCalibratorError('Fit the concentration concentrator calibration before!')

        return self._lol

    @property
    def dynamic_range(self) -> DynamicRange:
        if self._dynamic_range is None:
            raise ConcentrationCalibratorError('Fit the concentration concentrator calibration before!')

        return self._dynamic_range

    def fit(self, data: Frame, blank: Frame | None = None) -> Self:

        self._data = data
        self._blank = blank

        #
        data = data.copy()
        mask = data['mask'].groupby(level=0, sort=False).max()
        data = data[['concentration', 'intensity']]
        data = data.groupby(level=0, sort=False).mean()
        data = data[~mask]
        data = data.map(lambda x: np.log10(x))
        slope, intercept = np.polyfit(
            x=data['concentration'].values,
            y=data['intensity'].values,
            deg=1,
        )

        self._coeff = intercept, slope
        self._lod = LOD.from_blank(
            data=self.blank,
            coeff=self.coeff,
        )
        self._loq = LOQ.from_blank(
            data=self.blank,
            coeff=self.coeff,
        )
        self._lol = estimate_lol(
            data=self.data,
            coeff=self.coeff,
        )
        self._dynamic_range = DynamicRange(
            intensity=(self.loq.intensity, self.lol.intensity),
            coeff=self.coeff,
        )

        return self

    def predict(self, intensity: Series) -> Series:
        interpect, slope = self.coeff

        return 10**((intensity.apply(lambda x: np.log10(x)) - interpect) / slope)

    def show(self, ref: Frame | None = None, save: bool = False):
        """Show concentration calibration."""

        if ref is None:
            data = self.data.copy()
        else:
            data = ref.copy()

        color = self._get_color(
            mask=data['mask'].groupby(level=0, sort=False).max().astype(bool),
            color=COLOR['yellow'] if ref is None else COLOR['green'],
        )
        alpha = self._get_alpha(
            mask=data['mask'].groupby(level=0, sort=False).max().astype(bool),
            alpha=ALPHA['probe'],
        )

        fig, (ax_left, ax_mid, ax_right) = plt.subplots(ncols=3, figsize=(15, 15/3), sharex=True, tight_layout=True)

        title = ''
        plt.suptitle(title)  # TODO: add title's config

        #
        plt.sca(ax_left)

        x = data['concentration']
        y = data['intensity']
        plt.scatter(
            x, y,
            s=20,
            marker='s',
            facecolors='none',
            edgecolors=[0, 0, 0, 0],
            alpha=ALPHA['parallel'],
        )

        x = data['concentration'].groupby(level=0, sort=False).mean()
        y = data['intensity'].groupby(level=0, sort=False).mean()
        plt.scatter(
            x, y,
            s=40,
            marker='s',
            facecolors=color,
            edgecolors=color,
            alpha=alpha,
            label='emulated' if ref is None else 'recorded',
        )

        x = self.lod.concentration
        y = self.lod.intensity
        plt.scatter(
            x, y,
            s=40,
            marker='*',
            facecolors='none',
            edgecolors=COLOR['red'],
            label='LoD',
        )

        x = self.dynamic_range.concentration
        y = self.dynamic_range.intensity
        plt.plot(
            x, y,
            color='black',
            linestyle=':',
            alpha=ALPHA['default'],
        )

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$C$')
        plt.ylabel(r'$R$')
        plt.grid(color='grey', linestyle=':')
        plt.legend()

        #
        plt.sca(ax_mid)

        c_true = data['concentration'].groupby(level=0, sort=False).mean()
        c_hat = self.predict(data['intensity'].groupby(level=0, sort=False).mean())
        bias = 100*(c_hat - c_true)/c_true
        x = c_true
        y = bias
        plt.scatter(
            x, y,
            s=20,
            marker='s',
            facecolors=color,
            edgecolors=color,
            alpha=alpha,
            label='emulated' if ref is None else 'recorded',
        )

        # lim = 1.1*np.max(np.abs(bias))
        # if lim > 50: lim = 50  # restrict lim
        # if lim < 20: lim = 20  # restrict lim
        # plt.ylim(-lim, +lim)
        plt.xscale('log')
        plt.xlabel(r'$C$')
        plt.ylabel(r'$bias, \%$')
        plt.grid(color='grey', linestyle=':')
        plt.legend()

        #
        plt.sca(ax_right)

        c_true = data['concentration'].groupby(level=0, sort=False).mean()
        c_hat = self.predict(data['intensity'])
        deviation = 100 * c_hat.groupby(level=0, sort=False).std(ddof=1) / c_true
        x = c_true
        y = deviation
        plt.scatter(
            x, y,
            s=40,
            marker='s',
            facecolors=color,
            edgecolors=color,
            alpha=alpha,
            label='emulated' if ref is None else 'recorded',
        )

        # plt.ylim([-.1, 2])
        plt.xscale('log')
        plt.xlabel(r'$C$')
        plt.ylabel(r'$deviation, \%$')
        plt.grid(color='grey', linestyle=':')
        plt.legend()

        # save
        if save:
            filedir = os.path.join('.', 'img')
            if not os.path.isdir(filedir):
                os.mkdir(filedir)
            filename = self._get_filename(extension='png')
            filepath = os.path.join(filedir, filename)

            plt.savefig(filepath)

        #
        plt.show()

    def _get_filename(self, content: str, extension: Literal['png', 'txt']):

        return f'concentration_calibration ({content}).{extension}'


def calibrate(
    spectra: Frame,
    handler: Callable[[Spectrum], R],
    show: bool = False,
    calibrator: ConcentrationCalibratorABC | None = None,
) -> ConcentrationCalibratorABC:

    # blank
    index = spectra[spectra.index.get_level_values(0) == 'blank'].index
    blank = pd.DataFrame(
        data={'concentration': None, 'intensity': None, 'mask': False},
        columns=['concentration', 'intensity', 'mask'],
        index=index,
    )
    for i, j in index:
        spectrum = spectra.loc[(i, j), 'spectrum']
        concentration = spectra.loc[(i, j), 'concentration']

        intensity = handler(spectrum=spectrum)

        blank.loc[(i, j), 'concentration'] = concentration
        blank.loc[(i, j), 'intensity'] = intensity
        blank.loc[(i, j), 'mask'] = any(spectrum.clipped)
    values = blank['intensity'].values.astype(float)
    loq = LOQ.calculate(
        mean=np.nanmean(values),
        deviation=np.nanstd(values, ddof=1),
        k=10,
    )

    # data
    index = spectra.drop(index='blank').index
    data = pd.DataFrame(
        data={'concentration': None, 'intensity': None, 'mask': False},
        columns=['concentration', 'intensity', 'mask'],
        index=index,
    )
    for i, j in index:
        spectrum = spectra.loc[(i, j), 'spectrum']
        concentration = spectra.loc[(i, j), 'concentration']

        intensity = handler(spectrum=spectrum)

        data.loc[(i, j), 'concentration'] = concentration
        data.loc[(i, j), 'intensity'] = intensity

        is_traced = data.loc[(i, j), 'intensity'] < loq
        is_clipped = any(spectrum.clipped)
        data.loc[(i, j), 'mask'] = is_traced or is_clipped

    # calibrator
    calibrator = calibrator or RegressionConcentrationCalibrator()
    calibrator = calibrator.fit(
        data=data,
        blank=blank,
    )

    if show:
        calibrator.show()

    return calibrator
