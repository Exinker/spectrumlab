import os
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt

from spectrumlab.alias import Frame, Series
from spectrumlab.picture.config import COLOR, ALPHA

from .exceptions import FitError
from .metrology import Intercept, Slope, LOD, LOQ, LOL, DynamicRange, estimate_lol


class BaseCalibrationCurve(ABC):

    @abstractmethod
    def fit(self, intensity: Series, concentration: Series):
        pass
    
    @abstractmethod
    def predict(self, intensity: Series) -> Series:
        pass

    # --------        handlers        --------
    @abstractmethod
    def show(self, save: bool = False):
        """Show calibration curve."""
        pass

    def write(self):
        """Write calibration curve's data to file."""

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

    # --------        private        --------
    @abstractmethod
    def _get_filename(extension: Literal['png', 'txt']) -> str:
        pass

    def _get_color(self, mask: Series, color: str) -> list[str]:
        mapping = {
            True: 'grey',
            False: color,
        }

        return list(map(lambda x: mapping[x], mask))

    def _get_alpha(self, mask: Series, alpha: float) -> list[float]:
        mapping = {
            True: ALPHA['is_not_active'],
            False: alpha,
        }

        return list(map(lambda x: mapping[x], mask))


class CalibrationCurve(BaseCalibrationCurve):

    def __init__(self, data: Frame, blank: Frame | None = None):
        self._data = data
        self._blank = blank

        self._coeff = None
        self._lod = None
        self._loq = None
        self._lol = None
        self._dynamic_range = None

        # fit
        self.fit()

    @property
    def data(self) -> Frame:
        return self._data

    @property
    def blank(self) -> Frame:
        return self._blank

    @property
    def coeff(self) -> tuple[Intercept, Slope]:
        if self._coeff is None:
            raise FitError('fit the calibration curve before!')

        return self._coeff

    @property
    def lod(self) -> LOD:
        if self._lod is None:
            raise FitError('fit the calibration curve before!')

        return self._lod

    @property
    def loq(self) -> LOQ:
        if self._loq is None:
            raise FitError('fit the calibration curve before!')

        return self._loq

    @property
    def lol(self) -> LOL:
        if self._lol is None:
            raise FitError('fit the calibration curve before!')

        return self._lol

    @property
    def dynamic_range(self) -> DynamicRange:
        if self._dynamic_range is None:
            raise FitError('fit the calibration curve before!')

        return self._dynamic_range

    # --------        handlers        --------
    def fit(self):

        # fit
        mask = self.data['mask'].groupby(level=0, sort=False).max()

        data = self.data.copy()
        data = data[['concentration', 'intensity']]
        data = data.groupby(level=0, sort=False).mean()
        data = data[~mask]
        data = data.map(lambda x: np.log10(x))

        slope, intercept = np.polyfit(
            x=data['concentration'].values,
            y=data['intensity'].values,
            deg=1,
        )

        # coeff
        self._coeff = intercept, slope

        # limits
        blank_deviation = self._calculate_blank_deviation()

        self._lod = LOD.from_deviation(
            deviation=blank_deviation,
            coeff=self.coeff,
        )
        self._loq = LOQ.from_deviation(
            deviation=blank_deviation,
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

    def predict(self, intensity: Series) -> Series:
        interpect, slope = self.coeff

        return 10**((intensity.apply(lambda x: np.log10(x)) - interpect) / slope)

    def show(self, ref: Frame | None = None, save: bool = False):
        """Show calibration curve."""

        if ref is None:
            data = self.data.copy()
        else:
            data = ref.copy()
            data = data.set_index(['probe', 'parallel'])

        color = self._get_color(
            mask=data['mask'].groupby(level=0, sort=False).max().astype(bool),
            color=COLOR['yellow'] if ref is None else COLOR['green'],
        )
        alpha = self._get_alpha(
            mask=data['mask'].groupby(level=0, sort=False).max().astype(bool),
            alpha=ALPHA['probe'],
        )

        # show
        fig, (ax_left, ax_mid, ax_right) = plt.subplots(ncols=3, figsize=(15, 15/3), sharex=True, tight_layout=True,)

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
        plt.xlabel('$C$')
        plt.ylabel('$R$')
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
        plt.xlabel('$C$')
        plt.ylabel('$bias, \%$')
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
        plt.xlabel('$C$')
        plt.ylabel('$deviation, \%$')
        plt.grid(color='grey', linestyle=':')
        plt.legend()

        # save
        if save:
            filedir = os.path.join('.', 'img')
            if not os.path.isdir(filedir):
                os.mkdir(filedir)
            filename = self._get_filename(extension='png')

            plt.savefig(
                os.path.join(filedir, filename)
            )

        #
        plt.show()

    # --------        private        --------
    def _calculate_blank_deviation(self) -> float:

        if self.blank is None:
            print('blank: blank is not found!')  # FIXME: add exception!

            index = self.data.index[self.data['concentration'] == 0]
            if index.empty:
                print('blank: min concentration of data is not zero!')  # FIXME: add exception!
                index = self.data.index[self.data['concentration'] == min(self.data['concentration'])]
            intensity = self.data.loc[index, 'intensity'].values

        else:
            intensity = self.blank.loc[0, 'intensity'].values

        return np.std(intensity, ddof=1)

    def _get_filename(content: str, extension: Literal['png', 'txt']):

        return f'calibration_curve ({content}).{extension}'
