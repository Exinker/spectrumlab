
import os
from dataclasses import dataclass, field
from typing import Literal, NewType

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import interpolate

from spectrumlab.alias import Array, Frame, Number
from spectrumlab.picture.config import COLOR
from .metrology import Intercept, Slope, LOD, LOQ, LOL, DynamicRange, estimate_lol, estimate_dynamic_range


class CalibrationCurve:

    def __init__(self, data: Frame):  # , lol: LOL
        self._data = data
        self._coeff = None

        self._lod = None
        self._loq = None
        self._lol = None
        self._dynamic_range = None

    @property
    def data(self) -> Frame:
        return self._data

    def predict(self, intensity: Array[float]) -> Array[float]:
        interpect, slope = self.coeff

        return 10**((np.log10(intensity) - interpect) / slope)

    # --------        coeff        --------
    def _calculate_coeff(self) -> tuple[Intercept, Slope]:
        data = self.data.copy()
        data = data[~data['mask']][['concentration', 'intensity']]
        data = data.groupby(level=0, sort=False).mean()
        data = data.map(lambda x: np.log10(x))

        x = data['concentration'].values
        y = data['intensity'].values
        slope, intercept = np.polyfit(x, y, deg=1)

        return intercept, slope

    @property
    def coeff(self) -> tuple[Intercept, Slope]:
        if self._coeff is None:
            self._coeff = self._calculate_coeff()

        return self._coeff

    # --------        limit of detective (LOD)        --------
    @property
    def lod(self) -> LOD:
        if self._lod is None:
            self._lod = self._calculate_lod()

        return self._lod

    def _calculate_lod(self) -> 'LOD':
        data = self.data

        #
        index = data.index[data['concentration'] == 0]
        if index.empty:
            print('LOD: blank is not found!')  # FIXME: add exception!
            index = data.index[data['concentration'] == min(data['concentration'])]

        #
        return LOD.from_deviation(
            deviation=np.std(data.loc[index, 'intensity'].values, ddof=1),
            coeff=self.coeff,
        )

    # --------        limit of quantity (LOQ)        --------
    @property
    def loq(self) -> LOQ:
        if self._loq is None:
            self._loq = self._calculate_loq()

        return self._loq

    def _calculate_loq(self) -> 'LOQ':
        data = self.data

        #
        index = data.index[data['concentration'] == 0]
        if index.empty:
            print('LOQ: blank is not found!')  # FIXME: add exception!
            index = data.index[data['concentration'] == min(data['concentration'])]

        #
        return LOQ.from_deviation(
            deviation=np.std(data.loc[index, 'intensity'].values, ddof=1),
            coeff=self.coeff,
        )

    # --------        limit of linearity (LOL)        --------
    @property
    def lol(self) -> LOL:
        if self._lol is None:
            self._lol = estimate_lol(
                data=self.data,
                coeff=self.coeff,
            )

        return self._lol

    # --------        dynamic_range        --------
    @property
    def dynamic_range(self) -> DynamicRange:
        if self._dynamic_range is None:
            self._dynamic_range = self._calculate_dynamic_range()

        return self._dynamic_range

    def _calculate_dynamic_range(self) -> DynamicRange:
        low = self.loq.intensity
        high = self.lol.intensity

        return DynamicRange(
            intensity=(low, high),
            coeff=self.coeff,
        )

    # --------        handlers        --------
    def show(self, ref: Frame | None = None, concentration_ratio: float = 1, ylim: tuple[float, float] | None = None, save: bool = False):
        intercept, slope = self.coeff

        if ref is None:
            data = self.data.copy()
            data['concentration'] = data['concentration'].apply(lambda x: np.log10(x) + np.log10(concentration_ratio))
            data['intensity'] = data['intensity'].apply(lambda x: np.log10(x))
        else:
            data = ref.copy()
            data = data.set_index(['probe', 'parallel'])
            data['concentration'] = data['concentration'].apply(lambda x: np.log10(x))
            data['intensity'] = data['intensity'].apply(lambda x: np.log10(x))

        # show
        fig, (ax_left, ax_right) = plt.subplots(ncols=2, figsize=(10, 5), sharex=True, tight_layout=True,)

        title = '\n'.join([
            # fr'dispersion: {emulation.config.device.config.dispersion:.4f}, nm/mm',
            # fr'detector: {emulation.config.detector.config.name}',
            # fr'$\alpha: {emulation.config.scattering_ratio}$',
            # fr'$\beta: {emulation.config.background_level}$',
            # fr'dynamic_range: {self.dynamic_range.c_min:.4f} - {self.dynamic_range.c_max:.4f} ({np.log10(self.dynamic_range.c_max) - np.log10(self.dynamic_range.c_min):.4f})',
        ])
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
            alpha=.2,
        )

        x = data['concentration'].groupby(level=0, sort=False).mean()
        y = data['intensity'].groupby(level=0, sort=False).mean()
        plt.scatter(
            x, y,
            s=40,
            marker='s',
            facecolors=COLOR['yellow'] if ref is None else COLOR['green'],
            edgecolors=COLOR['yellow'] if ref is None else COLOR['green'],
            alpha=.5,
            label='emulated' if ref is None else 'recorded',
        )

        x = np.log10(self.lod.concentration)
        y = np.log10(self.lod.intensity)
        plt.scatter(
            x, y,
            s=40,
            marker='*',
            facecolors='none',
            edgecolors=COLOR['red'],
            label='LoD',
        )

        x = np.log10(self.dynamic_range.concentration)
        y = np.log10(self.dynamic_range.intensity)
        plt.plot(
            x, y,
            color='black',
            linestyle=':',
            alpha=.5,
        )

        if ylim:
            plt.ylim(ylim)
        plt.xlabel('$\log_{10}{C}$')
        plt.ylabel('$\log_{10}{I}$')
        plt.grid(color='grey', linestyle=':')
        plt.legend()

        #
        plt.sca(ax_right)

        predicted = self.predict(data['intensity'])
        x = data['concentration'].groupby(level=0, sort=False).mean()
        y = 100 * predicted.groupby(level=0, sort=False).std(ddof=1) / predicted.groupby(level=0, sort=False).mean()
        plt.scatter(
            x, y,
            s=40,
            marker='s',
            facecolors=COLOR['yellow'] if ref is None else COLOR['green'],
            edgecolors=COLOR['yellow'] if ref is None else COLOR['green'],
            alpha=.5,
            label='emulated' if ref is None else 'recorded',
        )

        plt.ylim([-.1, 10])
        plt.xlabel('$\log_{10}{C}$')
        plt.ylabel('$\sigma_{{C}}/C, \%$')
        plt.grid(color='grey', linestyle=':')
        plt.legend()

        # save
        if save:
            filedir = os.path.join('.', 'img')
            if not os.path.isdir(filedir):
                os.mkdir(filedir)
            filename = _get_filename(emulation, extension='png')

            plt.savefig(
                os.path.join(filedir, filename)
            )

        #
        plt.show()

    def write(self):
        emulation = self.emulation
        data = self.data

        #
        filedir = os.path.join('.', 'txt')
        if not os.path.isdir(filedir):
            os.mkdir(filedir)
        filename = _get_filename(emulation, extension='txt')
        filepath = os.path.join(filedir, filename)

        #
        data.to_csv(
            filepath,
            decimal=',',
            sep=';',
            encoding='utf-8',
            index=True,
            columns=data.columns,
        )
