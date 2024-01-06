
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import matplotlib.pyplot as plt

from spectrumlab.alias import NanoMeter
from .characteristic import WindowCharacteristic, DatasheetCharacteristic


@dataclass
class Filter(ABC):
    """Interface for any filter"""

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError

    # --------        handlers        --------
    @abstractmethod
    def show(self, ax: plt.Axes | None = None) -> None:
        raise NotImplementedError


@dataclass
class WindowFilter(Filter, WindowCharacteristic):
    span: tuple[NanoMeter, NanoMeter]
    smooth: float  # smoothing rectangular edges by gauss
    wavelength_bounds: tuple[float, float]
    wavelength_step: float

    def __str__(self) -> str:
        return '{}, нм'.format('-'.join(map(str, self.span)))

    # --------        handlers        --------
    def show(self, info: Literal['title', 'text', 'none'] = 'text', save: bool = False, ax: plt.Axes | None = None) -> None:
        fill = ax is not None

        if not fill:
            fig, ax = plt.subplots(ncols=1, figsize=(6, 4), tight_layout=True)

        x, y = 1e+9 * self._x, self._y
        ax.plot(
            x, y,
            color='black',
        )
        text = 'Filter: {}, нм'.format('-'.join(map(str, self.span)))
        match info:
            case 'title':
                ax.set_title(text)
            case 'text': 
                ax.text(
                    0.05, 0.95,
                    text,
                    transform=ax.transAxes,
                    ha='left', va='top',
                )
            case 'none':
                pass

        xlim = (x[0], x[-1])
        ax.set_xlim(xlim)
        ax.set_ylim([0, 1.25])
        
        ax.set_xlabel('$\lambda, нм$')
        ax.set_ylabel(r'Коэффициент пропускания')
        
        ax.grid(color='grey', linestyle=':')

        if save:
            filepath = os.path.join('.', 'report', 'img', 'filter-response.png')
            plt.savefig(filepath)

        if not fill:
            plt.show()


@dataclass
class DatasheetFilter(Filter, DatasheetCharacteristic):
    path: str
    xscale: float  # transform to meter units
    norm: float = field(default=1)  # normalization scale

    def __str__(self) -> str:
        head, tail = os.path.split(self.path)
        return 'file: {}'.format(head)

    # --------        handlers        --------
    def show(self, ax: plt.Axes | None = None) -> None:
        raise NotImplementedError

