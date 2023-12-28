from collections.abc import Sequence, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt

from spectrumlab.alias import Array
from spectrumlab.spectrum import Spectrum


if TYPE_CHECKING:
    from spectrumlab.peak.blink_peak import BlinkPeak


class GridIterator:

    def __init__(self, xvalues: Array, yvalues: Array):
        self._xvalues = xvalues
        self._yvalues = yvalues
        self._index = -1

    def __iter__(self) -> Iterator:
        return self

    def __next__(self) -> tuple[float, float]:

        try:
            self._index += 1
            return self._xvalues[self._index], self._yvalues[self._index]

        except IndexError:
            raise StopIteration


@dataclass(frozen=True)
class Grid:
    xvalues: Array
    yvalues: Array

    def __post_init__(self):
        assert len(self.xvalues) == len(self.yvalues)

    @property
    def n_points(self) -> int:
        return len(self.xvalues)

    # --------        handlers        --------
    def show(self) -> None:
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

        x, y = self.xvalues, self.yvalues
        plt.plot(
            x, y,
            color='red', linestyle='none', marker='s', markersize=3,
            alpha=1,
        )

        plt.xlabel(r'$number$')
        plt.ylabel(r'$f$')
        plt.grid(color='grey', linestyle=':')

        plt.show()

    # --------        fabric        --------
    @classmethod
    def from_frames(cls, spectrum: Spectrum, offset: Array[float] | None = None, scale: Array[float] | None = None, background: Array[float] | None = None, threshold: float = 100) -> 'Grid':
        """Get a grid from frames of spectra (for example, series of shifted on wavelength)."""
        assert spectrum.n_times > 1, 'only kinetics spectra are supported!'

        #
        def _get_item(number: Array, intensity: Array, deviation: Array, clipped: Array, threshold: float) -> tuple[Array, Array]:
            is_clipped = clipped
            is_low_snr = intensity / deviation < threshold
            mask = ~is_clipped & ~is_low_snr

            return number[mask], intensity[mask]

        items = tuple(
            _get_item(spectrum.number, spectrum.intensity[t], spectrum.deviation[t], spectrum.clipped[t], threshold=threshold)
            for t in range(spectrum.n_times)
        )

        #
        return cls._from_items(
            items=items,
            offset=offset,
            scale=scale,
            background=background,
        )

    @classmethod
    def from_blinks(cls, spectrum: Spectrum, blinks: Sequence['BlinkPeak'], offset: Array[float] | None = None, scale: Array[float] | None = None, background: Array[float] | None = None) -> 'Grid':
        """Get a grid from sequence of blinks from spectrum."""
        assert spectrum.n_times == 1, 'kinetics spectra are not supported!'

        #
        def _get_item(number: Array, intensity: Array, blink: 'BlinkPeak') -> tuple[Array, Array]:
            lb, ub = blink.minima

            return number[lb:ub], intensity[lb:ub]

        items = tuple(
            _get_item(spectrum.number, spectrum.intensity, blink=blink)
            for blink in blinks
        )

        #
        return cls._from_items(
            items=items,
            offset=offset,
            scale=scale,
            background=background,
        )

    @classmethod
    def _from_items(cls, items: Sequence[tuple[Array, Array]], offset: Array[float] | None = None, scale: Array[float] | None = None, background: Array[float] | None = None) -> 'Grid':
        """Get a grid from sequence of items."""
        n_times = len(items)

        if offset is None:
            offset = tuple(0 for _ in range(n_times))
        assert len(offset) == n_times, f'len of offset have to be equal of n_times: {n_times}'

        if scale is None:
            scale = tuple(1 for _ in range(n_times))
        assert len(scale) == n_times, f'len of scale have to be equal of n_times: {n_times}'

        if background is None:
            background = np.zeros(n_times,)
        assert len(background) == n_times, f'len of background have to be equal of n_times: {n_times}'

        #
        xvalues, yvalues = [], []
        for t in range(n_times):
            x, y = items[t]

            xvalues.extend(x - offset[t])
            yvalues.extend((y - background[t]) / scale[t])
        xvalues, yvalues = np.array(xvalues), np.array(yvalues)

        index = np.argsort(xvalues)

        #
        return cls(
            xvalues=xvalues[index],
            yvalues=yvalues[index],
        )

    # --------        private        --------
    def __iter__(self) -> Iterator:
        return GridIterator(
            xvalues=self.xvalues,
            yvalues=self.yvalues,
        )

    def __repr__(self) -> str:
        cls = self.__class__

        return f'{cls.__name__}(n_points={self.n_points})'
