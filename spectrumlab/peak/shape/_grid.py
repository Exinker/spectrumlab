from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from spectrumlab.alias import Array, Number
from spectrumlab.core.grid import Grid
from spectrumlab.spectrum import Spectrum


if TYPE_CHECKING:
    from spectrumlab.peak.blink_peak import BlinkPeak


class _Grid(Grid):
    
    def __init__(self, x: Array[Number], y: Array[float]):
        super().__init__(x=x, y=y)

    @property
    def n_points(self) -> int:
        return len(self.x)

    # --------        fabric        --------
    @classmethod
    def from_frames(cls, spectrum: Spectrum, offset: Array[Number] | None = None, scale: Array[float] | None = None, background: Array[float] | None = None, threshold: float = 0) -> '_Grid':
        """Get a grid from frames of spectra (for example, series of shifted on wavelength)."""
        # assert spectrum.n_times > 1, 'only kinetics spectra are supported!'

        #
        def _get_item(spectrum: Spectrum, t: int, threshold: float) -> tuple[Array, Array]:
            is_clipped = spectrum.clipped[t]
            is_snr_low = np.abs(spectrum.intensity[t]) / spectrum.deviation[t] < threshold
            mask = ~is_clipped & ~is_snr_low

            if spectrum.n_times == 1:
                return spectrum.number[mask], spectrum.intensity[mask]
            return spectrum.number[mask], spectrum.intensity[t, mask]

        items = tuple(
            _get_item(spectrum, t=t, threshold=threshold)
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
    def from_blinks(cls, spectrum: Spectrum, blinks: Sequence['BlinkPeak'], offset: Array[Number] | None = None, scale: Array[float] | None = None, background: Array[float] | None = None, threshold: float = 0) -> '_Grid':
        """Get a grid from sequence of blinks from spectrum."""
        assert spectrum.n_times == 1, 'kinetics spectra are not supported!'

        #
        def _get_item(spectrum: Spectrum, blink: 'BlinkPeak', threshold: float) -> tuple[Array, Array]:
            lb, ub = blink.minima

            is_clipped = spectrum.clipped[lb:ub]
            is_snr_low = np.abs(spectrum.intensity[lb:ub]) / spectrum.deviation[lb:ub] < threshold
            mask = ~is_clipped & ~is_snr_low

            return spectrum.number[lb:ub][mask], spectrum.intensity[lb:ub][mask]

        items = tuple(
            _get_item(spectrum, blink=blink, threshold=threshold)
            for blink in blinks
        )

        #
        return cls._from_items(
            items=items,
            offset=offset,
            scale=scale,
            background=background,
        )

    # --------        private        --------
    @classmethod
    def _from_items(cls, items: Sequence[tuple[Array, Array]], offset: Array[Number] | None = None, scale: Array[float] | None = None, background: Array[float] | None = None) -> '_Grid':
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
        _x, _y = [], []
        for t in range(n_times):
            x, y = items[t]

            _x.extend(x - offset[t])
            _y.extend((y - background[t]) / scale[t])
        _x, _y = np.array(_x).squeeze(), np.array(_y).squeeze()

        index = np.argsort(_x)

        #
        return cls(
            x=_x[index],
            y=_y[index],
        )
