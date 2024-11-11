from collections.abc import Iterator, Sequence
from typing import Callable, TYPE_CHECKING, TypeVar
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, interpolate

from spectrumlab.spectrum import Spectrum
from spectrumlab.types import Absorbance, Array, Digit, Electron, MicroMeter, NanoMeter, Number, Percent, PicoMeter

if TYPE_CHECKING:
    from spectrumlab.peak.blink_peak import BlinkPeak


U = TypeVar('U', Digit, Electron, Percent, Absorbance)
T = TypeVar('T', Number, MicroMeter, NanoMeter, PicoMeter)


class IteratorGrid:

    def __init__(self, x: Array[T], y: Array[float]):
        self.x = x
        self.y = y

        self._index = -1

    def __iter__(self) -> Iterator:
        return self

    def __next__(self) -> tuple[float, float]:

        try:
            self._index += 1
            return self.x[self._index], self.y[self._index]

        except IndexError:
            raise StopIteration


# --------        grid        --------
class _FactoryBatch:

    def __init__(self, spectrum: Spectrum):
        self.spectrum = spectrum

    def create_from_blink(self, blink: 'BlinkPeak', threshold: float) -> '_Batch':
        lb, ub = blink.minima

        is_clipped = self.spectrum.clipped[lb:ub]
        is_snr_low = np.abs(self.spectrum.intensity[lb:ub]) / self.spectrum.deviation[lb:ub] < threshold
        mask = ~is_clipped & ~is_snr_low

        x = self.spectrum.number[lb:ub][mask]
        y = self.spectrum.intensity[lb:ub][mask]

        return _Batch(x, y)

    def create_from_frame(self, t: int, threshold: float) -> '_Batch':
        is_clipped = self.spectrum.clipped[t]
        is_snr_low = np.abs(self.spectrum.intensity[t]) / self.spectrum.deviation[t] < threshold
        mask = ~is_clipped & ~is_snr_low

        x = self.spectrum.number[mask]
        y = self.spectrum.intensity[mask] if self.spectrum.n_times == 1 else self.spectrum.intensity[t, mask]

        return _Batch(x, y)


class _Batch:
    factory = _FactoryBatch

    def __init__(self, x: Array[T], y: Array[U]):
        self.x = x
        self.y = y


class FactoryGrid:

    def __init__(self, spectrum: Spectrum):
        self.spectrum = spectrum

    def create_from_blinks(
        self,
        blinks: Sequence['BlinkPeak'],
        offset: Array[T] | None = None,
        scale: Array[float] | None = None,
        background: Array[float] | None = None,
        threshold: float = 0,
    ) -> 'Grid':
        """Get a grid from sequence of blinks from spectrum."""
        assert self.spectrum.n_times == 1, 'time resolved spectra are not supported!'

        batches = tuple(
            _Batch.factory(spectrum=self.spectrum).create_from_blink(blink=blink, threshold=threshold)
            for blink in blinks
        )

        return self._create(
            batches=batches,
            offset=offset,
            scale=scale,
            background=background,
        )

    def create_from_frames(self, offset: Array[T] | None = None, scale: Array[float] | None = None, background: Array[float] | None = None, threshold: float = 0) -> 'Grid':
        """Get a grid from frames of spectra (for example, series of shifted on wavelength)."""
        # assert spectrum.n_times > 1, 'only time resolved spectra are supported!'

        batches = tuple(
            _Batch.factory(spectrum=self.spectrum).create_from_frame(t=t, threshold=threshold)
            for t in range(self.spectrum.n_times)
        )

        return self._create(
            batches=batches,
            offset=offset,
            scale=scale,
            background=background,
        )

    def _create(self, batches: Sequence[_Batch], offset: Array[T] | None = None, scale: Array[float] | None = None, background: Array[U] | None = None) -> 'Grid':
        """Get a grid from sequence of batches."""
        n_batches = len(batches)

        if offset is None:
            offset = np.full(n_batches, 0)
        assert len(offset) == n_batches, f'len of `offset` have to be equal of `n_batches`: {n_batches}'

        if scale is None:
            scale = np.full(n_batches, 1)
        assert len(scale) == n_batches, f'len of `scale` have to be equal of `n_batches`: {n_batches}'

        if background is None:
            background = np.full(n_batches, 0)
        assert len(background) == n_batches, f'len of `background` have to be equal of `n_batches`: {n_batches}'

        #
        x, y = [], []
        for t, batch in enumerate(batches):
            x.extend(batch.x - offset[t])
            y.extend((batch.y - background[t]) / scale[t])

        x, y = np.array(x).squeeze(), np.array(y).squeeze()

        index = np.argsort(x)

        #
        return Grid(
            x=x[index],
            y=y[index],
        )


class Grid:
    factory = FactoryGrid

    def __init__(self, x: Array[T], y: Array[float] | None = None, units: T | None = None):
        assert len(x) == len(y)

        #
        self._x = x
        self._y = y
        self._units = units

        self._interpolate = None

    @property
    def x(self) -> Array[T]:
        return self._x

    @property
    def y(self) -> Array[float]:
        return self._y

    @property
    def units(self) -> T | None:
        return self._units

    @property
    def xlabel(self) -> str:
        return '{label} {units}'.format(
            label={
                Number: r'$number$',
                MicroMeter: r'$x$',
                PicoMeter: r'$x$',
            }.get(self.units, ''),
            units=self.xunits,
        )

    @property
    def xunits(self) -> str:
        return {
            Number: r'',
            MicroMeter: r'[$\mu m$]',
            PicoMeter: r'[$pm$]',
        }.get(self.units, '')

    @property
    def interpolate(self) -> Callable[[Array[T]], Array[float]]:
        """Interpolate `grid` by linear interpolator."""
        if self._interpolate is None:
            self._interpolate = interpolate.interp1d(
                self.x, self.y,
                kind='linear',
                bounds_error=False,
                fill_value=0,
            )

        return self._interpolate

    def space(self, n_points: int = 1000) -> Array[T]:
        return np.linspace(min(self.x), max(self.x), n_points)

    def shift(self, value: T) -> 'Grid':
        """Shift `grid` by the `value`."""

        return Grid(
            x=self.x - value,
            y=self.y,
            units=self.units,
        )

    def rescale(self, value: float, units: T) -> 'Grid':
        """Rescale `grid` by the `value`. It is used to change `units`!"""

        return Grid(
            x=self.x/value,
            y=self.y*value,
            units=units,
        )

    def normalize(self, value: float | None = None) -> 'Grid':
        """Normalize `grid`."""
        value = value or 1/integrate.quad(self.interpolate, a=min(self.x), b=max(self.x))[0]

        return Grid(
            x=self.x,
            y=self.y*value,
            units=self.units,
        )

    def show(self) -> None:
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

        x, y = self.x, self.y
        plt.plot(
            x, y,
            color='red', linestyle='none', marker='s', markersize=3,
            alpha=1,
        )

        plt.xlabel(self.xlabel)
        plt.ylabel(r'$f$')
        plt.grid(color='grey', linestyle=':')

        plt.show()

    def __len__(self) -> int:
        return len(self.x)

    def __iter__(self) -> Iterator:
        warn(
            message='Iteration on the `grid` by points will be removed in the future!',
            category=DeprecationWarning,
            stacklevel=1,
        )

        return IteratorGrid(
            x=self.x,
            y=self.y,
        )

    def __str__(self) -> str:
        cls = self.__class__

        return f'{cls.__name__}({self.units})'

    def __add__(self, other: float | Array[float]) -> 'Grid':
        cls = self.__class__

        return cls(
            x=self.x,
            y=self.y + other,
            units=self.units,
        )

    def __iadd__(self, other: float | Array[float]) -> 'Grid':
        return self + other

    def __radd__(self, other: float | Array[float]) -> 'Grid':
        return self + other

    def __sub__(self, other: float | Array[float]) -> 'Grid':
        cls = self.__class__

        return cls(
            x=self.x,
            y=self.y - other,
            units=self.units,
        )

    def __isub__(self, other: float | Array[float]) -> 'Grid':
        return self - other

    def __rsub__(self, other: float | Array[float]) -> 'Grid':
        return self - other
