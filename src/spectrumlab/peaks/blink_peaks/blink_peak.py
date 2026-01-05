from dataclasses import dataclass, field

from spectrumlab.peaks.base_peak import PeakABC
from spectrumlab.types import Number


@dataclass(slots=True)
class BlinkPeak(PeakABC):
    """Peak for any secondary application: masking of peaks for background algorithms, masking of overlapping peaks for intensity calculation and etc..."""  # noqa: 501

    minima: tuple[Number, Number]
    maxima: tuple[Number] | tuple[Number, Number] | tuple[Number, ...]

    except_edges: bool = field(default=False)

    def __repr__(self) -> str:
        cls = self.__class__

        content = '; '.join([
            f'minima: {self.minima}',
            f'maxima: {self.maxima}',
        ])
        return f'{cls.__name__}({content})'
