from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from spectrumlab.picture.colors import Color
from spectrumlab.types import R

if TYPE_CHECKING:
    from spectrumlab.peaks.analyte_peaks.analyte_peak import AnalytePeak


class IntensityEstimatorABC(ABC):

    def __init__(
        self,
        verbose: bool = False,
    ) -> None:

        self.verbose = verbose

    @property
    @abstractmethod
    def color(self) -> Color:
        raise NotImplementedError

    @abstractmethod
    def calculate(self, peak: 'AnalytePeak') -> R:
        raise NotImplementedError
