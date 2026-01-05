from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from spectrumlab.types import Number

if TYPE_CHECKING:
    from spectrumlab.peaks.analyte_peaks.analyte_peak import AnalytePeak


class PositionEstimatorABC(ABC):

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    @abstractmethod
    def calculate(self, peak: 'AnalytePeak') -> Number:
        raise NotImplementedError
