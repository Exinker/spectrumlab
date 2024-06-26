import abc
from typing import TYPE_CHECKING

from spectrumlab.types import Number

if TYPE_CHECKING:
    from spectrumlab.peak.analyte_peak import AnalytePeak


class AbstractPositionCalculator(abc.ABC):

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    @abc.abstractmethod
    def calculate(self, peak: 'AnalytePeak') -> Number:
        raise NotImplementedError
