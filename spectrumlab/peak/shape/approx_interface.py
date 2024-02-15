from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Mapping, TYPE_CHECKING

from spectrumlab.alias import Array

if TYPE_CHECKING:
    from spectrumlab.peak.analyte_peak import AnalytePeak


class ApproxInterface(ABC):
    """Base peak's approx."""

    @abstractmethod
    def approx_keys(self) -> tuple[str]:
        raise NotImplementedError

    @abstractmethod
    def approx_initial(self, peak: 'AnalytePeak') -> Array[float]:
        raise NotImplementedError

    @abstractmethod
    def approx_bounds(self, peak: 'AnalytePeak') -> tuple[tuple[float,float]]:
        raise NotImplementedError

    def approx_parse(self, params: Sequence[float]) -> Mapping[str, float]:
        assert len(self.approx_keys()) == len(params)

        return {
            key: param
            for key, param in zip(self.approx_keys(), params)
        }
