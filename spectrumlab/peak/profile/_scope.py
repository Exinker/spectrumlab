
from dataclasses import dataclass, asdict

from spectrumlab.alias import Number
from spectrumlab.peak.profile.scope_variables import ScopeVariables


@dataclass
class Scope:
    """PeakProfile's scope type."""
    position: Number
    intensity: float
    background: float

    def __repr__(self) -> str:
        cls = self.__class__

        return f'{cls.__name__}(w={self.position:.4f}; a={self.intensity:.4f}; r={self.background:.4f})'

    def __iter__(self) -> None:
        return iter(asdict(self))
