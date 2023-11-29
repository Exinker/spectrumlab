
from dataclasses import dataclass, field
from typing import TypeAlias


@dataclass
class CalibrationCurveConcentrationConfig:
    verbose: bool = field(default=False)


# --------        typing        --------
ConcentrationConfig: TypeAlias = CalibrationCurveConcentrationConfig
