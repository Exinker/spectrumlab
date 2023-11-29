
from dataclasses import dataclass, field


@dataclass
class MaskConfig:
    noise_level: int = field(default=5)
