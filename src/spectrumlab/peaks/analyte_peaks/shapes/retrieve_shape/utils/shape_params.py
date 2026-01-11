from spectrumlab.peaks.analyte_peaks.shapes.retrieve_shape.config import (
    RETRIEVE_SHAPE_CONFIG as CONFIG,
)
from spectrumlab.peaks.analyte_peaks.shapes.retrieve_shape.utils.base_params import (
    Param,
    ParamsABC,
)
from spectrumlab.types import Number


class ShapeParams(ParamsABC):

    def __init__(
        self,
        width: Number | None = None,
        asymmetry: float | None = None,
        ratio: float | None = None,
    ) -> None:
        super().__init__([
            Param('width', 2.0, (CONFIG.min_width, CONFIG.max_width), width),
            Param('asymmetry', 0.0, (-CONFIG.max_asymmetry, +CONFIG.max_asymmetry), asymmetry),
            Param('ratio', 0.1, (0, 1), ratio),
        ])

        self.name = 'shape'
