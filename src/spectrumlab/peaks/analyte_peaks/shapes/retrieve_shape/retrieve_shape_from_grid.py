from collections.abc import Sequence
from functools import partial

from scipy import optimize

from spectrumlab.grids import Grid
from spectrumlab.peaks.analyte_peaks.shapes import PeakShape
from spectrumlab.peaks.analyte_peaks.shapes.retrieve_shape.utils import FullParams
from spectrumlab.types import MicroMeter
from spectrumlab.utils import mse


def retrieve_shape_from_grid(
    grid: Grid,
    rx: MicroMeter = 20,
    dx: MicroMeter = 1e-2,
) -> PeakShape:

    def _loss(grid: Grid, params: Sequence[float]) -> float:

        shape_params, scope_params = FullParams.parse(
            grid=grid,
            params=params,
        )
        shape = PeakShape(**shape_params, rx=rx, dx=dx)

        return mse(
            y=grid.y,
            y_hat=shape(x=grid.x, **scope_params),
        )

    params = FullParams(grid=grid)
    res = optimize.minimize(
        partial(_loss, grid),
        params.initial,
        # method='SLSQP',
        bounds=params.bounds,
    )
    # assert res['success'], 'Optimization is not succeeded!'

    shape_params, _ = FullParams.parse(
        grid=grid,
        params=res['x'],
    )

    shape = PeakShape(**shape_params, rx=rx, dx=dx)
    return shape
