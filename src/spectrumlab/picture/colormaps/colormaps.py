import os
from typing import Literal, TypeAlias

import matplotlib as mpl


Colormap: TypeAlias = mpl.colors.LinearSegmentedColormap


def fetch_cmap(kind: Literal['emission', 'absorption'], name: str | None = None) -> Colormap:
    name = name or {
        'emission': None,
        'absorption': 'pantone-290',
    }.get(kind, None)

    filepath = os.path.join(os.path.dirname(__file__), kind, f'{name}.txt')
    with open(filepath, 'r') as file:
        lines = [list(map(float, line.strip().split(','))) for line in file.readlines()]

    cmap = mpl.colors.LinearSegmentedColormap.from_list(kind, lines)
    return cmap
