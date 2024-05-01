import os
from typing import TypeAlias

import matplotlib as mpl


Colormap: TypeAlias = mpl.colors.LinearSegmentedColormap


def fetch_cmap(filename: str) -> Colormap:

    filepath = os.path.join(os.path.dirname(__file__), 'colormaps', filename)
    with open(filepath, 'r') as file:
        lines = [list(map(float, line.strip().split(','))) for line in file.readlines()]

    cmap = mpl.colors.LinearSegmentedColormap.from_list('absorbance', lines)

    return cmap
