import os

import matplotlib as mpl


def fetch_cmap(filename: str):

    filepath = os.path.join(os.path.dirname(__file__), 'colormaps', filename)
    with open(filepath, 'r') as file:
        lines = [list(map(float, line.strip().split(','))) for line in file.readlines()]

    cmap = mpl.colors.LinearSegmentedColormap.from_list('absorbance', lines)

    return cmap
