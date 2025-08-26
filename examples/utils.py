from enum import StrEnum, auto
from typing import Any


class Example(StrEnum):
    TWO_DIM = "2d"
    THREE_DIM = "3d"
    ALL = "all"


class Analysis(StrEnum):
    RUNTIME = auto()
    DAMP = auto()
    ALL = auto()


class DocType(StrEnum):
    ARTICLE = auto()
    BEAMER = auto()


def get_rcparams() -> Any:
    SMALL_SIZE = 9
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 11

    FONT = {"family": "serif", "serif": ["Palatino"], "size": SMALL_SIZE}

    rc_params = {
        "axes.titlesize": MEDIUM_SIZE,
        "axes.labelsize": MEDIUM_SIZE,
        "xtick.labelsize": SMALL_SIZE,
        "ytick.labelsize": SMALL_SIZE,
        "legend.fontsize": SMALL_SIZE,
        "figure.titlesize": BIGGER_SIZE,
        "font.family": FONT["family"],
        "font.serif": FONT["serif"],
        "font.size": FONT["size"],
        "text.usetex": True,
        "figure.constrained_layout.use": True,
        "xtick.bottom": True,
        "ytick.left": True,
        "axes.grid": False,
    }

    return rc_params


def set_fig_size(
    width: float | str = DocType.ARTICLE,
    fraction: float = 1.0,
    subplots: tuple = (1, 1),
    adjust_height: float | None = None,
) -> tuple[float, float]:
    if width == DocType.ARTICLE:
        width_pt = 426.79135
    elif width == DocType.BEAMER:
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    if adjust_height is not None:
        golden_ratio += adjust_height

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)
