from enum import StrEnum, auto


class Initializer(StrEnum):
    RANDOM = auto()
    BANDED = auto()
    CLUSTERED = auto()
    MIXED_BANDED_AND_RANDOM = auto()


class Gradient(StrEnum):
    INCREASING = auto()
    LARGE_AT_MIDDLE = auto()


class PyvistaSupportedExtension(StrEnum):
    HTML = auto()
    SVG = auto()
    EPS = auto()
    PS = auto()
    PDF = auto()
    TEX = auto()
