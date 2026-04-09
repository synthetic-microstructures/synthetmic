from enum import StrEnum, auto


class _Initializer(StrEnum):
    RANDOM = auto()
    BANDED = auto()
    CLUSTERED = auto()
    MIXED_BANDED_AND_RANDOM = auto()


class _Gradient(StrEnum):
    INCREASING = auto()
    LARGE_AT_MIDDLE = auto()


class _PyvistaSupportedExtension(StrEnum):
    HTML = auto()
    SVG = auto()
    EPS = auto()
    PS = auto()
    PDF = auto()
    TEX = auto()
