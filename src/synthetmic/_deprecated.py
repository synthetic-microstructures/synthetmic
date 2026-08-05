import warnings


class SynthetMicDeprecationWarning(RuntimeWarning):
    pass


warnings.simplefilter("always", SynthetMicDeprecationWarning)


def warn_deprecated(message: str, stacklevel: int = 3) -> None:
    warnings.warn(message, SynthetMicDeprecationWarning, stacklevel=stacklevel)
