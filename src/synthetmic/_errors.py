class NotFittedError(ValueError, AttributeError):
    """
    Raised when attempting to use an unfitted generator.
    """


def check_is_fitted(cls_: object, attributes: list[str]) -> None:
    for attr in attributes:
        if not hasattr(cls_, attr):
            raise NotFittedError(
                f"This {cls_.__class__.__name__} instance is not fitted yet. "
                f"Call 'fit' with appropriate arguments before using this instance."
            )
