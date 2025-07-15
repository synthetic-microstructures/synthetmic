class NotFittedError(ValueError, AttributeError):
    """Raised when attempting to use an unfitted generator."""
