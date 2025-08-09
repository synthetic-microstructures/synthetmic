from typing import Any, Callable, Type

import numpy as np


class NotFittedError(ValueError, AttributeError):
    """Raised when attempting to use an unfitted generator."""


def _gt(rhs: float) -> Callable[[float | None, str], None]:
    def _out(x: float | None, name: str) -> None:
        if x <= rhs or x is None:
            raise ValueError(f"{name} must be greater than {rhs} but {x} is given.")

        return None

    return _out


def _gte(rhs: float) -> Callable[[float | None, str], None]:
    def _out(x: float | None, name: str) -> None:
        if x < rhs or x is None:
            raise ValueError(
                f"{name} must be greater than or equal to {rhs} but {x} is given."
            )
        return None

    return _out


def _is_instance(
    instance: Type, allow_none: bool = False
) -> Callable[[Any, str], None]:
    def _out(x: Any, name: str) -> None:
        rule = (
            isinstance(x, instance) or (x is None)
            if allow_none
            else isinstance(x, instance)
        )

        if not rule:
            raise TypeError(
                f"{name} must be of type {instance} but {type(x)} is provided."
            )

        return None

    return _out


def _between(
    left: float,
    right: float,
    left_open: bool = False,
    right_open: bool = False,
    both_open: bool = False,
) -> Callable[[float | None, str], None]:
    def _rule(x: float) -> bool:
        if left_open:
            return left < x <= right

        if right_open:
            return left <= x < right

        if both_open:
            return left < x < right

        return left <= x <= right

    def _out(x: float | None, name: str) -> None:
        if (not _rule(x)) or x is None:
            raise ValueError(
                f"{name} must be between {left} and {right}, but {x} is given."
            )

        return None

    return _out


def _compose_rules(*args) -> Callable:
    rule_fns = [arg for arg in args if callable(arg)]

    def _out(x: Any, name: str):
        for rule_fn in rule_fns:
            res = rule_fn(x, name)
            if res is not None:
                return res

    return _out


def _check_array(
    allowed_types: list[Type], allowed_shapes: list[tuple[int, int]] | None = None
) -> Callable[[np.ndarray, str], None]:
    def _out(x: np.ndarray, name: str) -> None:
        if x.size == 0:
            raise ValueError(f"{name} is empty. Input required a non-empty ndarray.")

        if x.dtype not in allowed_types:
            raise ValueError(
                f"{name} contain elements of wrong type {x.dtype}. Allowed types are {allowed_types}."
            )

        if allowed_shapes is not None:
            if x.shape not in allowed_shapes:
                raise ValueError(
                    f"{name} hase a wrong shape {x.shape}. Allowed shapes are {allowed_shapes}."
                )

        return None

    return _out


def _check_periodic(x: list[bool], name: str) -> None:
    if len(x) not in (2, 3):
        raise ValueError(
            f"invalid {name} length {len(x)}; expected length to be 2 or 3."
        )

    if not all(isinstance(var, bool) for var in x):
        raise ValueError(f"all entries in {name} must be bool.")

    return None


def validate_generator_params(
    tol: float,
    n_iter: int,
    damp_param: float,
    verbose: bool,
) -> None:
    _compose_rules(_is_instance(instance=float), _gt(rhs=0.0))(tol, "tol")
    _compose_rules(_is_instance(instance=int), _gte(rhs=0))(n_iter, "n_iter")
    _compose_rules(_is_instance(instance=float), _between(left=0.0, right=1.0))(
        damp_param, "damp_param"
    )
    _is_instance(instance=bool)(verbose, "verbose")

    return None


def validate_fit_args(
    seeds: np.ndarray,
    volumes: np.ndarray,
    domain: np.ndarray,
    periodic: list[bool] | None,
    init_weights: np.ndarray | None,
) -> None:
    _compose_rules(
        _is_instance(instance=np.ndarray), _check_array(allowed_types=[float, int])
    )(seeds, "seeds")

    _compose_rules(
        _is_instance(instance=np.ndarray), _check_array(allowed_types=[float, int])
    )(volumes, "volumes")

    _compose_rules(
        _is_instance(instance=np.ndarray),
        _check_array(allowed_types=[float, int], allowed_shapes=[(2, 2), (3, 2)]),
    )(domain, "domain")

    _is_instance(instance=list, allow_none=True)(periodic, "periodic")
    if periodic is not None:
        _check_periodic(periodic, "periodic")

    _is_instance(instance=np.ndarray, allow_none=True)(init_weights, "init_weights")
    if init_weights is not None:
        _check_array(allowed_types=[float, int])(init_weights, "init_weights")

    # check if the number of samples match
    num_samples = []
    for x in (seeds, volumes, init_weights):
        if x is not None:
            num_samples.append(x.shape[0])

    if len(set(num_samples)) > 1:
        raise ValueError(
            f"one or more of seeds, volumes, and init_weights have inconsistent number of samples: {num_samples}."
        )

    # check if space dimensions match
    space_dims = [seeds.shape[1], domain.shape[0]]
    if periodic is not None:
        space_dims.append(len(periodic))

    if len(set(space_dims)) > 1:
        raise ValueError(
            f"one or more of seeds, domain, and periodic have inconsistent space dimension: {space_dims}."
        )

    if not set(space_dims).issubset({2, 3}):
        raise ValueError(f"""one or more of seeds, domain, and periodic have wrong space dimension: {space_dims}.
            Supported space dimensions are 2 and 3.""")

    return None
