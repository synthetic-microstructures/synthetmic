from typing import Any, Callable, Type

import numpy as np

from synthetmic.typing import BoolSequence, FloatArray, NumericArray


def validate_generator_config(
    tol: float | None,
    n_iter: int,
    damp_param: float,
) -> None:
    if tol is not None:
        compose_rules(is_instance(int, float), gt(rhs=0.0))(tol, "tol")

    compose_rules(is_instance(int), gte(rhs=0))(n_iter, "n_iter")
    compose_rules(is_instance(int, float), between(left=0.0, right=1.0))(
        damp_param, "damp_param"
    )


def check_points(points: FloatArray) -> None:
    points = np.asarray(points)

    if points.ndim != 2:
        raise ValueError(
            f"`points` must be a 2D array of shape (n, d). "
            f"Got array with shape {points.shape}."
        )

    if points.shape[0] == 0:
        raise ValueError("`points` must contain at least one point.")

    if points.shape[1] not in (2, 3):
        raise ValueError(
            f"`points` must have 2 or 3 columns (2D or 3D coordinates). "
            f"Got {points.shape[1]}."
        )


def gt(rhs: float) -> Callable[[float | None, str], None]:
    def _out(x: float | None, name: str) -> None:
        if x is None or x <= rhs:
            raise ValueError(f"{name} must be greater than {rhs} but {x} is given.")

    return _out


def gte(rhs: float) -> Callable[[float | None, str], None]:
    def _out(x: float | None, name: str) -> None:
        if x is None or x < rhs:
            raise ValueError(
                f"{name} must be greater than or equal to {rhs} but {x} is given."
            )

    return _out


def is_instance(*args, allow_none: bool = False) -> Callable[[Any, str], None]:
    def _out(x: Any, name: str) -> None:
        check = any(isinstance(x, i) for i in args)
        rule = check or (x is None) if allow_none else check

        if not rule:
            raise TypeError(
                f"{name} must be of type {'or '.join(args)} but {type(x)} is provided."
            )

    return _out


def between(
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
        if x is None or (not _rule(x)):
            raise ValueError(
                f"{name} must be between {left} and {right}, but {x} is given."
            )

    return _out


def compose_rules(*args) -> Callable:
    rule_fns = [arg for arg in args if callable(arg)]

    def _out(x: Any, name: str):
        for rule_fn in rule_fns:
            res = rule_fn(x, name)
            if res is not None:
                return res

    return _out


def check_array(
    allowed_types: list[Type], allowed_shapes: list[tuple[int, int]] | None = None
) -> Callable[[NumericArray, str], None]:
    def _out(x: NumericArray, name: str) -> None:
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

    return _out


def check_periodic(x: BoolSequence, name: str) -> None:
    if len(x) not in (2, 3):
        raise ValueError(
            f"invalid {name} length {len(x)}; expected length to be 2 or 3."
        )

    if not all(isinstance(var, bool) for var in x):
        raise ValueError(f"all entries in {name} must be bool.")
