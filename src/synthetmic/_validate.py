from typing import Any, Callable, Type

import numpy as np

from synthetmic.typing import BoolSequence, FloatArray, IntArray, NumericArray, StrArray


def check_space_dim(
    seeds: FloatArray, domain: FloatArray, periodic: BoolSequence
) -> None:
    space_dims = [seeds.shape[1], domain.shape[0], len(periodic)]
    if len(set(space_dims)) > 1:
        raise ValueError(
            f"one or more of seeds, domain, and periodic have inconsistent space dimension: {space_dims}."
        )
    if not set(space_dims).issubset({2, 3}):
        raise ValueError(f"""one or more of seeds, domain, and periodic have wrong space dimension: {space_dims}.
                Supported space dimensions are 2 and 3.""")


def check_num_samples(
    seeds: FloatArray,
    phases: IntArray | StrArray,
    initial_weights: FloatArray,
    volumes: FloatArray,
) -> None:
    num_samples = [
        seeds.shape[0],
        phases.shape[0],
        initial_weights.shape[0],
        volumes.shape[0],
    ]

    if len(set(num_samples)) > 1:
        raise ValueError(
            f"one or more of seeds, volumes, and initial_weights have inconsistent number of samples: {num_samples}."
        )


def check_periodic(periodic: BoolSequence | None, space_dim: int) -> BoolSequence:
    if periodic is None:
        periodic = (False,) * space_dim
        compose_rules(is_instance(list, tuple), is_periodic())(periodic, "periodic")

    return tuple(periodic)


def check_volumes(
    volumes: FloatArray | None, domain: FloatArray, n_grains: int
) -> FloatArray:
    total_volume = np.prod(domain[:, 1] - domain[:, 0])

    if volumes is None:
        volumes = np.ones(n_grains) * total_volume / n_grains

    _TOTAL_VOL_TOL = 10e-6
    if abs(volumes.sum() - total_volume) > _TOTAL_VOL_TOL:
        raise ValueError(
            f"Total volume difference is greater than tolerance: {_TOTAL_VOL_TOL}."
        )

    compose_rules(
        is_instance(np.ndarray),
        check_array(allowed_types=[float, int], allowed_ndims=[1]),
    )(volumes, "volumes")

    return volumes


def check_initial_weights(
    initial_weights: FloatArray | None, n_grains: int
) -> FloatArray:
    if initial_weights is None:
        initial_weights = np.zeros(n_grains, dtype=float)
        compose_rules(
            is_instance(np.ndarray),
            check_array(allowed_types=[float, int], allowed_ndims=[1]),
        )(initial_weights, "initial_weights")

    return initial_weights


def check_seeds(seeds: FloatArray, domain: FloatArray) -> None:
    compose_rules(
        is_instance(np.ndarray),
        check_array(allowed_types=[float, int], allowed_ndims=[2, 3]),
    )(seeds, "seeds")
    check_points(seeds)

    coord_names = ("x", "y", "z")
    for i, bound in enumerate(domain):
        min_ = seeds[:, i].min()
        max_ = seeds[:, i].max()

        if not (bound[0] <= min_ <= max_ <= bound[1]):
            raise ValueError(
                f"""Expected {coord_names[i]}-coordinate values to be in {list(bound)}
                but values are in [{min_:.2f}, {max_:.2f}]."""
            )


def check_domain(domain: FloatArray) -> None:
    compose_rules(
        is_instance(np.ndarray),
        check_array(allowed_types=[float, int], allowed_shapes=[(2, 2), (3, 2)]),
    )(domain, "domain")


def check_phases(
    phases: IntArray | StrArray | None, n_grains: int
) -> IntArray | StrArray:
    if phases is None:
        phases = np.zeros(n_grains, dtype=int)

    compose_rules(
        is_instance(np.ndarray),
        check_array(allowed_types=[int, str], allowed_ndims=[1]),
    )(phases, "phases")

    return phases


def check_generator_config(
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
        check = any([isinstance(x, i) for i in args])
        rule = check or (x is None) if allow_none else check

        if not rule:
            raise TypeError(
                f"{name} must be of type {'or '.join([i.__name__ for i in args])} but {type(x).__name__} is provided."
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
    allowed_types: list[Type],
    allowed_shapes: list[tuple[int, int]] | None = None,
    allowed_ndims: list[int] | None = None,
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

        if allowed_ndims is not None:
            if x.ndim not in allowed_ndims:
                raise ValueError(
                    f"{name} hase a wrong ndim {x.ndim}. Allowed ndims are {allowed_ndims}."
                )

    return _out


def is_periodic() -> Callable[[BoolSequence, str], None]:
    def _out(x: BoolSequence, name: str) -> None:
        if len(x) not in (2, 3):
            raise ValueError(
                f"invalid {name} length {len(x)}; expected length to be 2 or 3."
            )

        if not all(isinstance(var, bool) for var in x):
            raise ValueError(f"all entries in {name} must be bool.")

    return _out
