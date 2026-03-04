import itertools
from typing import Any, Callable, Type

import numpy as np
from joblib import Parallel, delayed
from pysdot import OptimalTransport, PowerDiagram
from pysdot.domain_types import ConvexPolyhedraAssembly
from scipy.spatial.distance import cdist


def mesh_diagram(
    points: np.ndarray,
    pd: PowerDiagram,
    batch_size: int = 100,
    parallel: bool = False,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    For each point in `points`, compute which cell it belongs to
    in the power diagram `pd`.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (n_points, dim) with 2D or 3D coordinates.
    pd : PowerDiagram
        Power diagram object.
    batch_size : int, default=100
        Number of points processed per batch.
    parallel : bool, default=False
        If True, process batches in parallel using joblib.
        Recommended only for large numbers of points and/or grains
        (seeds), where distance computation becomes expensive.
    n_jobs : int, default=-1
        Number of parallel workers to use when `parallel=True`.
        -1 uses all available CPU cores.

    Returns
    -------
    grain_indices : np.ndarray
        Array of shape (n_points,) where grain_indices[i] is
        the index of the grain containing point i.
    """
    _check_points(points)

    x = pd.get_positions()

    if points.shape[1] != x.shape[1]:
        raise ValueError(
            "`points` and power diagram positions must have the same "
            f"number of coordinates, but got {points.shape[1]} vs {x.shape[1]}."
        )

    w = pd.get_weights()
    num_points = len(points)

    if not parallel:
        grain_indices = np.empty(num_points, dtype=np.int32)

        for i in range(0, num_points, batch_size):
            batch = points[i : i + batch_size]
            squared_distances = cdist(batch, x, metric="sqeuclidean")
            grain_indices[i : i + batch_size] = np.argmin(squared_distances - w, axis=1)

        return grain_indices

    batches = [
        (i, min(i + batch_size, num_points)) for i in range(0, num_points, batch_size)
    ]

    def _process_batch(start: int, end: int):
        batch = points[start:end]
        squared_distances = cdist(batch, x, metric="sqeuclidean")
        return np.argmin(squared_distances - w, axis=1)

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_process_batch)(start, end) for start, end in batches
    )

    return np.concatenate(results)


def build_domain(
    domain: np.ndarray, periodic: list[bool] | None
) -> tuple[ConvexPolyhedraAssembly, np.ndarray]:
    """
    Build a ConvexPolyhedraAssemply domain instance.
    """
    omega = ConvexPolyhedraAssembly()

    mins = domain[:, 0].copy()
    maxs = domain[:, 1].copy()
    lens = domain[:, 1] - domain[:, 0]

    if periodic is not None:
        for k, p in enumerate(periodic):
            if p:
                mins[k] = mins[k] - lens[k]
                maxs[k] = maxs[k] + lens[k]

    omega.add_box(mins, maxs)

    return omega, lens


def add_replicants(
    obj: OptimalTransport | PowerDiagram, periodic: list[bool], domain_lens: np.ndarray
) -> None:
    """
    Adds replicants to the underlying PowerDiagram instance.
    """

    if not any([isinstance(obj, OptimalTransport), isinstance(obj, PowerDiagram)]):
        raise ValueError("obj must be either OptimalTransport or PowerDiagram")

    if len(periodic) != domain_lens.size:
        raise ValueError("len of periodic must be the same as len of lens")

    periodic_dict = {True: [-1, 0, 1], False: [0]}
    periodic_list = [periodic_dict[p] for p in periodic]

    cartesian_periodic = list(itertools.product(*periodic_list))

    for rep in cartesian_periodic:
        if rep != (0,) * len(periodic):
            if isinstance(obj, OptimalTransport):
                obj.pd.add_replication(rep * domain_lens)
            else:
                obj.add_replication(rep * domain_lens)

    return None


class NotFittedError(ValueError, AttributeError):
    """
    Raised when attempting to use an unfitted generator.
    """


def _check_points(points: np.ndarray) -> None:
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

    return None


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
    tol: float | None,
    n_iter: int,
    damp_param: float,
    verbose: bool,
) -> None:
    if tol is not None:
        _compose_rules(_is_instance(instance=float), _gt(rhs=0.0))(tol, "tol")

    _compose_rules(_is_instance(instance=int), _gte(rhs=0))(n_iter, "n_iter")
    _compose_rules(_is_instance(instance=float), _between(left=0.0, right=1.0))(
        damp_param, "damp_param"
    )
    _is_instance(instance=bool)(verbose, "verbose")

    return None


def validate_fit_args(
    seeds: np.ndarray,
    volumes: np.ndarray | None,
    domain: np.ndarray,
    periodic: list[bool] | None,
    init_weights: np.ndarray | None,
) -> None:
    _compose_rules(
        _is_instance(instance=np.ndarray), _check_array(allowed_types=[float, int])
    )(seeds, "seeds")

    if volumes is not None:
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
