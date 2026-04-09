import itertools

import numpy as np
from pysdot import OptimalTransport, PowerDiagram
from pysdot.domain_types import ConvexPolyhedraAssembly

from synthetmic._internal._data import _kdtree_closest_points
from synthetmic._internal._validate import (
    _between,
    _check_array,
    _check_periodic,
    _check_points,
    _compose_rules,
    _gt,
    _gte,
    _is_instance,
)


def mesh_diagram(
    points: np.ndarray,
    pd: PowerDiagram,
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
    n_jobs : int, default=-1
        Number of parallel workers to use.
    Returns
    -------
    grain_indices : np.ndarray
        Array of shape (n_points,) where grain_indices[i] is
        the index of the grain containing point i.
    """
    _check_points(points)

    all_points = pd.get_positions()

    if points.shape[1] != all_points.shape[1]:
        raise ValueError(
            "`points` and power diagram positions must have the same "
            f"number of coordinates, but got {points.shape[1]} vs {all_points.shape[1]}."
        )

    weights = pd.get_weights()

    TOL = 1e-16
    if weights is None or np.allclose(weights, TOL):
        return _kdtree_closest_points(
            points=points, all_points=all_points, workers=n_jobs
        )

    # Lift the seeds to d + 1 dimensions
    extra_coords = np.sqrt(weights.max() - weights)
    all_points = np.column_stack((all_points, extra_coords))

    # Lift the points to 4 dimensions
    points = np.column_stack((points, np.zeros(points.shape[0])))

    return _kdtree_closest_points(points=points, all_points=all_points, workers=n_jobs)


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


def validate_generator_params(
    tol: float | None,
    n_iter: int,
    damp_param: float,
    verbose: bool,
) -> None:
    if tol is not None:
        _compose_rules(_is_instance(int, float), _gt(rhs=0.0))(tol, "tol")

    _compose_rules(_is_instance(int), _gte(rhs=0))(n_iter, "n_iter")
    _compose_rules(_is_instance(int, float), _between(left=0.0, right=1.0))(
        damp_param, "damp_param"
    )
    _is_instance(bool)(verbose, "verbose")

    return None


def validate_fit_args(
    seeds: np.ndarray,
    volumes: np.ndarray | None,
    domain: np.ndarray,
    periodic: list[bool] | None,
    init_weights: np.ndarray | None,
) -> None:
    _compose_rules(_is_instance(np.ndarray), _check_array(allowed_types=[float, int]))(
        seeds, "seeds"
    )

    if volumes is not None:
        _compose_rules(
            _is_instance(np.ndarray), _check_array(allowed_types=[float, int])
        )(volumes, "volumes")

    _compose_rules(
        _is_instance(np.ndarray),
        _check_array(allowed_types=[float, int], allowed_shapes=[(2, 2), (3, 2)]),
    )(domain, "domain")

    _is_instance(list, allow_none=True)(periodic, "periodic")
    if periodic is not None:
        _check_periodic(periodic, "periodic")

    _is_instance(np.ndarray, allow_none=True)(init_weights, "init_weights")
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
