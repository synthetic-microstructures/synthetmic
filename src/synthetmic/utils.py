import itertools

import numpy as np
from pysdot import OptimalTransport, PowerDiagram
from pysdot.domain_types import ConvexPolyhedraAssembly

from synthetmic._internal import _data as dt
from synthetmic._internal import _validate as vd


def mesh_diagram(
    points: np.ndarray,
    pd: PowerDiagram,
    domain: np.ndarray | None = None,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    For each point in `points`, compute which cell it belongs to
    in the power diagram `pd`.

    Parameters
    ----------
    points : numpy.ndarray
        Array of shape (n_points, dim) with 2D or 3D coordinates.
    pd : pysdot.PowerDiagram
        Power diagram object.
    n_jobs : int, default=-1
        Number of parallel workers to use.
    domain: numpy.ndarray or None, default=None
        If not None, it represents the minimum and maximum coordinates of the box
        in each of the d dimensions (d=2,3) and the underlying power diagram will
        be treated as periodic in all dimensions. The positions of the power diagram will be
        mapped to the domain.

    Returns
    -------
    grain_indices : numpy.ndarray
        Array of shape (len(points),) where grain_indices[i] is
        the index of the grain containing point i.
    """

    points = np.asarray(points)
    vd._check_points(points)

    positions = pd.get_positions()
    if points.shape[1] != positions.shape[1]:
        raise ValueError(
            "`points` and diagram positions must have the same "
            f"number of coordinates, but got {points.shape[1]} vs {positions.shape[1]}."
        )
    weights = pd.get_weights()

    lifted_points = dt._lift_points(points)

    if domain is None:
        lifted_positions = dt._lift_positions(positions=positions, weights=weights)

        return dt._kdtree_closest_points(
            points=lifted_points,
            all_points=lifted_positions,
            workers=n_jobs,
            boxsize=None,
        )

    # Map  positions back to domain before lifting.
    domain = np.asarray(domain)
    boxsize = domain[:, 1] - domain[:, 0]
    mapped_positions = dt._map_positions(positions=positions, boxsize=boxsize)
    lifted_positions = dt._lift_positions(positions=mapped_positions, weights=weights)

    boxsize = np.append(
        boxsize,
        dt._compute_non_periodic_size(
            max_coord=lifted_positions[:, -1].max(), boxsize=boxsize
        ),
    )

    return dt._kdtree_closest_points(
        points=lifted_points,
        all_points=lifted_positions,
        workers=n_jobs,
        boxsize=boxsize,
    )


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
        vd._compose_rules(vd._is_instance(int, float), vd._gt(rhs=0.0))(tol, "tol")

    vd._compose_rules(vd._is_instance(int), vd._gte(rhs=0))(n_iter, "n_iter")
    vd._compose_rules(vd._is_instance(int, float), vd._between(left=0.0, right=1.0))(
        damp_param, "damp_param"
    )
    vd._is_instance(bool)(verbose, "verbose")

    return None


def validate_fit_args(
    seeds: np.ndarray,
    volumes: np.ndarray | None,
    domain: np.ndarray,
    periodic: list[bool] | None,
    init_weights: np.ndarray | None,
) -> None:
    vd._compose_rules(
        vd._is_instance(np.ndarray), vd._check_array(allowed_types=[float, int])
    )(seeds, "seeds")

    if volumes is not None:
        vd._compose_rules(
            vd._is_instance(np.ndarray), vd._check_array(allowed_types=[float, int])
        )(volumes, "volumes")

    vd._compose_rules(
        vd._is_instance(np.ndarray),
        vd._check_array(allowed_types=[float, int], allowed_shapes=[(2, 2), (3, 2)]),
    )(domain, "domain")

    vd._is_instance(list, allow_none=True)(periodic, "periodic")
    if periodic is not None:
        vd._check_periodic(periodic, "periodic")

    vd._is_instance(np.ndarray, allow_none=True)(init_weights, "init_weights")
    if init_weights is not None:
        vd._check_array(allowed_types=[float, int])(init_weights, "init_weights")

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
