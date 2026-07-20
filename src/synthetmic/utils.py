import itertools

import numpy as np
from pysdot import OptimalTransport, PowerDiagram
from pysdot.domain_types import ConvexPolyhedraAssembly

from synthetmic import _data as dt
from synthetmic import _validate as vd
from synthetmic._deprecated import warn_deprecated
from synthetmic.typing import (
    BoolSequence,
    FloatArray,
    FloatSequence,
    IntArray,
    IntSequence,
)


def compute_cell_centers(
    origin: FloatSequence, size: FloatSequence, points_per_dim: IntSequence
) -> FloatArray:
    """
    Compute the coordinates of uniformly spaced cell centers on a Cartesian grid.

    The domain is defined by its physical ``origin`` and ``size``. Each dimension
    is divided into the number of cells specified by ``points_per_dim``, and the
    returned coordinates correspond to the center of every cell.

    Parameters
    ----------
    origin : sequence of float, shape (N,)
        Physical coordinates of the lower corner of the domain.
    size : sequence of float, shape (N,)
        Physical extent of the domain along each dimension.
    points_per_dim : sequence of int, shape (N,)
        Number of grid points along each dimension.

    Returns
    -------
    centers : FloatArray
        Array of shape ``(*points_per_dim, N)`` containing the coordinates of
        every cell center. The last axis stores the coordinate components, so
        ``centers[i0, i1, ..., iN]`` gives the physical coordinates of the
        corresponding cell center.

    Notes
    -----
    The cell spacing in each dimension is

    ``Δ = size / points_per_dim``

    and the center coordinates are located at

    ``origin + (k + 0.5) * Δ``

    for ``k = 0, ..., points_per_dim - 1``.

    Examples
    --------
    Create the centers of a 2D grid with four cells in x and three cells in y:

    >>> centers = compute_cell_centers(
    ...     origin=(0.0, 0.0),
    ...     size=(4.0, 3.0),
    ...     points_per_dim=(4, 3),
    ... )
    >>> centers.shape
    (4, 3, 2)
    >>> centers[0, 0]
    array([0.5, 0.5])
    >>> centers[-1, -1]
    array([3.5, 2.5])
    """
    delta = np.asarray(size, dtype=float) / np.asarray(points_per_dim, dtype=int)

    axes = [
        origin[dim] + (np.arange(points_per_dim[dim]) + 0.5) * delta[dim]
        for dim in range(len(points_per_dim))
    ]
    mesh_grids = np.meshgrid(*axes, indexing="ij")

    return np.stack(mesh_grids, axis=-1)


def assign_points_to_grains(
    points: FloatArray,
    pd: PowerDiagram,
    domain: FloatArray | None = None,
    n_jobs: int = -1,
) -> IntArray:
    """
    For each point in `points`, compute which cell it belongs to
    in the power diagram `pd`.

    Parameters
    ----------
    points : FloatArray
        Array of shape (n_points, dim) with 2D or 3D coordinates.
    pd : pysdot.PowerDiagram
        Power diagram object.
    domain : FloatArray or None, optional, default=None
        If not None, it represents the minimum and maximum coordinates of the box
        in each of the d dimensions (d = 2 or 3), and the underlying power diagram will
        be treated as periodic in all dimensions. The positions of the power diagram will be
        mapped to the domain.
    n_jobs : int, optional, default=-1
        Number of parallel workers to use.

    Returns
    -------
    grain_indices : IntArray
        Array of shape (len(points),) where grain_indices[i] is
        the index of the grain containing point i.
    """

    points = np.asarray(points)
    vd.check_points(points)

    positions = pd.get_positions()
    if points.shape[1] != positions.shape[1]:
        raise ValueError(
            "`points` and diagram positions must have the same "
            f"number of coordinates, but got {points.shape[1]} vs {positions.shape[1]}."
        )
    weights = pd.get_weights()

    lifted_points = dt.lift_points(points)

    if domain is None:
        lifted_positions = dt.lift_positions(positions=positions, weights=weights)

        return dt.kdtree_closest_points(
            points=lifted_points,
            all_points=lifted_positions,
            workers=n_jobs,
            boxsize=None,
        )

    # Map  positions back to domain before lifting.
    domain = np.asarray(domain)
    boxsize = domain[:, 1] - domain[:, 0]
    mapped_positions = dt.map_positions(positions=positions, boxsize=boxsize)
    lifted_positions = dt.lift_positions(positions=mapped_positions, weights=weights)

    boxsize = np.append(
        boxsize,
        dt.compute_non_periodic_size(
            max_coord=lifted_positions[:, -1].max(), boxsize=boxsize
        ),
    )

    return dt.kdtree_closest_points(
        points=lifted_points,
        all_points=lifted_positions,
        workers=n_jobs,
        boxsize=tuple(boxsize),
    )


def mesh_diagram(
    points: FloatArray,
    pd: PowerDiagram,
    domain: FloatArray | None = None,
    n_jobs: int = -1,
) -> IntArray:
    """
    Deprecated. Use `synthetmic.utils.assign_points_to_cells` instead.
    """
    warn_deprecated(
        "`synthetmic.utils.mesh_diagram` is deprecated and will be removed in a future "
        "release. Please use `synthetmic.utils.assign_points_to_grains` instead."
    )
    return assign_points_to_grains(
        points=points,
        pd=pd,
        domain=domain,
        n_jobs=n_jobs,
    )


def build_domain(
    domain: FloatArray, periodic: BoolSequence | None
) -> tuple[ConvexPolyhedraAssembly, FloatSequence]:
    """
    Build a ConvexPolyhedraAssemply domain instance.

    Parameters
    ----------
    domain : FloatArray
        Represents the minimum and maximum coordinates of the box
        in each of the d dimensions (d = 2 or 3).
    periodic : BoolSequence or None
        Sequence of bool indicating whether or not the domain is periodic in
        each d dimensions. None indicates no periodicity in any direction.

    Returns
    -------
    tuple[ConvexPolyhedraAssembly, FloatSequence]
        Built domain, boxsize
    """
    omega = ConvexPolyhedraAssembly()

    domain = np.asarray(domain)
    mins = domain[:, 0].copy()
    maxs = domain[:, 1].copy()
    boxsize = domain[:, 1] - domain[:, 0]

    if periodic is not None:
        for k, p in enumerate(periodic):
            if p:
                mins[k] = mins[k] - boxsize[k]
                maxs[k] = maxs[k] + boxsize[k]

    omega.add_box(mins, maxs)

    return omega, tuple(boxsize)


def add_replicants(
    obj: OptimalTransport | PowerDiagram, periodic: BoolSequence, boxsize: FloatSequence
) -> None:
    """
    Adds replicants to the underlying PowerDiagram instance.

    Parameters
    ----------
    obj : pysdot.OptimalTransport or pysdot.PowerDiagram
        Object to add replicants to.
    periodic : BoolSequence
        Sequence of bool indicating whether or not the domain is periodic in
        each d dimensions. None indicates no periodicity in any direction.
    boxsize : FloatSequence
       Size of the box in each of the d dimensions.
    """

    if not any([isinstance(obj, OptimalTransport), isinstance(obj, PowerDiagram)]):
        raise ValueError("obj must be either OptimalTransport or PowerDiagram")

    if len(periodic) != len(boxsize):
        raise ValueError(
            "Length of periodic must be the same as the length of boxsize."
        )

    periodic_dict = {True: [-1, 0, 1], False: [0]}
    periodic_list = [periodic_dict[p] for p in periodic]

    cartesian_periodic = list(itertools.product(*periodic_list))

    for rep in cartesian_periodic:
        if rep != (0,) * len(periodic):
            modified_boxsize = rep * np.asarray(boxsize)
            if isinstance(obj, OptimalTransport):
                obj.pd.add_replication(modified_boxsize)
            else:
                obj.add_replication(modified_boxsize)
