from dataclasses import asdict

import numpy as np
from pysdot import PowerDiagram
from pysdot.domain_types import ConvexPolyhedraAssembly

from synthetmic.data.toy import (
    create_data_with_lognormal_volumes,
    create_periodicity,
    create_unit_domain,
    sample_random_seeds,
)
from synthetmic.data.utils import create_constant_volumes
from synthetmic.generate import LaguerreDiagramGenerator
from synthetmic.utils import mesh_diagram


def test_mesh_diagram_with_non_periodic_domain() -> None:
    seeds = np.array([[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]])
    domain = ConvexPolyhedraAssembly()
    domain.add_box(np.zeros(seeds.shape[1]), np.ones(seeds.shape[1]))
    weights = np.zeros(seeds.shape[0])
    pd = PowerDiagram(positions=seeds, weights=weights, domain=domain)

    points = np.array([[0.1, 0.1], [0.9, 0.2], [0.7, 0.6], [0.3, 0.8]])
    expected = np.array([0, 1, 2, 3], dtype=np.int32)
    assert np.allclose(expected, mesh_diagram(points=points, pd=pd))

    return None


def test_mesh_diagram_with_periodic_domain() -> None:
    SPACE_DIM = 2
    N_GRAINS = 4
    IS_PERIODIC = True

    domain, vol = create_unit_domain(space_dim=SPACE_DIM)
    seeds = sample_random_seeds(domain=domain, n_grains=N_GRAINS, random_state=42)
    volumes = create_constant_volumes(n_grains=N_GRAINS, domain_volume=vol)
    periodic = create_periodicity(space_dim=SPACE_DIM, is_periodic=IS_PERIODIC)

    g = LaguerreDiagramGenerator(verbose=False)
    g.fit(seeds=seeds, volumes=volumes, domain=domain, periodic=periodic)

    expected = np.array([2, 3, 0, 3, 2, 1], dtype=np.int32)
    points = np.array(
        [[0.1, 0.1], [0.2, 0.8], [0.5, 0.1], [0.9, 0.9], [0.9, 0.4], [0.6, 0.5]]
    )
    calculated = mesh_diagram(points=points, pd=g.optimal_transport_.pd, domain=domain)
    calculated_bruteforce = _bruteforce_mesh_periodic_diagram(
        points=points, pd=g.optimal_transport_.pd, box=domain
    )
    assert np.allclose(expected, calculated)
    assert np.allclose(expected, calculated_bruteforce)

    return None


def test_mesh_diagram_against_bruteforce() -> None:
    data = create_data_with_lognormal_volumes(is_periodic=True, random_state=40)
    g = LaguerreDiagramGenerator(verbose=False)
    g.fit(**asdict(data))

    points = sample_random_seeds(domain=data.domain, n_grains=100, random_state=40)
    calculated = mesh_diagram(
        points=points, pd=g.optimal_transport_.pd, domain=data.domain
    )
    calculated_bruteforce = _bruteforce_mesh_periodic_diagram(
        points=points, pd=g.optimal_transport_.pd, box=data.domain
    )
    assert np.allclose(calculated, calculated_bruteforce)

    return None


def _tile_positions(positions: np.ndarray, boxsize: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions)
    boxsize = np.asarray(boxsize)
    dimensions = positions.shape[1]

    tiled = positions.copy()

    for i in range(dimensions):
        # Create a displacement vector for the current dimension
        # e.g., for i=0 in 3D: [L1, 0, 0]
        offset = np.zeros(dimensions)
        offset[i] = boxsize[i]

        # Stack the current tiled block with a version shifted
        # negatively and a version shifted positively
        tiled = np.vstack((tiled - offset, tiled, tiled + offset))

    return tiled


def _tile_weights(weights: np.ndarray, dimensions: int) -> np.ndarray:
    return np.tile(weights, reps=3**dimensions)


def _bruteforce_mesh_periodic_diagram(
    points: np.ndarray, pd: PowerDiagram, box: np.ndarray
):

    x = pd.get_positions()
    N, D = x.shape
    periodic_seeds = _tile_positions(positions=x, boxsize=box[:, 1] - box[:, 0])

    w = pd.get_weights()
    periodic_weights = _tile_weights(weights=w, dimensions=D)

    num_points = np.size(points, 0)
    grain_indices = [0] * num_points

    for i in range(num_points):
        p = points[i]
        grain_indices[i] = (
            np.argmin(
                np.sum((periodic_seeds - p) * (periodic_seeds - p), axis=1)
                - periodic_weights
            )
            % N
        )

    return np.asarray(grain_indices, dtype=np.int32)
