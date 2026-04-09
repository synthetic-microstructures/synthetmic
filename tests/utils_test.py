import numpy as np
from pysdot import PowerDiagram
from pysdot.domain_types import ConvexPolyhedraAssembly

from synthetmic.utils import mesh_diagram


def test_mesh_diagram() -> None:
    seeds = np.array([[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]])
    domain = ConvexPolyhedraAssembly()
    domain.add_box(np.zeros(seeds.shape[1]), np.ones(seeds.shape[1]))
    weights = np.zeros(seeds.shape[0])
    pd = PowerDiagram(positions=seeds, weights=weights, domain=domain)

    points = np.array([[0.1, 0.1], [0.9, 0.2], [0.7, 0.6], [0.3, 0.8]])

    expected = np.array([0, 1, 2, 3], dtype=np.int32)

    assert np.allclose(expected, mesh_diagram(points=points, pd=pd))

    return None
