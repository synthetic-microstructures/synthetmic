import numpy as np
import pytest

from synthetmic import LaguerreDiagramGenerator
from synthetmic.data import toy, utils


@pytest.mark.parametrize(
    "seeds, expected",
    [
        (np.array([[0.5, 0.5], [0.5, 0.75]]), 8),
        (np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.75]]), 48),
    ],
)
def test_get_vertices(seeds: np.ndarray, expected: int) -> None:
    n_grains, space_dim = seeds.shape
    domain, domain_volume = toy._create_unit_domain(space_dim=space_dim)
    volumes = utils.create_constant_volumes(
        n_grains=n_grains, domain_volume=domain_volume
    )

    ldg = LaguerreDiagramGenerator(n_iter=0)
    ldg.fit(seeds=seeds, volumes=volumes, domain=domain)

    res = ldg.get_vertices()

    if space_dim == 2:
        sum_vertices = sum(len(v) for v in res.values())
    else:
        sum_vertices = 0
        for faces in res.values():
            for vertices in faces:
                sum_vertices += len(vertices)

    assert len(res) == n_grains
    assert sum_vertices == expected

    return None


def test_periodic_args() -> None:
    seeds = np.array(
        [
            [0.37454012, 0.15599452],
            [0.95071431, 0.05808361],
            [0.73199394, 0.86617615],
            [0.59865848, 0.60111501],
            [0.15601864, 0.70807258],
        ]
    )

    n_grains, space_dim = seeds.shape
    domain, domain_volume = toy._create_unit_domain(space_dim=space_dim)
    volumes = utils.create_constant_volumes(
        n_grains=n_grains, domain_volume=domain_volume
    )

    periodic_list = [None, [False, False]]
    results = []

    for periodic in periodic_list:
        generator = LaguerreDiagramGenerator(
            tol=1.0,
            n_iter=0,
            damp_param=1.0,
            verbose=False,
        )
        generator.fit(seeds=seeds, volumes=volumes, domain=domain, periodic=periodic)

        counts = [len(k) for k in generator.get_vertices().values()]
        results.append(counts)

        print(f"periodic: {periodic}, verts counts: {counts}")

    assert results[0] == results[1]

    return None
