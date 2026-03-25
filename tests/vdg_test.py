import numpy as np
import pytest

from synthetmic.data import toy
from synthetmic.generate import VoronoiDiagramGenerator


@pytest.mark.parametrize(
    "domain, seeds, expected_pos, expected_vols",
    [
        (
            np.array([[0, 1]] * 2),
            toy.sample_random_seeds(domain=np.array([[0, 1]] * 2), n_grains=4),
            np.array([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]]),
            np.array([[0.25] * 4]),
        ),
    ],
)
def test_pos_and_vols(
    domain: np.ndarray,
    seeds: np.ndarray,
    expected_pos: np.ndarray,
    expected_vols: np.ndarray,
) -> None:
    vdg = VoronoiDiagramGenerator(
        n_iter=50,
        damp_param=1,
        verbose=True,
    )
    vdg.fit(
        seeds=seeds,
        domain=domain,
        periodic=None,
    )

    print(vdg.get_positions())
    assert np.allclose(vdg.get_fitted_volumes(), expected_vols)

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

    _, space_dim = seeds.shape
    domain, _ = toy._create_unit_domain(space_dim=space_dim)

    periodic_list = [None, [False] * space_dim]
    results = []

    for periodic in periodic_list:
        vdg = VoronoiDiagramGenerator(
            n_iter=0,
            damp_param=1.0,
            verbose=False,
        )
        vdg.fit(seeds=seeds, domain=domain, periodic=periodic)

        counts = [len(k) for k in vdg.get_vertices().values()]
        results.append(counts)

    assert results[0] == results[1]

    return None
