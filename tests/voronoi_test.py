import numpy as np
import pytest

from synthetmic import DiagramConfig, VoronoiDiagramGenerator
from synthetmic.data import toy


@pytest.fixture
def config() -> DiagramConfig:
    return toy.create_data_with_constant_volumes(space_dim=3)


@pytest.mark.parametrize(
    "domain, seeds, expected_vols",
    [
        (
            np.array([[0, 1]] * 2),
            toy.sample_random_seeds(domain=np.array([[0, 1]] * 2), n_grains=4),
            np.array([[0.25] * 4]),
        ),
    ],
)
def test_pos_and_vols(
    domain: np.ndarray,
    seeds: np.ndarray,
    expected_vols: np.ndarray,
) -> None:
    vdg = VoronoiDiagramGenerator(n_iter=50, damp_param=1)
    vdg.fit(DiagramConfig(seeds=seeds, domain=domain, periodic=None))

    assert np.allclose(vdg.get_fitted_volumes(), expected_vols)
