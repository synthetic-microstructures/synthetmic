import numpy as np
import pytest

from synthetmic import DiagramConfig, VoronoiDiagramGenerator, VoronoiEvent
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


@pytest.mark.parametrize("damp_param", (0.0, 0.5, 1.0))
@pytest.mark.parametrize("n_iter", (10, 20, 30, 40, 50, 100))
def test_centroid_error_norm_monotonicity(
    damp_param: float, n_iter: int, config: DiagramConfig
) -> None:
    errors = np.zeros(n_iter)

    def callback(e: VoronoiEvent) -> None:
        errors[e.iteration - 1] = e.centroid_error_norm

    vdg = VoronoiDiagramGenerator(n_iter=n_iter, damp_param=damp_param)
    vdg.fit(config, callback=callback)

    assert np.all(np.diff(errors) <= 0) is np.True_
