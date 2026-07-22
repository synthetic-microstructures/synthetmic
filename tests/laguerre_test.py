import numpy as np
import pytest
from pysdot import PowerDiagram

from synthetmic import DiagramConfig, LaguerreDiagramGenerator, LaguerreEvent
from synthetmic.data import toy, utils
from synthetmic.data.toy import create_data_with_constant_volumes


@pytest.fixture
def config() -> DiagramConfig:
    return create_data_with_constant_volumes(space_dim=3)


@pytest.mark.parametrize("n_iter", (10, 20, 30, 40, 50))
@pytest.mark.parametrize("tol", (1.0, 3.0, 5.0))
def test_max_volume_percentage_error_against_tol(
    n_iter: int, tol: float, config: DiagramConfig
) -> None:
    errors = np.zeros(n_iter, dtype=int)

    def callback(e: LaguerreEvent) -> None:
        errors[e.iteration - 1] = e.max_percentage_error

    ldg = LaguerreDiagramGenerator(tol=tol)
    ldg.fit(config, callback=callback)

    assert np.all(errors <= tol) is np.True_


@pytest.mark.parametrize("tol", (-1.1, 0.0))
def test_tol_error(tol: float, config: DiagramConfig) -> None:
    with pytest.raises(ValueError):
        ldg = LaguerreDiagramGenerator(tol=tol)
        ldg.fit(config)


@pytest.mark.parametrize("n_iter", (-10, 20.5))
def test_n_iter_error(n_iter: int, config: DiagramConfig) -> None:
    with pytest.raises((ValueError, TypeError)):
        ldg = LaguerreDiagramGenerator(n_iter=n_iter)
        ldg.fit(config)


@pytest.mark.parametrize("damp_param", (-1, 2.3))
def test_damp_param_error(damp_param: float, config: DiagramConfig) -> None:
    with pytest.raises(ValueError):
        ldg = LaguerreDiagramGenerator(damp_param=damp_param)
        ldg.fit(config)


def test_attribute_types(config: DiagramConfig) -> None:
    ldg = LaguerreDiagramGenerator(tol=1, damp_param=1)
    ldg.fit(config)

    assert isinstance(ldg.pd_, PowerDiagram) is True
    assert isinstance(ldg.max_percentage_error_, float) is True
    assert isinstance(ldg.mean_percentage_error_, float) is True
    assert isinstance(ldg.centroid_error_norm_, float) is True
    assert isinstance(ldg.n_grains_in_, int) is True
    assert isinstance(ldg.space_dim_in_, int) is True


def test_output_dim(config: DiagramConfig) -> None:
    ldg = LaguerreDiagramGenerator(tol=1, damp_param=1)
    ldg.fit(config)

    assert ldg.get_centroids().shape == config.seeds.shape
    assert ldg.get_positions().shape == config.seeds.shape
    assert ldg.get_fitted_volumes().shape == config.volumes.shape
    assert ldg.get_orientations().shape == (ldg.n_grains_in_, 4)


@pytest.mark.parametrize(
    "seeds, expected",
    [
        (np.array([[0.5, 0.5], [0.5, 0.75]]), 8),
        (np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.75]]), 48),
    ],
)
def test_get_vertices(seeds: np.ndarray, expected: int) -> None:
    n_grains, space_dim = seeds.shape
    domain, domain_volume = toy.create_unit_domain(space_dim=space_dim)
    volumes = utils.create_constant_volumes(
        n_grains=n_grains, domain_volume=domain_volume
    )

    ldg = LaguerreDiagramGenerator(n_iter=0)
    ldg.fit(DiagramConfig(seeds=seeds, volumes=volumes, domain=domain))

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
