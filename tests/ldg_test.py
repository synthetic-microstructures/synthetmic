from dataclasses import asdict

import numpy as np
import pytest
from pysdot import OptimalTransport

from synthetmic import LaguerreDiagramGenerator
from synthetmic.data.toy import create_data_with_constant_volumes
from synthetmic.data.utils import SynthetMicData
from synthetmic.utils import NotFittedError


@pytest.fixture
def fixture_data() -> SynthetMicData:
    return create_data_with_constant_volumes(space_dim=3)


def test_generator_params(fixture_data) -> None:
    with pytest.raises(ValueError):
        ldg = LaguerreDiagramGenerator(damp_param=2.3, verbose=False)
        ldg.fit(**asdict(fixture_data))

    with pytest.raises(ValueError):
        ldg = LaguerreDiagramGenerator(tol=0.0, verbose=False)
        ldg.fit(**asdict(fixture_data))

    with pytest.raises(ValueError):
        ldg = LaguerreDiagramGenerator(n_iter=-12, verbose=False)
        ldg.fit(**asdict(fixture_data))


def test_fit_args(fixture_data) -> None:
    with pytest.raises(TypeError):
        ldg = LaguerreDiagramGenerator(verbose=False)
        ldg.fit(seeds=fixture_data.seeds, volumes=1, domain=fixture_data.domain)

    with pytest.raises(ValueError):
        ldg = LaguerreDiagramGenerator(verbose=False)
        ldg.fit(
            seeds=fixture_data.seeds, volumes=np.zeros(50), domain=fixture_data.domain
        )


def test_valid_attributes(fixture_data) -> None:
    ldg = LaguerreDiagramGenerator(verbose=False)
    ldg.fit(**asdict(fixture_data))

    assert isinstance(ldg.optimal_transport_, OptimalTransport) is True
    assert isinstance(ldg.max_percentage_error_, float) is True
    assert isinstance(ldg.mean_percentage_error_, float) is True


def test_output_dim(fixture_data) -> None:
    ldg = LaguerreDiagramGenerator(verbose=False)
    ldg.fit(**asdict(fixture_data))

    assert ldg.get_centroids().shape == fixture_data.seeds.shape
    assert ldg.get_fitted_volumes().shape == fixture_data.volumes.shape


def test_ensure_fit() -> None:
    with pytest.raises(NotFittedError):
        ldg = LaguerreDiagramGenerator()
        ldg.get_centroids()


def test_get_vertices_2d():
    n_grains = 2

    seeds = np.array(
        [
            [
                0.5,
                0.5,
            ],
            [
                0.5,
                0.75,
            ],
        ]
    )
    volumes = np.ones(n_grains) / n_grains
    domain = np.array([[0, 1], [0, 1]])

    ldg = LaguerreDiagramGenerator(n_iter=0)
    ldg.fit(seeds=seeds, volumes=volumes, domain=domain)

    res = ldg.get_vertices()

    assert len(res) == n_grains
    assert sum(len(v) for v in res.values()) == 8


def test_get_vertices_3d():
    n_grains = 2

    seeds = np.array(
        [
            [
                0.5,
                0.5,
                0.5,
            ],
            [
                0.5,
                0.5,
                0.75,
            ],
        ]
    )
    volumes = np.ones(n_grains) / n_grains
    domain = np.array([[0, 1], [0, 1], [0, 1]])

    ldg = LaguerreDiagramGenerator(n_iter=0)
    ldg.fit(seeds=seeds, volumes=volumes, domain=domain)

    res = ldg.get_vertices()

    assert len(res) == n_grains

    sum_vertices = 0
    for faces in res.values():
        for vertices in faces:
            sum_vertices += len(vertices)

    assert sum_vertices == 48
