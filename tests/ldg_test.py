import numpy as np
import pytest
from pysdot import PowerDiagram

from synthetmic import LaguerreDiagramGenerator
from synthetmic.data import toy, utils
from synthetmic.data.toy import create_data_with_constant_volumes
from synthetmic.data.utils import DiagramConfig


@pytest.fixture
def config() -> DiagramConfig:
    return create_data_with_constant_volumes(space_dim=3)


def test_generator_params(config) -> None:
    with pytest.raises(ValueError):
        ldg = LaguerreDiagramGenerator(damp_param=2.3)
        ldg.fit(config)

    with pytest.raises(ValueError):
        ldg = LaguerreDiagramGenerator(tol=0.0)
        ldg.fit(config)

    with pytest.raises(ValueError):
        ldg = LaguerreDiagramGenerator(n_iter=-12)
        ldg.fit(config)


def test_valid_attributes(config) -> None:
    ldg = LaguerreDiagramGenerator(tol=1, damp_param=1)
    ldg.fit(config)

    assert isinstance(ldg.pd_, PowerDiagram) is True
    assert isinstance(ldg.max_percentage_error_, float) is True
    assert isinstance(ldg.mean_percentage_error_, float) is True


def test_output_dim(config) -> None:
    ldg = LaguerreDiagramGenerator(tol=1, damp_param=1)
    ldg.fit(config)

    assert ldg.get_centroids().shape == config.seeds.shape
    assert ldg.get_fitted_volumes().shape == config.volumes.shape


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
    domain, domain_volume = toy.create_unit_domain(space_dim=space_dim)
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
        )
        generator.fit(
            DiagramConfig(
                seeds=seeds, volumes=volumes, domain=domain, periodic=periodic
            )
        )

        counts = [len(k) for k in generator.get_vertices().values()]
        results.append(counts)

        print(f"periodic: {periodic}, verts counts: {counts}")

    assert results[0] == results[1]
