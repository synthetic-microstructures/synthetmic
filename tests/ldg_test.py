from dataclasses import asdict

import numpy as np
import pytest
from pysdot import OptimalTransport

from synthetmic import LaguerreDiagramGenerator
from synthetmic._internal._errors import _NotFittedError
from synthetmic.data import toy, utils
from synthetmic.data.toy import create_data_with_constant_volumes
from synthetmic.data.utils import SynthetMicData


@pytest.fixture
def const_vol_data() -> SynthetMicData:
    return create_data_with_constant_volumes(space_dim=3)


def test_generator_params(const_vol_data) -> None:
    with pytest.raises(ValueError):
        ldg = LaguerreDiagramGenerator(damp_param=2.3, verbose=False)
        ldg.fit(**asdict(const_vol_data))

    with pytest.raises(ValueError):
        ldg = LaguerreDiagramGenerator(tol=0.0, verbose=False)
        ldg.fit(**asdict(const_vol_data))

    with pytest.raises(ValueError):
        ldg = LaguerreDiagramGenerator(n_iter=-12, verbose=False)
        ldg.fit(**asdict(const_vol_data))


def test_fit_args(const_vol_data) -> None:
    with pytest.raises(TypeError):
        ldg = LaguerreDiagramGenerator(verbose=False)
        ldg.fit(seeds=const_vol_data.seeds, volumes=1, domain=const_vol_data.domain)

    with pytest.raises(ValueError):
        ldg = LaguerreDiagramGenerator(verbose=False)
        ldg.fit(
            seeds=const_vol_data.seeds,
            volumes=np.zeros(50),
            domain=const_vol_data.domain,
        )


def test_valid_attributes(const_vol_data) -> None:
    ldg = LaguerreDiagramGenerator(tol=1, damp_param=1, verbose=False)
    ldg.fit(**asdict(const_vol_data))

    assert isinstance(ldg.optimal_transport_, OptimalTransport) is True
    assert isinstance(ldg.max_percentage_error_, float) is True
    assert isinstance(ldg.mean_percentage_error_, float) is True


def test_output_dim(const_vol_data) -> None:
    ldg = LaguerreDiagramGenerator(tol=1, damp_param=1, verbose=False)
    ldg.fit(**asdict(const_vol_data))

    assert ldg.get_centroids().shape == const_vol_data.seeds.shape
    assert ldg.get_fitted_volumes().shape == const_vol_data.volumes.shape


def test_ensure_fit() -> None:
    with pytest.raises(_NotFittedError):
        ldg = LaguerreDiagramGenerator()
        ldg.get_centroids()


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
            verbose=False,
        )
        generator.fit(seeds=seeds, volumes=volumes, domain=domain, periodic=periodic)

        counts = [len(k) for k in generator.get_vertices().values()]
        results.append(counts)

        print(f"periodic: {periodic}, verts counts: {counts}")

    assert results[0] == results[1]

    return None
