import numpy as np

from synthetmic.data.utils import (
    SynthetMicData,
    create_constant_volumes,
    create_periodicity,
    sample_random_seeds,
)


def _create_unit_domain(space_dim: int) -> tuple[np.ndarray, float]:
    match space_dim:
        case 2:
            return np.array([[0, 1], [0, 1]]), 1.0
        case 3:
            return np.array([[0, 1], [0, 1], [0, 1]]), 1.0
        case _:
            raise ValueError(
                f"invalid space_dim: {space_dim}; value must be either 2 or 3."
            )


def create_data_with_constant_volumes(
    space_dim: int = 2,
    n_grains: int = 1000,
    is_periodic: bool = False,
    random_state: int | None = None,
) -> SynthetMicData:
    """
    Create data in a unit square or cube. All target volumes will be the same and seeds will be
    randomly generated in the domain.

    Parameters
    ----------
    space_dim : int, optional
        The spce dimension of the domain. Value must be 2 or 3.

    n_grains : int, optional
        The number of grains in the unit cube. Value must be greater 0.

    is_period : bool, optional
        If True, the corresponding domain will be periodic in all directions.

    random_state : int or None, optional
        Pass an int to make the generated data reproducible.

    Returns
    -------
    synthetmic.data.utils.SyntyhetMicData
    """
    domain, domain_volume = _create_unit_domain(space_dim=space_dim)

    seeds = sample_random_seeds(
        domain=domain, n_grains=n_grains, random_state=random_state
    )
    volumes = create_constant_volumes(n_grains=n_grains, domain_volume=domain_volume)
    periodic = create_periodicity(space_dim=domain.shape[0], is_periodic=is_periodic)

    return SynthetMicData(
        seeds=seeds,
        volumes=volumes,
        domain=domain,
        periodic=periodic,
        init_weights=None,
    )


def create_data_with_lognormal_volumes(
    mean: float = 1.0,
    std: float = 0.35,
    space_dim: int = 2,
    n_grains: int = 1000,
    is_periodic: bool = False,
    random_state: int | None = None,
) -> SynthetMicData:
    """
    Create data in a unit square or cube. All target volumes will be distributed with
    lognormal distribution in the domain.

    Parameters
    ----------
    mean : float, optional
        The mean of the lognormal distribution.

    std : float, optional
        The standard deviation of the lognormal distribution.

    space_dim : int, optional
        The spce dimension of the domain. Value must be 2 or 3.

    n_grains : int, optional
        The number of grains in the unit cube. Value must be greater 0.

    is_period : bool, optional
        If True, the corresponding domain will be periodic in all directions.

    random_state : int or None, optional
        Pass an int to make the generated data reproducible.

    Returns
    -------
    synthetmic.data.utils.SyntyhetMicData
    """
    domain, domain_volume = _create_unit_domain(space_dim=space_dim)

    seeds = sample_random_seeds(
        domain=domain, n_grains=n_grains, random_state=random_state
    )

    ln_std = np.sqrt(np.log(1 + (std / mean) ** 2))
    ln_mean = -0.5 * ln_std**2 + np.log(mean)
    samples = np.random.lognormal(mean=ln_mean, sigma=ln_std, size=n_grains)
    scaling_factor = domain_volume / samples.sum()
    volumes = scaling_factor * samples

    periodic = create_periodicity(space_dim=domain.shape[0], is_periodic=is_periodic)

    return SynthetMicData(
        seeds=seeds,
        volumes=volumes,
        domain=domain,
        periodic=periodic,
        init_weights=None,
    )
