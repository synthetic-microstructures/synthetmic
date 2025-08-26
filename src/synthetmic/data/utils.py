from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SynthetMicData:
    seeds: np.ndarray
    volumes: np.ndarray
    domain: np.ndarray
    periodic: list[bool] | None
    init_weights: np.ndarray | None


def sample_random_seeds(
    domain: np.ndarray, n_grains: int, random_state: int | None = None
) -> np.ndarray:
    np.random.seed(seed=random_state)

    return np.column_stack(
        [np.random.uniform(low=d[0], high=d[1], size=n_grains) for d in domain]
    )


def create_periodicity(space_dim: int, is_periodic: bool) -> list[bool] | None:
    return [True] * space_dim if is_periodic else None


def create_constant_volumes(
    n_grains: int,
    domain_volume: float,
) -> np.ndarray:
    return (np.ones(n_grains) / n_grains) * domain_volume
