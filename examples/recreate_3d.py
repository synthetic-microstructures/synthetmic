from dataclasses import asdict

import numpy as np

from examples.utils import RecreateData
from synthetmic import LaguerreDiagramGenerator
from synthetmic.plot import plot_cells3d


def create_layered_points(
    n_layer_arr: np.ndarray, r_layer_arr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(n_layer_arr, list):
        n_layer_arr = np.array(n_layer_arr)

    if isinstance(n_layer_arr, list):
        r_layer_arr = np.array(r_layer_arr)

    # Relative volumes of the grains (relative to the volume of the box)
    rel_vols = np.repeat(r_layer_arr, n_layer_arr)
    total_vol: float = (n_layer_arr * r_layer_arr).sum()
    rel_vols = rel_vols / total_vol

    # Thickness of each layer
    t_layer_arr = n_layer_arr * r_layer_arr / total_vol
    o_layer_arr = t_layer_arr.cumsum()
    o_layer_arr = np.hstack(([0], o_layer_arr[:-1]))

    # Deduce the z-coord
    z_coord = []
    for i, n in enumerate(n_layer_arr):
        z_coord.extend((o_layer_arr[i] + t_layer_arr[i] * np.random.rand(n)).tolist())

    DIM = 3
    z_coord = np.array(z_coord)
    xy_coord = np.random.rand(n_layer_arr.sum(), DIM - 1)

    return np.column_stack((xy_coord, z_coord)), rel_vols


def create_example5p4_data(is_periodic: bool) -> RecreateData:
    # Define box size
    L1 = 2
    L2 = 3
    L3 = 2

    # Define box
    domain = np.array([[0, L1], [0, L2], [0, L3]])
    domain_vol = np.prod(domain[:, 1] - domain[:, 0])

    # Set periodicity in the three directions
    periodic = [True] * 3 if is_periodic else None

    # Number of grains and relative vols
    n_layer_arr = np.array([1000, 8000, 1000])
    r_layer_arr = np.array([1, 0.05, 1])

    # Target volumes and initial seed locations
    X, rel_vols = create_layered_points(n_layer_arr, r_layer_arr)
    X = X @ np.diag([L1, L2, L3])
    y = rel_vols * domain_vol

    return RecreateData(
        seeds=X, volumes=y, domain=domain, periodic=periodic, init_weights=None
    )


def create_example5p5_data(is_periodic: bool) -> RecreateData:
    # Define box size
    L1 = 2
    L2 = 2
    L3 = 2

    # Define box
    domain = np.array([[0, L1], [0, L2], [0, L3]])
    domain_vol = np.prod(domain[:, 1] - domain[:, 0])

    # Set periodicity in the three directions
    periodic = [True] * 3 if is_periodic else None

    # Number of grains
    N = 10000

    # Random initial seed locations
    X = np.random.rand(N, 3)
    X = X @ np.diag([L1, L2, L3])

    # Log-normal target volumes
    DESIRED_LN_MEAN = 1
    DESIRED_LN_STD = 0.35

    sigma = np.sqrt(np.log(1 + (DESIRED_LN_STD / DESIRED_LN_MEAN) ** 2))
    mu = -0.5 * sigma**2 + np.log(DESIRED_LN_MEAN)

    radii = np.random.lognormal(mu, sigma, N)
    target_vols = radii**3
    rel_vols = target_vols / np.sum(target_vols)

    target_vols = rel_vols * domain_vol

    return RecreateData(
        seeds=X,
        volumes=target_vols,
        domain=domain,
        periodic=periodic,
        init_weights=None,
    )


def recreate_fig12(save_path: str, is_periodic: bool, interactive: bool) -> None:
    TOL = 1.0
    N_ITER = 20

    data = create_example5p4_data(is_periodic=is_periodic)

    ldg = LaguerreDiagramGenerator(tol=TOL, n_iter=N_ITER)
    ldg.fit(**asdict(data))

    plot_cells3d(
        ldg.optimal_transport_,
        titlestr=None,
        colorby=data.volumes,
        save_path=save_path,
        interactive=interactive,
    )

    return None


def recreate_fig13(save_path: str, is_periodic: bool, interactive: bool) -> None:
    TOL = 1.0
    N_ITER = 5

    data = create_example5p5_data(is_periodic=is_periodic)

    ldg = LaguerreDiagramGenerator(tol=TOL, n_iter=N_ITER)
    ldg.fit(**asdict(data))

    plot_cells3d(
        ldg.optimal_transport_,
        titlestr=None,
        colorby=data.volumes,
        save_path=save_path,
        interactive=interactive,
    )

    return None
