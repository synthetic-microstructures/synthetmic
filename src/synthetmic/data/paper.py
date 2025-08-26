import math
from enum import StrEnum, auto

import numpy as np

from synthetmic.data.utils import (
    SynthetMicData,
    create_periodicity,
    sample_random_seeds,
)


class _Initializer(StrEnum):
    RANDOM = auto()
    BANDED = auto()
    CLUSTERED = auto()
    MIXED_BANDED_AND_RANDOM = auto()


class _Gradient(StrEnum):
    INCREASING = auto()
    LARGE_AT_MIDDLE = auto()


def _calulate_rel_vols(n1: int, n2: int, r: int) -> np.ndarray:
    vols = np.concatenate((np.ones(n1), r * np.ones(n2)))

    return vols / np.sum(vols)


def _create_layered_points(
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


def _create_subdomains(domain: np.ndarray, n_subdomains: int) -> list[np.ndarray]:
    x_min, x_max = domain[0]
    y_min, y_max = domain[1]

    x_divisions = np.linspace(x_min, x_max, n_subdomains + 1)

    subdomains = []
    for i in range(n_subdomains):
        subdomain = np.array([[x_divisions[i], x_divisions[i + 1]], [y_min, y_max]])
        subdomains.append(subdomain)

    return subdomains


def _create_banded_points(
    subdomains: list[np.ndarray],
    n_points_small: int,
    n_points_large: int,
    volume_frac: float,
) -> tuple[list, list]:
    X = []
    y = []
    for i, subdomain in enumerate(subdomains):
        if i % 2 == 0:
            X.extend(sample_random_seeds(subdomain, n_points_large).tolist())
            y.extend([20 * volume_frac] * n_points_large)
        else:
            X.extend(sample_random_seeds(subdomain, n_points_small).tolist())
            y.extend([volume_frac] * n_points_small)

    return X, y


def _create_uniform_disc_points(
    center: tuple[float, float], radius: float, n_points: int
) -> np.ndarray:
    r = radius * np.sqrt(np.random.random(n_points))
    theta = 2 * np.pi * np.random.random(n_points)
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)

    return np.column_stack([x, y])


def _generate_discs(
    centers: list[tuple[float, float]], radius: float, n_points: int
) -> list[np.ndarray]:
    all_points = []

    for center in centers:
        points = _create_uniform_disc_points(center, radius, n_points)
        all_points.append(points)

    return all_points


def _sample_points_outside_discs(
    rect: np.ndarray,
    disc_centers: list[tuple[float, float]],
    disc_radius: float,
    n_points: int,
) -> np.ndarray:
    accepted = []
    batch_size = max(1000, n_points * 2)
    radius_sq = disc_radius**2
    centers_array = np.array(disc_centers)

    while len(accepted) < n_points:
        points = sample_random_seeds(rect, batch_size)

        dists_sq = np.sum(
            (points[:, np.newaxis, :] - centers_array[np.newaxis, :, :]) ** 2, axis=2
        )

        outside_mask = np.all(dists_sq > radius_sq, axis=1)

        valid_points = points[outside_mask]
        accepted.extend(valid_points.tolist())

    accepted = np.array(accepted[:n_points])
    return accepted


def create_example3_data(is_periodic: bool) -> SynthetMicData:
    """
    Create data for generating figures 1 and 2 of the following paper:

    Bourne, D.P., Kok, P.J.J., Roper, S.M. & Spanjer, W.D.T. (2020)
    Laguerre tessellations and polycrystalline microstructures:
    A fast algorithm for generating grains of given volumes,
    Philosophical Magazine, 100, 2677-2707. Link:
    https://www.tandfonline.com/doi/full/10.1080/14786435.2020.1790053

    Parameters
    ----------
    is_periodic : bool
        If True, the underlying domain will be periodic in all
        directions.

    Returns
    -------
    synthetmic.data.utils.SynthetMicData
    """

    domain = np.array([[0, 1], [0, 1]])

    N_GRAINS = 50
    AREA_FRAC = 1 / 185

    y = np.zeros(N_GRAINS)
    y[:35] = AREA_FRAC
    y[35:] = 10 * AREA_FRAC

    X = sample_random_seeds(domain, N_GRAINS)

    periodic = create_periodicity(domain.shape[0], is_periodic)

    return SynthetMicData(
        seeds=X, volumes=y, domain=domain, periodic=periodic, init_weights=None
    )


def create_example4_data(initializer: str, is_periodic: bool) -> SynthetMicData:
    """
    Create data for generating figure 4 of the following paper:

    Bourne, D.P., Kok, P.J.J., Roper, S.M. & Spanjer, W.D.T. (2020)
    Laguerre tessellations and polycrystalline microstructures:
    A fast algorithm for generating grains of given volumes,
    Philosophical Magazine, 100, 2677-2707. Link:
    https://www.tandfonline.com/doi/full/10.1080/14786435.2020.1790053

    Parameters
    ----------
    initializer : str
        The initializer for the generated data. Value must be one of
        ["random", "banded", "clustered", "mixed_banded_and_random"].

    is_periodic : bool
        If True, the underlying domain will be periodic in all
        directions.

    Returns
    -------
    synthetmic.data.utils.SynthetMicData
    """

    AREA_FRAC = 1 / 800
    N1 = 800
    N2 = 200

    domain = np.array([[0, 3], [0, 2]])

    match initializer:
        case _Initializer.RANDOM:
            y = np.zeros(N1 + N2)
            y[:N1] = AREA_FRAC
            y[N1:] = 20 * AREA_FRAC

            X = sample_random_seeds(domain, N1 + N2)

        case _Initializer.BANDED:
            subdomains = _create_subdomains(domain=domain, n_subdomains=7)

            X, y = _create_banded_points(
                subdomains=subdomains,
                n_points_small=math.ceil(N1 / 3),
                n_points_large=math.ceil(N2 / 4),
                volume_frac=AREA_FRAC,
            )
            X = np.array(X)[: N1 + N2]
            y = np.array(y)[: N1 + N2]

        case _Initializer.CLUSTERED:
            X = []
            y = []

            DISC_CENTERS = [(0.6, 0.6), (2.4, 0.6), (1.5, 1.5)]
            DISC_RADIUS = 0.2

            in_disc_points = _generate_discs(
                centers=DISC_CENTERS,
                radius=DISC_RADIUS,
                n_points=math.ceil(N1 / 3),
            )
            for point in in_disc_points:
                X.extend(point.tolist())
                y.extend([AREA_FRAC] * point.shape[0])

            # remove the last point; we have more than N1 points now
            X = X[:-1]
            y = y[:-1]

            # generate points outside circular discs
            out_disc_points = _sample_points_outside_discs(
                rect=domain,
                disc_centers=DISC_CENTERS,
                disc_radius=DISC_RADIUS,
                n_points=N2,
            )
            X.extend(out_disc_points.tolist())
            y.extend([20 * AREA_FRAC] * out_disc_points.shape[0])

            X = np.array(X)
            y = np.array(y)

        case _Initializer.MIXED_BANDED_AND_RANDOM:
            subdomains = _create_subdomains(domain=domain, n_subdomains=7)

            n_points_small = math.ceil(N1 / 4)
            X, y = _create_banded_points(
                subdomains=subdomains,
                n_points_small=n_points_small,
                n_points_large=math.ceil(N2 / 4),
                volume_frac=AREA_FRAC,
            )

            # add random N2 / 4 random points for the small cells
            X.extend(sample_random_seeds(domain, n_points_small).tolist())
            y.extend([AREA_FRAC] * n_points_small)

            X = np.array(X)
            y = np.array(y)

        case _:
            raise ValueError(
                f"initializer must be one of {', '.join(_Initializer)}; but {initializer} was given."
            )

    periodic = create_periodicity(domain.shape[0], is_periodic)

    return SynthetMicData(
        seeds=X, volumes=y, domain=domain, periodic=periodic, init_weights=None
    )


def create_example4b_data(gradient: str, is_periodic: bool) -> SynthetMicData:
    """
    Create data for generating figure 5 of the following paper:

    Bourne, D.P., Kok, P.J.J., Roper, S.M. & Spanjer, W.D.T. (2020)
    Laguerre tessellations and polycrystalline microstructures:
    A fast algorithm for generating grains of given volumes,
    Philosophical Magazine, 100, 2677-2707. Link:
    https://www.tandfonline.com/doi/full/10.1080/14786435.2020.1790053

    Parameters
    ----------
    gradient: str
        The gradient type for the generated data. Value must be one of
        ["increasing", "large_at_middle"].

    is_periodic : bool
        If True, the underlying domain will be periodic in all
        directions.

    Returns
    -------
    synthetmic.data.utils.SynthetMicData
    """

    if gradient not in _Gradient:
        raise ValueError(
            f"gradient must be one of {', '.join(_Gradient)}; but {gradient} was given."
        )

    N = 1000
    MAX_RATIO = 100

    domain = np.array([[0, 3], [0, 2]])
    total_area = (domain[0][1] - domain[0][0]) * (domain[1][1] - domain[1][0])

    uniform_areas = 1 + np.random.uniform(0, 1, N) * (MAX_RATIO - 1)
    # scale so total area equals target
    scaling_factor = total_area / np.sum(uniform_areas)
    y = scaling_factor * uniform_areas

    y = np.sort(y)

    x_coord = np.linspace(start=domain[0][0], stop=domain[0][1], num=N)
    y_coord = np.random.uniform(low=domain[1][0], high=domain[1][1], size=N)
    X = np.column_stack((x_coord, y_coord))

    periodic = create_periodicity(domain.shape[0], is_periodic)

    if gradient == _Gradient.INCREASING:
        return SynthetMicData(
            seeds=X, volumes=y, domain=domain, periodic=periodic, init_weights=None
        )

    y = np.concatenate(
        (y[len(y) % 2 :: 2], y[::-2])
    )  # ensure the large areas fall in the middle and small areas fall at the ends

    return SynthetMicData(
        seeds=X, volumes=y, domain=domain, periodic=periodic, init_weights=None
    )


def create_example5p1_data(n_grains: int, r: int, is_periodic: bool) -> SynthetMicData:
    """
    Create data for generating figure 6 of the following paper:

    Bourne, D.P., Kok, P.J.J., Roper, S.M. & Spanjer, W.D.T. (2020)
    Laguerre tessellations and polycrystalline microstructures:
    A fast algorithm for generating grains of given volumes,
    Philosophical Magazine, 100, 2677-2707. Link:
    https://www.tandfonline.com/doi/full/10.1080/14786435.2020.1790053

    Parameters
    ----------
    n_grains : int
        The number of grains in the microstructure.

    r : int
        The volume ratio of the idealized dual-phase
        microstructure. If the total volume of the first phase
        is v, then that of the second phase will be r*v.

    is_periodic : bool
        If True, the underlying domain will be periodic in all
        directions.

    Returns
    -------
    synthetmic.data.utils.SynthetMicData
    """

    L1, L2, L3 = 100, 100, 100

    domain = np.array([[0, L1], [0, L2], [0, L3]])
    domain_vol = np.prod(domain[:, 1] - domain[:, 0])

    periodic = create_periodicity(domain.shape[0], is_periodic)

    X = sample_random_seeds(domain, n_grains)
    target_vols = domain_vol * _calulate_rel_vols(n_grains // 2, n_grains // 2, r)

    return SynthetMicData(
        seeds=X,
        volumes=target_vols,
        domain=domain,
        periodic=periodic,
        init_weights=None,
    )


def create_example5p4_data(is_periodic: bool) -> SynthetMicData:
    """
    Create data for generating figure 12 of the following paper:

    Bourne, D.P., Kok, P.J.J., Roper, S.M. & Spanjer, W.D.T. (2020)
    Laguerre tessellations and polycrystalline microstructures:
    A fast algorithm for generating grains of given volumes,
    Philosophical Magazine, 100, 2677-2707. Link:
    https://www.tandfonline.com/doi/full/10.1080/14786435.2020.1790053

    Parameters
    ----------
    is_periodic : bool
        If True, the underlying domain will be periodic in all
        directions.

    Returns
    -------
    synthetmic.data.utils.SynthetMicData
    """

    L1 = 2
    L2 = 3
    L3 = 2

    domain = np.array([[0, L1], [0, L2], [0, L3]])
    domain_vol = np.prod(domain[:, 1] - domain[:, 0])

    periodic = create_periodicity(domain.shape[0], is_periodic)

    # Number of grains and relative vols
    n_layer_arr = np.array([1000, 8000, 1000])
    r_layer_arr = np.array([1, 0.05, 1])

    # Target volumes and initial seed locations
    X, rel_vols = _create_layered_points(n_layer_arr, r_layer_arr)
    X = X @ np.diag([L1, L2, L3])
    y = rel_vols * domain_vol

    return SynthetMicData(
        seeds=X, volumes=y, domain=domain, periodic=periodic, init_weights=None
    )


def create_example5p5_data(is_periodic: bool) -> SynthetMicData:
    """
    Create data for generating figure 13 of the following paper:

    Bourne, D.P., Kok, P.J.J., Roper, S.M. & Spanjer, W.D.T. (2020)
    Laguerre tessellations and polycrystalline microstructures:
    A fast algorithm for generating grains of given volumes,
    Philosophical Magazine, 100, 2677-2707. Link:
    https://www.tandfonline.com/doi/full/10.1080/14786435.2020.1790053

    Parameters
    ----------
    is_periodic : bool
        If True, the underlying domain will be periodic in all
        directions.

    Returns
    -------
    synthetmic.data.utils.SynthetMicData
    """

    L1 = 2
    L2 = 2
    L3 = 2

    domain = np.array([[0, L1], [0, L2], [0, L3]])
    domain_vol = np.prod(domain[:, 1] - domain[:, 0])

    # Set periodicity in the three directions
    periodic = create_periodicity(domain.shape[0], is_periodic)

    # Number of grains
    N = 10000

    # Random initial seed locations
    X = sample_random_seeds(domain, N)

    # Log-normal target volumes
    DESIRED_LN_MEAN = 1
    DESIRED_LN_STD = 0.35

    sigma = np.sqrt(np.log(1 + (DESIRED_LN_STD / DESIRED_LN_MEAN) ** 2))
    mu = -0.5 * sigma**2 + np.log(DESIRED_LN_MEAN)

    radii = np.random.lognormal(mu, sigma, N)
    target_vols = radii**3
    rel_vols = target_vols / np.sum(target_vols)

    target_vols = rel_vols * domain_vol

    return SynthetMicData(
        seeds=X,
        volumes=target_vols,
        domain=domain,
        periodic=periodic,
        init_weights=None,
    )
