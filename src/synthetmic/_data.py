import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

from synthetmic.data.utils import sample_random_seeds
from synthetmic.typing import FloatArray, FloatSequence, IntArray


def map_positions(positions: FloatArray, boxsize: FloatSequence) -> FloatArray:
    return np.remainder(positions, np.asarray(boxsize))


def lift_positions(positions: FloatArray, weights: FloatArray) -> FloatArray:
    extra_coords = np.sqrt(weights.max() - weights)
    return np.column_stack((positions, extra_coords))


def lift_points(points: FloatArray) -> FloatArray:
    return np.column_stack((points, np.zeros(points.shape[0])))


def compute_non_periodic_size(max_coord: float, boxsize: FloatSequence) -> float:
    TAU = 1e-6

    return (
        np.sqrt(4 * np.sum(np.asarray(boxsize) ** 2) + max_coord**2) + max_coord + TAU
    )


def kdtree_closest_points(
    points: FloatArray,
    all_points: FloatArray,
    boxsize: FloatSequence | None = None,
    workers: int = -1,
) -> IntArray:
    tree = KDTree(data=all_points, boxsize=boxsize)
    _, indices = tree.query(points, workers=workers)
    return np.asarray(indices, dtype=np.int32)


def cdist_closest_points(
    points: FloatArray, all_points: FloatArray, weights: FloatArray | None = None
) -> FloatArray:

    squared_distances = cdist(points, all_points, metric="sqeuclidean")

    if weights is None:
        return np.argmin(squared_distances, axis=1)

    return np.argmin(squared_distances - weights, axis=1)


def calulate_rel_vols(n1: int, n2: int, r: int) -> FloatArray:
    vols = np.concatenate((np.ones(n1), r * np.ones(n2)))

    return vols / np.sum(vols)


def create_layered_points(
    n_layer_arr: FloatArray, r_layer_arr: FloatArray
) -> tuple[FloatArray, FloatArray]:
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


def create_subdomains(domain: FloatArray, n_subdomains: int) -> list[FloatArray]:
    x_min, x_max = domain[0]
    y_min, y_max = domain[1]

    x_divisions = np.linspace(x_min, x_max, n_subdomains + 1)

    subdomains = []
    for i in range(n_subdomains):
        subdomain = np.array([[x_divisions[i], x_divisions[i + 1]], [y_min, y_max]])
        subdomains.append(subdomain)

    return subdomains


def create_banded_points(
    subdomains: list[FloatArray],
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


def create_uniform_disc_points(
    center: tuple[float, float], radius: float, n_points: int
) -> FloatArray:
    r = radius * np.sqrt(np.random.random(n_points))
    theta = 2 * np.pi * np.random.random(n_points)
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)

    return np.column_stack([x, y])


def generate_discs(
    centers: list[tuple[float, float]], radius: float, n_points: int
) -> list[FloatArray]:
    all_points = []

    for center in centers:
        points = create_uniform_disc_points(center, radius, n_points)
        all_points.append(points)

    return all_points


def sample_points_outside_discs(
    rect: FloatArray,
    disc_centers: list[tuple[float, float]],
    disc_radius: float,
    n_points: int,
) -> FloatArray:
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
