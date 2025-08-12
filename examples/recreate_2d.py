import math
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np

from examples.utils import (
    Gradient,
    Initializer,
    RecreateData,
    create_periodicity,
    get_rcparams,
    sample_random_seeds,
    set_fig_size,
)
from synthetmic import LaguerreDiagramGenerator
from synthetmic.plot import plot_2dcells_as_matplotlib_fig

plt.rcParams.update(get_rcparams())


def create_subdomains(domain: np.ndarray, n_subdomains: int) -> list[np.ndarray]:
    x_min, x_max = domain[0]
    y_min, y_max = domain[1]

    x_divisions = np.linspace(x_min, x_max, n_subdomains + 1)

    subdomains = []
    for i in range(n_subdomains):
        subdomain = np.array([[x_divisions[i], x_divisions[i + 1]], [y_min, y_max]])
        subdomains.append(subdomain)

    return subdomains


def create_banded_points(
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


def create_uniform_disc_points(
    center: tuple[float, float], radius: float, n_points: int
) -> np.ndarray:
    r = radius * np.sqrt(np.random.random(n_points))
    theta = 2 * np.pi * np.random.random(n_points)
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)

    return np.column_stack([x, y])


def generate_discs(
    centers: list[tuple[float, float]], radius: float, n_points: int
) -> list[np.ndarray]:
    all_points = []

    for center in centers:
        points = create_uniform_disc_points(center, radius, n_points)
        all_points.append(points)

    return all_points


def sample_points_outside_discs(
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


def create_example3_data(is_periodic: bool) -> RecreateData:
    domain = np.array([[0, 1], [0, 1]])

    N_GRAINS = 50
    AREA_FRAC = 1 / 185

    y = np.zeros(N_GRAINS)
    y[:35] = AREA_FRAC
    y[35:] = 10 * AREA_FRAC

    X = sample_random_seeds(domain, N_GRAINS)

    periodic = create_periodicity(domain.shape[0], is_periodic)

    return RecreateData(
        seeds=X, volumes=y, domain=domain, periodic=periodic, init_weights=None
    )


def create_example4_data(initializer: str, is_periodic: bool) -> RecreateData:
    AREA_FRAC = 1 / 800
    N1 = 800
    N2 = 200

    domain = np.array([[0, 3], [0, 2]])

    match initializer:
        case Initializer.RANDOM:
            y = np.zeros(N1 + N2)
            y[:N1] = AREA_FRAC
            y[N1:] = 20 * AREA_FRAC

            X = sample_random_seeds(domain, N1 + N2)

        case Initializer.BANDED:
            subdomains = create_subdomains(domain=domain, n_subdomains=7)

            X, y = create_banded_points(
                subdomains=subdomains,
                n_points_small=math.ceil(N1 / 3),
                n_points_large=math.ceil(N2 / 4),
                volume_frac=AREA_FRAC,
            )
            X = np.array(X)[: N1 + N2]
            y = np.array(y)[: N1 + N2]

        case Initializer.CLUSTERED:
            X = []
            y = []

            DISC_CENTERS = [(0.6, 0.6), (2.4, 0.6), (1.5, 1.5)]
            DISC_RADIUS = 0.2

            in_disc_points = generate_discs(
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
            out_disc_points = sample_points_outside_discs(
                rect=domain,
                disc_centers=DISC_CENTERS,
                disc_radius=DISC_RADIUS,
                n_points=N2,
            )
            X.extend(out_disc_points.tolist())
            y.extend([20 * AREA_FRAC] * out_disc_points.shape[0])

            X = np.array(X)
            y = np.array(y)

        case Initializer.MIXED_BANDED_AND_RANDOM:
            subdomains = create_subdomains(domain=domain, n_subdomains=7)

            n_points_small = math.ceil(N1 / 4)
            X, y = create_banded_points(
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
            raise ValueError(f"initializer must be one of {', '.join(Initializer)}")

    periodic = create_periodicity(domain.shape[0], is_periodic)

    return RecreateData(
        seeds=X, volumes=y, domain=domain, periodic=periodic, init_weights=None
    )


def create_example4b_data(gradient: str, is_periodic: bool) -> RecreateData:
    if gradient not in Gradient:
        raise ValueError(f"gradient must be one of {', '.join(Gradient)}")

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

    if gradient == Gradient.INCREASING:
        return RecreateData(
            seeds=X, volumes=y, domain=domain, periodic=periodic, init_weights=None
        )

    y = np.concatenate(
        (y[len(y) % 2 :: 2], y[::-2])
    )  # ensure the large areas fall in the middle and small areas fall at the ends

    return RecreateData(
        seeds=X, volumes=y, domain=domain, periodic=periodic, init_weights=None
    )


def recreate_fig1(save_path: str, is_periodic: bool) -> None:
    TOL = 1e-2
    N_ITER = 0

    data = create_example3_data(is_periodic)

    ldg = LaguerreDiagramGenerator(tol=TOL, n_iter=N_ITER)
    ldg.fit(**asdict(data))

    _, ax = plt.subplots(figsize=set_fig_size())
    ax = plot_2dcells_as_matplotlib_fig(
        generator=ldg, ax=ax, title=None, colorby=data.volumes
    )

    positions = ldg.get_positions()
    ax.scatter(positions[:, 0], positions[:, 1], color="black", s=5)

    ax.axis("off")
    ax.set_aspect("equal")

    plt.savefig(save_path, bbox_inches="tight")

    return None


def recreate_fig2(save_path: str, is_periodic: bool) -> None:
    data = create_example3_data(is_periodic)

    ITERATIONS = (1, 2, 3, 4, 5, 10, 25, 50, 100)
    TOL = 1e-2

    grid = len(ITERATIONS) // 3
    fig = plt.figure(figsize=set_fig_size(subplots=(grid, grid), adjust_height=0.5))
    tags = ("a", "b", "c", "d", "e", "f", "g", "h", "i")

    for i, n_iter in enumerate(ITERATIONS):
        ax = fig.add_subplot(grid, grid, i + 1)
        ax.text(
            x=-0.05,
            y=1.05,
            s=r"\bf \large {}".format(tags[i]),
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
        )

        ldg = LaguerreDiagramGenerator(tol=TOL, n_iter=n_iter)
        ldg.fit(**asdict(data))

        ax = plot_2dcells_as_matplotlib_fig(
            generator=ldg,
            ax=ax,
            title=f"$k={n_iter}$",
            colorby=data.volumes,
        )

        centroids = ldg.get_centroids()
        ax.scatter(centroids[:, 0], centroids[:, 1], color="black", s=5)

        ax.axis("off")
        ax.set_aspect("equal")

    plt.savefig(save_path, bbox_inches="tight")

    return None


def recreate_fig4(save_path: str, is_periodic: bool) -> None:
    N_ITER = 20
    TOL = 1e-2

    grid = len(Initializer) // 2
    fig = plt.figure(figsize=set_fig_size(subplots=(grid, grid), adjust_height=0.1))
    tags = ("a", "b", "c", "d")

    for i, init in enumerate(Initializer):
        ax = fig.add_subplot(grid, grid, i + 1)

        ax.text(
            x=-0.05,
            y=1.05,
            s=r"\bf \large {}".format(tags[i]),
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
        )

        data = create_example4_data(init, is_periodic)

        ldg = LaguerreDiagramGenerator(tol=TOL, n_iter=N_ITER)
        ldg.fit(**asdict(data))

        plot_2dcells_as_matplotlib_fig(
            generator=ldg,
            ax=ax,
            title=f"{init.replace('_', ' ').capitalize()}",
            colorby=data.volumes,
        )

        ax.axis("off")
        ax.set_aspect("equal")

    plt.savefig(save_path, bbox_inches="tight")

    return None


def recreate_fig5(save_path: str, is_periodic: bool) -> None:
    N_ITER = 20
    TOL = 1e-2

    grid = len(Gradient)
    fig = plt.figure(figsize=set_fig_size(subplots=(grid, grid)))
    tags = ("a", "b")

    for i, gradient in enumerate(Gradient):
        ax = fig.add_subplot(grid, grid, i + 1)

        ax.text(
            x=-0.05,
            y=1.05,
            s=r"\bf \large {}".format(tags[i]),
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
        )

        data = create_example4b_data(gradient, is_periodic)

        ldg = LaguerreDiagramGenerator(tol=TOL, n_iter=N_ITER)
        ldg.fit(**asdict(data))

        plot_2dcells_as_matplotlib_fig(
            generator=ldg,
            ax=ax,
            title=f"{gradient.replace('_', ' ').capitalize()}",
            colorby=data.volumes,
        )

        ax.axis("off")
        ax.set_aspect("equal")

    plt.savefig(save_path, bbox_inches="tight")

    return None
