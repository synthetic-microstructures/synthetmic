from dataclasses import asdict

import matplotlib.pyplot as plt

from examples.utils import (
    get_rcparams,
    set_fig_size,
)
from synthetmic import LaguerreDiagramGenerator
from synthetmic.data.paper import (
    create_example3_data,
    create_example4_data,
    create_example4b_data,
)
from synthetmic.plot import plot_2dcells_as_matplotlib_fig

plt.rcParams.update(get_rcparams())


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
    INITIALIZERS = ("random", "banded", "clustered", "mixed_banded_and_random")

    grid = len(INITIALIZERS) // 2
    fig = plt.figure(figsize=set_fig_size(subplots=(grid, grid), adjust_height=0.1))
    tags = ("a", "b", "c", "d")

    for i, init in enumerate(INITIALIZERS):
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
    GRADIENTS = ("increasing", "large_at_middle")

    grid = len(GRADIENTS)
    fig = plt.figure(figsize=set_fig_size(subplots=(grid, grid)))
    tags = ("a", "b")

    for i, gradient in enumerate(GRADIENTS):
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
