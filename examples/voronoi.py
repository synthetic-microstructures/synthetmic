from pathlib import Path

import matplotlib.pyplot as plt

from examples.utils import set_fig_size
from synthetmic import VoronoiDiagramGenerator
from synthetmic.data import toy
from synthetmic.plot import plot_2dcells_as_matplotlib_fig, plot_cells_as_pyvista_fig


def voronoi_with_random_seeds(
    space_dim: int, n_grains: int, n_iter: int, is_periodic: bool, save_path: str | Path
) -> None:
    domain, _ = toy.create_unit_domain(space_dim)
    seeds = toy.sample_random_seeds(domain=domain, n_grains=n_grains)

    vdg = VoronoiDiagramGenerator(
        n_iter=n_iter,
        damp_param=1.0,
        verbose=True,
    )
    vdg.fit(
        seeds=seeds,
        domain=domain,
        periodic=[True] * space_dim if is_periodic else None,
    )

    fig, ax = plt.subplots(figsize=set_fig_size())

    colorby = vdg.get_fitted_volumes()
    title = f"An example of Voronoi diagram in {space_dim}D"

    match space_dim:
        case 2:
            plot_2dcells_as_matplotlib_fig(
                generator=vdg, ax=ax, title=title, colorby=colorby, save_path=save_path
            )
            ax.axis("off")
            ax.set_aspect("equal")

            fig.savefig(save_path, bbox_inches="tight")

        case 3:
            plot_cells_as_pyvista_fig(
                generator=vdg,
                colorby=colorby,
                save_path=save_path,
                include_slices=True,
                title=None,
            )

        case _:
            raise ValueError("space_dim must be 2 or 3.")

    return None
