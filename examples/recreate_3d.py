from dataclasses import asdict

from synthetmic import LaguerreDiagramGenerator
from synthetmic.data.paper import create_example5p4_data, create_example5p5_data
from synthetmic.plot import plot_cells_as_pyvista_fig


def recreate_fig12(save_path: str, is_periodic: bool) -> None:
    TOL = 1.0
    N_ITER = 20

    data = create_example5p4_data(is_periodic=is_periodic)

    ldg = LaguerreDiagramGenerator(tol=TOL, n_iter=N_ITER)
    ldg.fit(**asdict(data))

    plot_cells_as_pyvista_fig(
        generator=ldg,
        title=None,
        colorby=data.volumes,
        save_path=save_path,
        include_slices=True,
    )

    return None


def recreate_fig13(save_path: str, is_periodic: bool) -> None:
    TOL = 1.0
    N_ITER = 5

    data = create_example5p5_data(is_periodic=is_periodic)

    ldg = LaguerreDiagramGenerator(tol=TOL, n_iter=N_ITER)
    ldg.fit(**asdict(data))

    plot_cells_as_pyvista_fig(
        generator=ldg,
        title=None,
        colorby=data.volumes,
        save_path=save_path,
        include_slices=True,
    )

    return None
