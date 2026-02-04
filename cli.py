from pathlib import Path

import click

from examples import params
from examples.analyse import damp_param_effect, recreate_figure6
from examples.recreate_2d import (
    recreate_fig1,
    recreate_fig2,
    recreate_fig4,
    recreate_fig5,
)
from examples.recreate_3d import recreate_fig12, recreate_fig13
from examples.utils import Analysis, Example
from examples.voronoi import voronoi_with_random_seeds


def color_text(text: str, color: str = "green", bold: bool = False) -> str:
    return click.style(text=text, fg=color, bold=bold)


@click.group()
def entry() -> None:
    """
    A cli for running various examples and analyses.

    Author: Rasheed Ibraheem, D. P. Bourne, and S. M. Roper.
    """
    pass


@entry.command()
@params.example
@params.save_dir
@params.interactive
@params.periodic
def recreate(example: str, save_dir: str, interactive: bool, periodic: bool) -> None:
    """
    Recreate figures in the source paper:

    Bourne, D.P., Kok, P.J.J., Roper, S.M. & Spanjer, W.D.T. (2020) Laguerre tessellations
    and polycrystalline microstructures: A fast algorithm for generating grains of
    given volumes, Philosophical Magazine, 100, 2677-2707.
    """

    def _two_d() -> None:
        for fn, sn in zip(
            (recreate_fig1, recreate_fig2, recreate_fig4, recreate_fig5),
            ("figure1", "figure2", "figure4", "figure5"),
        ):
            fn(save_path=Path(save_dir, f"{sn}.pdf"), is_periodic=periodic)

        return None

    def _three_d() -> None:
        ext = "html" if interactive else "pdf"
        for fn, sn in zip((recreate_fig12, recreate_fig13), ("figure12", "figure13")):
            fn(
                save_path=Path(save_dir, f"{sn}.{ext}"),
                is_periodic=periodic,
            )

        return None

    click.echo(
        color_text(
            f"Recreating figures for the example case: {example}",
            bold=True,
        ),
        color=True,
    )

    match example:
        case Example.TWO_DIM:
            if interactive:
                click.echo(
                    color_text(
                        "WARNING: interative mode is turned on but example case is 2d. "
                        "No interative figures will be generated and static figures will be produced instead.",
                        bold=True,
                        color="yellow",
                    ),
                    color=True,
                )

            _two_d()

        case Example.THREE_DIM:
            _three_d()

        case Example.ALL:
            _two_d()
            _three_d()

        case _:
            click.echo(
                color_text(
                    f"Invalid example case: {example}. Value must be one of {', '.join(Example)}",
                    bold=True,
                    color="red",
                ),
                color=True,
            )

            return None

    click.echo(
        color_text(
            f"Done! See {save_dir} for the recreated figures",
            bold=True,
        ),
        color=True,
    )

    return None


@entry.command()
@click.option(
    "--analysis",
    type=click.Choice([a.value for a in Analysis], case_sensitive=True),
    default=Analysis.RUNTIME.value,
    show_default=True,
    help="Type of analysis to run.",
)
@params.save_dir
@params.periodic
def analyse(analysis: str, save_dir: str, periodic: bool) -> None:
    """
    Analyse the performance of the algorithms developed in the source paper:

    Bourne, D.P., Kok, P.J.J., Roper, S.M. & Spanjer, W.D.T. (2020) Laguerre tessellations
    and polycrystalline microstructures: A fast algorithm for generating grains of
    given volumes, Philosophical Magazine, 100, 2677-2707.
    """

    def _runtime() -> None:
        recreate_figure6(save_path=Path(save_dir, "figure6.pdf"), is_periodic=periodic)

    def _damp() -> None:
        damp_param_effect(
            save_path=Path(save_dir, "damp_param_effect.pdf"), is_periodic=periodic
        )

    click.echo(
        color_text(
            f"Running analysis for the case: {analysis}",
            bold=True,
        ),
        color=True,
    )
    match analysis:
        case Analysis.RUNTIME:
            _runtime()

        case Analysis.DAMP:
            _damp()

        case Analysis.ALL:
            _runtime()
            _damp()

        case _:
            click.echo(
                color_text(
                    f"Invalid analysis case: {analysis}. Value must be one of {', '.join(Analysis)}",
                    bold=True,
                    color="red",
                ),
                color=True,
            )

            return None

    click.echo(
        color_text(
            f"Done! See {save_dir} for the generated figures",
            bold=True,
        ),
        color=True,
    )

    return None


@entry.command()
@params.example
@params.save_dir
@params.interactive
@params.periodic
@click.option(
    "--n-grains",
    type=click.IntRange(min=1, max=None, min_open=False),
    default=100,
    show_default=True,
    help="Number of grains or cells in the Voronoi diagram.",
)
@click.option(
    "--n-iter",
    type=click.IntRange(min=0, max=None, min_open=False),
    default=10,
    show_default=True,
    help="Number of Lloyd iterations.",
)
def voronoi(
    example: str,
    save_dir: str,
    interactive: bool,
    periodic: bool,
    n_grains: int,
    n_iter: int,
) -> None:
    """
    Generate Voronoi diagrams with random seeds.
    """

    def _two_d() -> None:
        voronoi_with_random_seeds(
            space_dim=2,
            save_path=Path(
                save_dir, f"2d_voronoi_diagram_n_grains_{n_grains}_n_iter_{n_iter}.pdf"
            ),
            is_periodic=periodic,
            n_grains=n_grains,
            n_iter=n_iter,
        )

        return None

    def _three_d() -> None:
        ext = "html" if interactive else "pdf"
        voronoi_with_random_seeds(
            space_dim=3,
            save_path=Path(
                save_dir,
                f"3d_voronoi_diagram_n_grains_{n_grains}_n_iter_{n_iter}.{ext}",
            ),
            is_periodic=periodic,
            n_grains=n_grains,
            n_iter=n_iter,
        )

        return None

    click.echo(
        color_text(
            f"Generating Voronoi diagram for the example case: {example}",
            bold=True,
        ),
        color=True,
    )

    match example:
        case Example.TWO_DIM:
            if interactive:
                click.echo(
                    color_text(
                        "WARNING: interative mode is turned on but example case is 2d. "
                        "No interative figures will be generated and static figures will be produced instead.",
                        bold=True,
                        color="yellow",
                    ),
                    color=True,
                )

            _two_d()

        case Example.THREE_DIM:
            _three_d()

        case Example.ALL:
            _two_d()
            _three_d()

        case _:
            click.echo(
                color_text(
                    f"Invalid example case: {example}. Value must be one of {', '.join(Example)}",
                    bold=True,
                    color="red",
                ),
                color=True,
            )

            return None

    click.echo(
        color_text(
            f"Done! See {save_dir} for the recreated figures",
            bold=True,
        ),
        color=True,
    )

    return None


if __name__ == "__main__":
    entry()
