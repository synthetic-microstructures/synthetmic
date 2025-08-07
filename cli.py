import click

from examples.analyse import damp_param_effect, recreate_figure6
from examples.recreate_2d import (
    recreate_fig1,
    recreate_fig2,
    recreate_fig4,
    recreate_fig5,
)
from examples.recreate_3d import recreate_fig12, recreate_fig13
from examples.utils import Analysis, Example


def color_text(text: str, color: str = "green", bold: bool = False) -> str:
    return click.style(text=text, fg=color, bold=bold)


@click.group()
def entry() -> None:
    """A cli for running various examples and analyses.

    Author: Rasheed Ibraheem, D. P. Bourne, and S. M. Roper.
    """
    pass


@entry.command()
@click.option(
    "--example",
    type=click.Choice([e.value for e in Example], case_sensitive=True),
    default=Example.TWO_DIM,
    show_default=True,
    help="""The example case(s) to recreate from the paper:
    Bourne, D.P., Kok, P.J.J., Roper, S.M. & Spanjer, W.D.T. (2020) Laguerre tessellations
    and polycrystalline microstructures: A fast algorithm for generating grains of
    given volumes, Philosophical Magazine, 100, 2677-2707.
    """,
)
@click.option(
    "--save-dir",
    default="./",
    type=click.Path(exists=True),
    show_default=True,
    help="The dir to save the recreated figures.",
)
@click.option(
    "--interactive",
    "-i",
    is_flag=True,
    help="""Give this flag if you want the generated figure to be in
    interative mode. Note: this only works for 3d examples.
    """,
)
@click.option(
    "--periodic",
    "-p",
    is_flag=True,
    help="""Give this flag if you want the generated figure to be periodic in all 
    directions.
    """,
)
def recreate(example: str, save_dir: str, interactive: bool, periodic: bool) -> None:
    """
    Recreate figures in the source paper.
    """

    def two_d_example() -> None:
        for fn, sn in zip(
            (recreate_fig1, recreate_fig2, recreate_fig4, recreate_fig5),
            ("figure1", "figure2", "figure4", "figure5"),
        ):
            fn(save_path=f"{save_dir}/{sn}.pdf", is_periodic=periodic)

        return None

    def three_d_example() -> None:
        ext = "html" if interactive else "pdf"
        for fn, sn in zip((recreate_fig12, recreate_fig13), ("figure12", "figure13")):
            fn(
                save_path=f"{save_dir}/{sn}.{ext}",
                is_periodic=periodic,
                interactive=interactive,
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

            two_d_example()

        case Example.THREE_DIM:
            three_d_example()

        case Example.ALL:
            two_d_example()
            three_d_example()

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
    default=Analysis.RUNTIME,
    show_default=True,
    help="""Run some analysis of the performance of the algorithms developed in the paper:
    Bourne, D.P., Kok, P.J.J., Roper, S.M. & Spanjer, W.D.T. (2020) Laguerre tessellations
    and polycrystalline microstructures: A fast algorithm for generating grains of
    given volumes, Philosophical Magazine, 100, 2677-2707.
    """,
)
@click.option(
    "--save-dir",
    default="./",
    type=click.Path(exists=True),
    show_default=True,
    help="The dir to save the generated figures.",
)
@click.option(
    "--periodic",
    "-p",
    is_flag=True,
    help="""Give this flag if you want the box or domain to be periodic in all 
    directions.
    """,
)
def analyse(analysis: str, save_dir: str, periodic: bool) -> None:
    """
    Analyse the performance of the algorithms developed in the source paper.
    """

    def runtime() -> None:
        recreate_figure6(save_path=f"./{save_dir}/figure6.pdf", is_periodic=periodic)

    def damp() -> None:
        damp_param_effect(
            save_path=f"./{save_dir}/damp_param_effect.pdf", is_periodic=periodic
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
            runtime()

        case Analysis.DAMP:
            damp()

        case Analysis.ALL:
            runtime()
            damp()

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


if __name__ == "__main__":
    entry()
