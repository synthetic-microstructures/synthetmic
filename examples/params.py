from pathlib import Path
from typing import Callable

import click

from examples.utils import Example


def save_dir(func: Callable) -> Callable:
    return click.option(
        "--save-dir",
        default=Path.cwd(),
        type=click.Path(exists=True, dir_okay=True, file_okay=False),
        show_default=False,
        help="The dir to save generated results to. Default to current working dir.",
    )(func)


def periodic(func: Callable) -> Callable:
    return click.option(
        "--periodic",
        "-p",
        is_flag=True,
        help="Give this flag if you want the box or domain to be periodic in all directions.",
    )(func)


def example(func: Callable) -> Callable:
    return click.option(
        "--example",
        type=click.Choice([e.value for e in Example], case_sensitive=True),
        default=Example.TWO_DIM.value,
        show_default=True,
        help="The example case to recreate.",
    )(func)


def interactive(func: Callable) -> Callable:
    return click.option(
        "--interactive",
        "-i",
        is_flag=True,
        help="""Give this flag if you want the generated figure to be in
    interative mode. Note: this only works for 3d examples.
    """,
    )(func)
