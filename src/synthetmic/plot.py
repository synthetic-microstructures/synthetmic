import tempfile
from enum import StrEnum, auto

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import vtk
from matplotlib import cm, colormaps
from matplotlib.axes import Axes
from vtk.numpy_interface import dataset_adapter as dsa
from vtk.util.numpy_support import vtk_to_numpy

from synthetmic import LaguerreDiagramGenerator


class _PyvistaSupportedExtension(StrEnum):
    HTML = auto()
    SVG = auto()
    EPS = auto()
    PS = auto()
    PDF = auto()
    TEX = auto()


def plot_2dcells_as_matplotlib_fig(
    generator: LaguerreDiagramGenerator,
    ax: Axes | None = None,
    title: str | None = None,
    colorby: np.ndarray | list[float] | None = None,
    colormap: str = "plasma",
    save_path: str | None = None,
) -> Axes:
    """
    A function that plots a 2D Laguerre cells as matplotlib figure.

    Parameters
    ----------

    generator : LaguerreDiagramGenerator
        a fitted LaguerreDiagramGenerator object.
    axis : Axis, optional
        a matplotlib axis object to handle the figure, if None, a new one will be created
    title : str or None, optional
        title of the figure.
    colorby : ndarray or list, shape (N,), optional
        a 1d array of scalars for coloring the cells, if None, the cells will be colored
        by their respective volume.
    colormap: str, optional
        a string representing one of the supported colormaps in the matplotlib library.
    save_path: str or None
        a string reperesenting the path to save the generated figure to, e.g., ./plots/figure2.pdf.
        If None, figure will not be saved.

    Returns
    -------

    ax : matplotlib Axes object
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".vtk", delete=True) as tmp_file:
        filename = tmp_file.name

        generator.diagram_to_vtk(filename)

        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(filename)
        reader.Update()

        data = reader.GetOutput()

    N = data.GetNumberOfCells()

    numpy_array_of_cells = dsa.WrapDataObject(data).Cells
    numpy_array_of_points = dsa.WrapDataObject(data).Points

    cell_data = data.GetCellData()
    array = cell_data.GetArray(1)

    # Extract array values
    cell_numbers = []
    for j in range(array.GetNumberOfTuples()):
        tuple_data = array.GetTuple(j)  # Get the data tuple for the cell
        cell_numbers.append(int(tuple_data[0]))

    cells = vtk_to_numpy(numpy_array_of_cells)
    verts = vtk_to_numpy(numpy_array_of_points)

    if ax is None:
        _, ax = plt.subplots()

    if colormap not in list(colormaps):
        raise ValueError(
            f"Invalid colormap string: {colormap}. Value must be one of the following: {', '.join(list(colormaps))}"
        )

    if colorby is None:
        # default to coloring cells by their volumes
        colorby = generator.get_fitted_volumes()

    cmap = cm.get_cmap(colormap)
    norm = mcolors.Normalize(vmin=np.min(colorby), vmax=np.max(colorby))
    colors = cmap(norm(colorby))

    idx = 0
    for k in range(N):
        nv = cells[idx]
        vidx = cells[idx + 1 : idx + nv + 1]
        idx = idx + nv + 1

        ax.fill(
            verts[vidx, 0],
            verts[vidx, 1],
            color=colors[cell_numbers[k]],
            linewidth=1,
            alpha=0.7,
        )

    idx = 0
    for k in range(N):
        nv = cells[idx]
        vidx = cells[idx + 1 : idx + nv + 1]
        idx = idx + nv + 1
        ax.plot(verts[vidx, 0], verts[vidx, 1], "k", linewidth=1)

    if title is not None:
        ax.set_title(title)

    if save_path is not None:
        ax.axis("off")
        ax.set_aspect("equal")

        plt.savefig(save_path, bbox_inches="tight")

    return ax


def plot_cells_as_pyvista_fig(
    generator: LaguerreDiagramGenerator,
    window_size: tuple[int, int] = (1024, 768),
    notebook: bool = False,
    title: str | None = None,
    colorby: np.ndarray | list[float] | None = None,
    colormap: str = "plasma",
    save_path: str | None = None,
    include_slices: bool = False,
) -> pv.Plotter:
    """
    A function that plots Laguerre cells (both 2D and 3D) as pyvista figure.

    Parameters
    ----------

    generator : LaguerreDiagramGenerator
        a fitted LaguerreDiagramGenerator object.
    window_size : tuple[int, int], optional
        the figure size.
    notebook : bool, optional
        when True, the resulting plot is placed inline a jupyter notebook. Assumes a jupyter console is active.
        Automatically enables off_screen.
    title : str or None, optional
        title of the figure.
    colorby : ndarray or list, shape (N,), optional
        a 1d array of scalars for coloring the cells, if None, the cells will be colored
        by their respective volume.
    colormap: str, optional
        a string representing one of the supported colormaps in the matplotlib library.
    save_path: str or None
        a string reperesenting the path to save the generated figure to, e.g., ./plots/figure2.pdf.
        If None, figure will not be saved. File extension must be one of the supported extensions in
        pyvista.
    include_slices : bool, optional
        when True, include othorgonal slice and slices along the axes coordinates to the figure.

    Returns
    -------

    plotter : pyvista Plotter object
    """
    mesh = generator.get_mesh()

    if colorby is None:
        colorby = generator.get_fitted_volumes()

    # create cell data that gives the cell volumes, this allows us to colour by cell volumes
    mesh.cell_data["vols"] = colorby[mesh.cell_data["num"].astype(int)]

    if colormap not in list(colormaps):
        raise ValueError(
            f"Invalid colormap string: {colormap}. Value must be one of the following: {', '.join(list(colormaps))}"
        )

    _, space_dim = generator.get_positions().shape

    if include_slices:
        N_ROW, N_COL = 2, 3
        plotter = pv.Plotter(
            off_screen=True,
            window_size=list(window_size),
            notebook=notebook,
            shape=(N_ROW, N_COL),
        )

        NUM_SLICES = 3
        meshes = (
            mesh,
            mesh.slice_orthogonal(),
            mesh.slice_along_axis(n=2, axis="x"),
            mesh.slice_along_axis(n=NUM_SLICES, axis="x"),
            mesh.slice_along_axis(n=NUM_SLICES, axis="y"),
            mesh.slice_along_axis(n=NUM_SLICES, axis="z"),
        )

        c = 0
        for i in range(N_ROW):
            for j in range(N_COL):
                plotter.subplot(i, j)
                plotter.add_mesh(
                    meshes[c],
                    show_edges=True,
                    scalars="vols",
                    show_scalar_bar=False,
                    cmap=colormap,
                )

                if space_dim == 2:
                    plotter.camera_position = "xy"

                c += 1

        plotter.link_views()

    else:
        plotter = pv.Plotter(
            off_screen=True,
            window_size=list(window_size),
            notebook=notebook,
        )
        plotter.add_mesh(
            mesh,
            show_edges=True,
            scalars="vols",
            show_scalar_bar=False,
            cmap=colormap,
        )

        if space_dim == 2:
            plotter.camera_position = "xy"

    if title is not None:
        plotter.add_title(title=title)

    plotter.show_axes()

    if save_path is not None:
        ext = save_path.split(".")[-1].lower()

        if ext in _PyvistaSupportedExtension:
            if ext == _PyvistaSupportedExtension.HTML:
                plotter.export_html(save_path)

            else:
                plotter.save_graphic(save_path)

        else:
            raise ValueError(
                f"Invalid file extension: {ext}. Extension must be one of [{', '.join(_PyvistaSupportedExtension)}]."
            )

    return plotter
