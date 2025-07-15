import tempfile
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import vtk
from matplotlib import cm
from pysdot import OptimalTransport
from vtk.numpy_interface import dataset_adapter as dsa
from vtk.util.numpy_support import vtk_to_numpy


def plot_cells2d(
    optimal_transport: OptimalTransport,
    ax: Any | None = None,
    titlestr: str | None = None,
    colorby: np.ndarray | list[float] | None = None,
    cmap: Any | None = cm.plasma,
) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".vtk", delete=True) as tmp_file:
        filename = tmp_file.name

        optimal_transport.pd.display_vtk(filename)

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

    if (colorby is not None) and (cmap is not None):
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

    if titlestr is not None:
        ax.set_title(titlestr)

    return None


def plot_cells3d(
    optimal_transport: OptimalTransport,
    window_size: tuple[int, int] = (1024, 768),
    notebook: bool = False,
    titlestr: str | None = None,
    colorby: np.ndarray | list[float] | None = None,
    save_path: str | None = None,
    interactive: bool = False,
) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".vtk", delete=True) as tmp_file:
        filename = tmp_file.name

        optimal_transport.pd.display_vtk(filename)

        otgrid = pv.read(filename)

    # Store the volumes in an array
    if colorby is None:
        colorby = optimal_transport.pd.integrals()

    # Specify the colours
    # create cell data that gives the cell volumes, this allows us to colour by cell volumes
    otcell_col = colorby[otgrid.cell_data["num"].astype(int)]
    otgrid.cell_data["vols"] = otcell_col

    N_ROW, N_COL = 2, 3
    otplotter = pv.Plotter(
        off_screen=True,
        window_size=list(window_size),
        notebook=notebook,
        title=titlestr,
        shape=(N_ROW, N_COL),
    )

    meshes = (
        otgrid,
        otgrid.slice_orthogonal(),
        otgrid.slice_along_axis(n=2, axis="x"),
        otgrid.slice_along_axis(axis="x"),
        otgrid.slice_along_axis(axis="y"),
        otgrid.slice_along_axis(axis="z"),
    )

    c = 0
    for i in range(N_ROW):
        for j in range(N_COL):
            otplotter.subplot(i, j)
            otplotter.add_mesh(
                meshes[c],
                show_edges=True,
                show_scalar_bar=False,
            )

            c += 1

    otplotter.link_views()
    otplotter.view_isometric()

    if interactive:
        otplotter.show_axes()
        otplotter.add_scalar_bar(vertical=False)

    # Add a headlight
    light = pv.Light(light_type="headlight")
    otplotter.add_light(light)

    if save_path is not None:
        otplotter.export_html(save_path) if interactive else otplotter.save_graphic(
            save_path
        )

    return None
