import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Self

import numpy as np
import pyvista as pv
from pysdot import OptimalTransport, PowerDiagram

from synthetmic._deprecated import warn_deprecated
from synthetmic._errors import check_is_fitted
from synthetmic._validate import validate_generator_config
from synthetmic.data.utils import DiagramConfig, VoxelGrid
from synthetmic.types import DEPRECATED, MissingType
from synthetmic.typing import FloatArray, IntSequence
from synthetmic.utils import (
    add_replicants,
    assign_points_to_grains,
    build_domain,
    compute_cell_centers,
)


@dataclass(frozen=True, slots=True)
class VoronoiEvent:
    """
    Event emitted after each Lloyd iteration during Voronoi diagram generation.

    Attributes
    ----------
    iteration : int
        Current iteration number (1-based).
    total : int
        Total number of Lloyd iterations to be performed.
    delta_pos_norm : float
        Euclidean norm of the displacement between the updated seed positions
        and the corresponding cell centroids.
    """

    iteration: int
    total: int
    delta_pos_norm: float


@dataclass(frozen=True, slots=True)
class LaguerreEvent:
    """
    Event emitted after each Lloyd iteration during Laguerre diagram generation.

    Attributes
    ----------
    iteration : int
        Current iteration number (1-based).
    total : int
        Total number of Lloyd iterations to be performed.
    mean_percentage_error : float
        Mean relative percentage error between the computed and target cell
        volumes.
    max_percentage_error : float
        Maximum relative percentage error between the computed and target cell
        volumes.
    """

    iteration: int
    total: int
    mean_percentage_error: float
    max_percentage_error: float


def _noop_callback(*args, **kwargs) -> None:
    pass


class DiagramGenerator(ABC):
    """
    Base class for diagram generator.

    Parameters
    ----------
    verbose : Deprecated, has no effect. Will be removed in the next release.

    Attributes
    ----------
    pd_ : pysdot.PowerDiagram
        Underlying power diagram object.
    space_dim_in_ : int
        Space dimension seen during fit.
    n_grains_in_ : int
        Number of grains seen during fit.
    """

    pd_: PowerDiagram
    space_dim_in_: int
    n_grains_in_: int

    def __init__(self, verbose: bool | MissingType = DEPRECATED) -> None:
        if not isinstance(verbose, MissingType):
            warn_deprecated(
                "The `verbose` argument is deprecated and has no effect. "
                "It will be removed in a future release. Please pass a "
                "callback to to the `fit` method."
            )

    @abstractmethod
    def fit(
        self,
        config: DiagramConfig,
        *,
        callback: Callable[..., None],
    ) -> Self:
        pass

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        pass

    def voxelise(
        self, points_per_dim: IntSequence, domain: FloatArray | None = None
    ) -> VoxelGrid:
        """
        Voxelise diagram.

        Parameters
        ----------
        points_per_dim : IntSequence, shape (N,)
            Number of grid points along each dimension.
        domain : FloatArray or None, optional, default=None
            If not None, it represents the minimum and maximum coordinates of the box
            in each of the d dimensions (d = 2 or 3), and the underlying power diagram will
            be treated as periodic in all dimensions. The positions of the power diagram will be
            mapped to the domain.

        Returns
        -------
        synthetmic.VoxelGrid
        """
        check_is_fitted(self, ["pd_", "space_dim_in_"])

        if domain is None:
            origin = self.pd_.get_domain().min_position()
            size = self.pd_.get_domain().max_position() - origin

        else:
            domain = np.asarray(domain)
            origin = domain[:, 0]
            size = domain[:, 1] - origin

        size = tuple(size)
        origin = tuple(origin)

        if not all(
            len(x) == self.space_dim_in_ for x in (size, origin, points_per_dim)
        ):
            raise ValueError("All space dimensions must match.")

        centers = compute_cell_centers(
            origin=origin, size=size, points_per_dim=points_per_dim
        )
        centers = centers.reshape(-1, self.space_dim_in_)
        grain_indices = assign_points_to_grains(
            points=centers, pd=self.pd_, domain=domain
        )

        return VoxelGrid(
            voxels=grain_indices.reshape(points_per_dim, order="F"),
            origin=origin,
            size=size,
        )

    # TODO: complete this with orientation or
    # texture or Euler angle calculations
    def get_orientations(self) -> FloatArray:
        check_is_fitted(self, ["pd_", "space_dim_in_"])

        raise NotImplementedError

    def get_fitted_volumes(self) -> FloatArray:
        """
        Get the computed diagram cell volumes.
        """
        check_is_fitted(self, ["pd_"])
        return self.pd_.integrals()

    def get_mesh(self) -> pv.UnstructuredGrid | pv.PolyData:
        """
        Get the underlying diagram mesh as a pyvista PolyData or UnstructuredGrid data object.
        """
        check_is_fitted(self, ["pd_"])

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = Path(tmpdir) / "diagram.vtk"
            self.pd_.display_vtk(str(filename), points=None, centroids=None)

            return pv.read(filename)

    def get_positions(self) -> FloatArray:
        """
        Get the final positions of seeds.
        """
        check_is_fitted(self, ["pd_"])
        return self.pd_.get_positions()

    def get_centroids(self) -> FloatArray:
        """
        Get the centroids of the cells in the Voronoi diagram.
        """
        check_is_fitted(self, ["pd_"])
        return self.pd_.centroids()

    def get_vertices(self) -> dict[int, list]:
        """
        Get the vertices of cells in the diagram.

        Return
        ------
        A dictionary with keys as cell ids and values as the
        corresponding vertices.

        In 2D, the format looks like this:

        {
            0: [v_1, v_2, ...],
            ...
            n-1: [v_1, v_2, ...],

        }
        where n is the number of cells or grains.

        In 3D, the format looks like this:

        {
            0: [[v_1, v_2, ...], [v_1, v_2, ...], ...],
            ...
            n-1: [[v_1, v_2, ...], [v_1, v_2, ...], ...],

        }
        where n is the number of cells or grains. Note that the inner
        list of vertices for each cell corresponds to the face vertices.
        """
        check_is_fitted(self, ["pd_", "space_dim_in_"])
        res = {}

        if self.space_dim_in_ == 2:
            offsets, coords = self.pd_.cell_polyhedra()

            for i in range(len(offsets) - 1):
                s, e = offsets[i : i + 2]
                res[i] = coords[s:e].tolist()

        elif self.space_dim_in_ == 3:
            offsets_polyhedra, offsets_polygon, coords = self.pd_.cell_polyhedra()

            for i in range(len(offsets_polyhedra) - 1):
                s1, e1 = offsets_polyhedra[i : i + 2]
                sub_offsets = offsets_polygon[
                    s1 : e1 + 1
                ]  # 1 is added to e1 to get the correct vertex index

                cell_vertices = []

                for j in range(len(sub_offsets) - 1):
                    s2, e2 = sub_offsets[j : j + 2]

                    cell_vertices.append(coords[s2:e2].tolist())

                res[i] = cell_vertices

        return res

    def get_weights(self) -> FloatArray:
        """
        Get the weights of the diagram.
        """
        check_is_fitted(self, ["pd_"])
        return self.pd_.get_weights()

    def to_vtk(self, filename: str | Path) -> None:
        """
        Write the generated diagram to .vtk file; filename must ends with .vtk.
        """
        check_is_fitted(self, ["pd_"])

        if Path(filename).suffix != ".vtk":
            raise ValueError("filename suffix must be .vtk")

        self.pd_.display_vtk(filename=filename, points=False, centroids=False)


class VoronoiDiagramGenerator(DiagramGenerator):
    """
    Voronoi diagram generator.

    Parameters
    ----------
    n_iter : int, optional
        Number of iterations of Lloyd's algorithm (move each seed to the
        centroid of its cell). If it is set to 0, then no Lloyd's iteration
        will be performed.
    damp_param : float [0, 1], optional
        The damping parametr of the damped Lloyd step; value must be between
        0 and 1 (inclusive at both ends).

    Attributes
    ----------
    delta_pos_norm_ : float
        Euclidean norm of the displacement between the updated seed positions
        and the corresponding cell centroids.
    """

    def __init__(
        self,
        n_iter: int = 5,
        damp_param: float = 1.0,
        verbose: bool | MissingType = DEPRECATED,
    ) -> None:
        validate_generator_config(
            tol=None,
            n_iter=n_iter,
            damp_param=damp_param,
        )

        super().__init__(verbose=verbose)
        self.n_iter = n_iter
        self.damp_param = damp_param

    def fit(
        self,
        config: DiagramConfig,
        *,
        callback: Callable[[VoronoiEvent], None] = _noop_callback,
    ) -> Self:
        """
        Fit Voronoi diagram on a given diagram configuration.

        Parameters
        ----------
        config : SynthetMic.DiagramConfig
            Diagram configuration.
        callback : callable, optional
            Invoked after every Lloyd iteration.

        Returns
        -------
        synthetmic.VoronoiDiagramGenerator
        """
        if not np.allclose(config.initial_weights, config.initial_weights[0]):
            raise ValueError("Weights must either be all zero or equal.")

        omega, boxsize = build_domain(domain=config.domain, periodic=config.periodic)

        n_grains, space_dim = config.seeds.shape
        self.n_grains_in_ = n_grains
        self.space_dim_in_ = space_dim

        pd = PowerDiagram(
            positions=config.seeds, weights=config.initial_weights, domain=omega
        )
        if config.periodic is not None:
            add_replicants(obj=pd, periodic=config.periodic, boxsize=boxsize)

        positions = pd.get_positions()
        centroids = pd.centroids()
        delta_pos_norm = np.linalg.norm(centroids - positions)

        if self.n_iter > 0:
            for k in range(self.n_iter):
                positions = (
                    1 - self.damp_param
                ) * positions + self.damp_param * centroids

                pd.set_positions(positions)
                centroids = pd.centroids()
                delta_pos_norm = np.linalg.norm(centroids - positions)

                callback(
                    VoronoiEvent(
                        iteration=k + 1,
                        total=self.n_iter,
                        delta_pos_norm=delta_pos_norm,
                    )
                )

        self.pd_ = pd
        self.data_pos_norm_ = delta_pos_norm

        return self

    def get_params(self) -> dict[str, Any]:
        """
        Get the parameters of this instance as a dictionary.
        """

        return dict(n_iter=self.n_iter, damp_param=self.damp_param)


class LaguerreDiagramGenerator(DiagramGenerator):
    """
    Fit Laguerre diagram on a given diagram configuration.

    Parameters
    ----------
    tol : float, optional
        Relative percentage error for volumes.
    n_iter : int, optional
        Number of iterations of Lloyd's algorithm (move each seed to the
        centroid of its cell). If it is set to 0, then no Lloyd's iteration
        will be performed.
    damp_param : float [0, 1], optional
        The damping parametr of the damped Lloyd step; value must be between
        0 and 1 (inclusive at both ends).

    Attributes
    ----------
    mean_percentage_error_ : float
        Mean relative percentage error between the computed and target cell
        volumes.
    max_percentage_error_ : float
        Maximum relative percentage error between the computed and target cell
        volumes.
    """

    def __init__(
        self,
        tol: float = 1.0,
        n_iter: int = 5,
        damp_param: float = 1.0,
        verbose: bool | MissingType = DEPRECATED,
    ):
        validate_generator_config(
            tol=tol,
            n_iter=n_iter,
            damp_param=damp_param,
        )

        super().__init__(verbose=verbose)
        self.tol = tol
        self.n_iter = n_iter
        self.damp_param = damp_param

    def _compute_errors(self, y_hat: FloatArray, y: FloatArray) -> tuple[float, float]:
        percentage_errors = 100.0 * np.abs(y_hat - y) / y
        return percentage_errors.mean(), percentage_errors.max()

    def fit(
        self,
        config: DiagramConfig,
        *,
        callback: Callable[[LaguerreEvent], None] = _noop_callback,
    ) -> Self:
        """
        Fit Laguerre diagram on a given diagram configuration.

        Parameters
        ----------
        config : SynthetMic.DiagramConfig
            Diagram configuration.
        callback: callable, optional
            Invoked after every Lloyd iteration.

        Returns
        -------
        synthetmic.LaguerreDiagramGenerator

        References
        ----------
        This function implements Algorithm 1 and 2 from the following paper:

        Bourne, D.P., Kok, P.J.J., Roper, S.M. & Spanjer, W.D.T. (2020)
        Laguerre tessellations and polycrystalline microstructures:
        A fast algorithm for generating grains of given volumes,
        Philosophical Magazine, 100, 2677-2707.
        https://www.tandfonline.com/doi/full/10.1080/14786435.2020.1790053
        """
        if config.volumes is None:
            raise ValueError("`volumes` must be provided for Laguerre diagrams.")

        n_grains, space_dim = config.seeds.shape
        self.n_grains_in_ = n_grains
        self.space_dim_in_ = space_dim

        # Turn the relative percentage error into an absolute error tolerance
        # by using the smallest volume
        err_tol = np.min(config.volumes) * self.tol / 100.0

        # Set up the optimal transport problem
        omega, boxsize = build_domain(domain=config.domain, periodic=config.periodic)
        ot = OptimalTransport(
            positions=config.seeds,
            masses=config.volumes,
            weights=config.initial_weights,
            domain=omega,
            obj_max_dm=err_tol,
            verbosity=0,
        )

        add_replicants(obj=ot, periodic=config.periodic, boxsize=boxsize)

        mean_percentage_error = max_percentage_error = np.nan
        if self.n_iter == 0:
            ot.adjust_weights()
            mean_percentage_error, max_percentage_error = self._compute_errors(
                y_hat=ot.pd.integrals(), y=config.volumes
            )

        else:
            _EMPTY_CELL_VOLUME_TOL = 1e-10
            positions = config.seeds.copy()
            for k in range(self.n_iter):
                positions = (
                    1 - self.damp_param
                ) * positions + self.damp_param * ot.get_centroids()
                ot.set_positions(positions)

                # Solve the optimal transport problem.
                # If moving the seeds and maintaining the weights gives a cell that is
                # empty, then we have a bad initial guess for the OT solver.
                # Check whether the smallest volume is bigger than some tolerance:
                # if so, then use the same weights; if not, reset the weights to initial_weights.
                cell_volumes = ot.pd.integrals()
                if np.min(cell_volumes) > _EMPTY_CELL_VOLUME_TOL:
                    ot.adjust_weights()
                else:
                    ot.set_weights(config.initial_weights)
                    ot.adjust_weights()

                mean_percentage_error, max_percentage_error = self._compute_errors(
                    y_hat=cell_volumes, y=config.volumes
                )

                callback(
                    LaguerreEvent(
                        iteration=k + 1,
                        total=self.n_iter,
                        max_percentage_error=max_percentage_error,
                        mean_percentage_error=mean_percentage_error,
                    )
                )

        self.pd_ = ot.pd
        self.mean_percentage_error_, self.max_percentage_error_ = (
            mean_percentage_error,
            max_percentage_error,
        )
        return self

    def get_params(self) -> dict[str, Any]:
        """
        Get the parameters of this instance as a dictionary.
        """

        return dict(tol=self.tol, n_iter=self.n_iter, damp_param=self.damp_param)
