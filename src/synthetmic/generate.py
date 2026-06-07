import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Self

import numpy as np
import pyvista as pv
from pysdot import OptimalTransport, PowerDiagram

from synthetmic._internal._errors import NotFittedError
from synthetmic.utils import (
    add_replicants,
    build_domain,
    validate_fit_args,
    validate_generator_params,
)


class DiagramGenerator(ABC):
    _pd: PowerDiagram | None = None
    _space_dim: int | None

    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose

    @abstractmethod
    def fit(self) -> Self:
        pass

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        pass

    @property
    def pd_(self):
        if self._pd is None:
            self._raise_not_fitted_error()

        return self._pd

    @property
    def space_dim_(self):
        if self._space_dim is None:
            self._raise_not_fitted_error()

        return self._space_dim

    def get_fitted_volumes(self) -> np.ndarray:
        """
        Get the computed diagram cell volumes.
        """
        return self.pd_.integrals()

    def get_mesh(self) -> pv.UnstructuredGrid | pv.PolyData:
        """
        Get the underlying diagram mesh as a pyvista PolyData or UnstructuredGrid data object.
        """

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".vtk", delete=True
        ) as tmp_file:
            filename = tmp_file.name

            self.pd_.display_vtk(filename, points=None, centroids=None)

            mesh = pv.read(filename)

        return mesh

    def get_positions(self) -> np.ndarray:
        """
        Get the final positions of seeds.
        """

        return self.pd_.get_positions()

    def get_centroids(self) -> np.ndarray:
        """
        Get the centroids of the cells in the Voronoi diagram.
        """

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
        res = {}

        if self.space_dim_ == 2:
            offsets, coords = self.pd_.cell_polyhedra()

            for i in range(len(offsets) - 1):
                s, e = offsets[i : i + 2]
                res[i] = coords[s:e].tolist()

        elif self.space_dim_ == 3:
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

    def get_weights(self) -> np.ndarray:
        """
        Get the weights of the diagram.
        """

        return self.pd_.get_weights()

    def diagram_to_vtk(self, filename: str | Path) -> None:
        """
        Write the generated diagram to .vkt file; filename must ends with .vtk.
        """

        self.pd_.display_vtk(filename=filename, points=False, centroids=False)

        return None

    def _set_pd(self, pd: PowerDiagram) -> None:
        self._pd = pd

        return None

    def _print_msg(self, msg: str) -> None:
        if self.verbose:
            print(msg)

        return None

    def _raise_not_fitted_error(self) -> None:
        raise NotFittedError(
            f"This {self.__class__.__name__} instance is not fitted yet. "
            f"Call 'fit' with appropriate arguments before using this instance."
        )


class VoronoiDiagramGenerator(DiagramGenerator):
    """
    n_iter : int, optional
        Number of iterations of Lloyd's algorithm (move each seed to the
        centroid of its cell). If it is set to 0, then no Lloyd's iteration
        will be performed.
    damp_param : float [0, 1], optional
        The damping parametr of the damped Lloyd step; value must be between
        0 and 1 (inclusive at both ends).
    verbose : bool, optional
        If set to True, print optimisation progress.
    """

    def __init__(
        self,
        n_iter: int = 5,
        damp_param: float = 1.0,
        verbose: bool = True,
    ) -> None:
        validate_generator_params(
            tol=None,
            n_iter=n_iter,
            damp_param=damp_param,
            verbose=verbose,
        )

        super().__init__(verbose)

        self.n_iter = n_iter
        self.damp_param = damp_param

    def fit(
        self,
        seeds: np.ndarray,
        domain: np.ndarray,
        periodic: list[bool] | None = None,
    ) -> Self:
        """
        Fit Voronoi diagram on the given seeds and domain specifications.

        Parameters
        ----------

        seeds : ndarray, shape (N,d)
            Locations of the N seeds.
        domain : ndarray, shape (d,2)
            minimum and maximum coordinates of the box in each of the d dimensions
            (d=2,3)
        periodic : list, optional, length d
            List of Booleans indicating whether or not the domain is periodic in
            the different directions. None indicates no periodicity in any direction.

        Returns
        -------

        synthetmic.generate.VoronoiDiagramGenerator
        """

        validate_fit_args(
            seeds=seeds,
            volumes=None,
            domain=domain,
            periodic=periodic,
            init_weights=None,
        )

        omega, lens = build_domain(domain=domain, periodic=periodic)

        num_grains, space_dim = seeds.shape
        self._space_dim = space_dim

        weights = np.zeros(num_grains)
        pd = PowerDiagram(positions=seeds, weights=weights, domain=omega)

        if periodic is not None:
            add_replicants(obj=pd, periodic=periodic, domain_lens=lens)

        if self.n_iter == 0:
            self._set_pd(pd)
            return self

        for k in range(self.n_iter):
            seeds = (1 - self.damp_param) * seeds + self.damp_param * pd.centroids()
            pd.set_positions(seeds)

            self._set_pd(pd)
            self._print_msg(
                f"iteration: {k + 1}/{self.n_iter}, norm of change in positions: {np.linalg.norm(pd.centroids() - seeds)}",
            )

        return self

    def get_params(self) -> dict[str, Any]:
        """
        Get the parameters of this instance as a dictionary.
        """

        return dict(
            n_iter=self.n_iter,
            damp_param=self.damp_param,
            verbose=self.verbose,
        )


class LaguerreDiagramGenerator(DiagramGenerator):
    """
    tol : float, optional
        Relative percentage error for volumes.
    n_iter : int, optional
        Number of iterations of Lloyd's algorithm (move each seed to the
        centroid of its cell). If it is set to 0, then no Lloyd's iteration
        will be performed.
    damp_param : float [0, 1], optional
        The damping parametr of the damped Lloyd step; value must be between
        0 and 1 (inclusive at both ends).
    verbose : bool, optional
        If set to True, print optimisation progress.
    """

    def __init__(
        self,
        tol: float = 1.0,
        n_iter: int = 5,
        damp_param: float = 1.0,
        verbose: bool = True,
    ):
        validate_generator_params(
            tol=tol,
            n_iter=n_iter,
            damp_param=damp_param,
            verbose=verbose,
        )

        super().__init__(verbose)

        self.tol = tol
        self.n_iter = n_iter
        self.damp_param = damp_param

        self.optimal_transport_: OptimalTransport | None = None
        self.max_percentage_error_: float | None = None
        self.mean_percentage_error_: float | None = None

    def _set_optimal_transport(self, optimal_transport: OptimalTransport) -> None:
        self.optimal_transport_ = optimal_transport

        return None

    def _set_errors(self, y: np.ndarray) -> None:
        percentage_errors = np.array(
            100.0 * np.abs(self.optimal_transport_.pd.integrals() - y) / y
        )
        self.max_percentage_error_ = percentage_errors.max()
        self.mean_percentage_error_ = percentage_errors.mean()

    def fit(
        self,
        seeds: np.ndarray,
        volumes: np.ndarray,
        domain: np.ndarray,
        periodic: list[bool] | None = None,
        init_weights: np.ndarray | None = None,
    ) -> Self:
        """
        This function implements Algorithm 1 and 2 from the following paper:

        Bourne, D.P., Kok, P.J.J., Roper, S.M. & Spanjer, W.D.T. (2020)
        Laguerre tessellations and polycrystalline microstructures:
        A fast algorithm for generating grains of given volumes,
        Philosophical Magazine, 100, 2677-2707.
        https://www.tandfonline.com/doi/full/10.1080/14786435.2020.1790053

        Parameters
        ----------

        seeds : ndarray, shape (N,d)
            Locations of the N seeds.
        volumes : ndarray, shape (N,)
            Target volumes or areas of the N Laguerre cells.
        domain : ndarray, shape (d,2)
            minimum and maximum coordinates of the box in each of the d dimensions
            (d=2,3)
        periodic : list, optional, length d
            List of Booleans indicating whether or not the domain is periodic in
            the different directions. None indicates no periodicity in any direction.
        init_weights : ndarray, optional, shape (N,)
            Initial guess for the weights for Algorithm 1.
            None indicates that the weights should be zero.

        Returns
        -------

        synthetmic.generate.LeguerreDiagramGenerator
        """

        validate_fit_args(
            seeds=seeds,
            volumes=volumes,
            domain=domain,
            periodic=periodic,
            init_weights=init_weights,
        )

        self._space_dim = domain.shape[0]

        # Turn the relative percentage error into an absolute error tolerance
        # by using the smallest volume
        err_tol = np.min(volumes) * self.tol / 100.0

        # If weight is None, then set the initial guess to zero
        if init_weights is None:
            init_weights = np.zeros(seeds.shape[0])

        # Set up the optimal transport problem
        omega, lens = build_domain(domain=domain, periodic=periodic)
        optimal_transport = OptimalTransport(
            positions=seeds,
            masses=volumes,
            weights=init_weights,
            domain=omega,
            obj_max_dm=err_tol,
            verbosity=0,
        )

        # If there is periodicity, then add the replicants
        if periodic is not None:
            add_replicants(obj=optimal_transport, periodic=periodic, domain_lens=lens)

        if self.n_iter == 0:
            optimal_transport.adjust_weights()

            self._set_optimal_transport(optimal_transport=optimal_transport)
            self._set_pd(optimal_transport.pd)
            self._set_errors(y=volumes)

            return self

        MIN_VOL_TOL = 1e-10
        for k in range(self.n_iter):
            seeds = (
                1 - self.damp_param
            ) * seeds + self.damp_param * optimal_transport.get_centroids()
            optimal_transport.set_positions(seeds)

            # Solve the optimal transport problem.
            # If moving the seeds and maintaining the weights gives a cell that is
            # empty, then we have a bad initial guess for the OT solver.
            # Check whether the smallest volume is bigger than some tolerance:
            # if so, then use the same weights; if not, reset the weights to zero.
            m = optimal_transport.pd.integrals()
            if np.min(m) > MIN_VOL_TOL:
                optimal_transport.adjust_weights()
            else:
                optimal_transport.set_weights(init_weights)
                optimal_transport.adjust_weights()

            self._set_optimal_transport(optimal_transport)
            self._set_pd(optimal_transport.pd)
            self._set_errors(volumes)

            self._print_msg(
                f"iteration: {k + 1}/{self.n_iter}, max_percentage_error: {self.max_percentage_error_:.4f}%, "
                f"mean_percentage_error: {self.mean_percentage_error_:.4f}%"
            )

        return self

    def get_params(self) -> dict[str, Any]:
        """
        Get the parameters of this instance as a dictionary.
        """

        return dict(
            tol=self.tol,
            n_iter=self.n_iter,
            damp_param=self.damp_param,
            verbose=self.verbose,
        )
