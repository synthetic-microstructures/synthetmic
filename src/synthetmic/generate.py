import itertools
import tempfile
from pathlib import Path
from typing import Any, Self

import numpy as np
import pyvista as pv
from pysdot import OptimalTransport
from pysdot.domain_types import ConvexPolyhedraAssembly

from synthetmic.utils import (
    NotFittedError,
    validate_fit_args,
    validate_generator_params,
)


class LaguerreDiagramGenerator:
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
        self.tol = tol
        self.n_iter = n_iter
        self.damp_param = damp_param
        self.verbose = verbose

        self.space_dim_: int | None = None
        self.optimal_transport_: OptimalTransport | None = None
        self.max_percentage_error_: float | None = None
        self.mean_percentage_error_: float | None = None

    def _print_msg(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _update_optimal_transport(self, optimal_transport: OptimalTransport) -> None:
        self.optimal_transport_ = optimal_transport

    def _update_errors(self, y: np.ndarray) -> None:
        percentage_errors = np.array(
            100.0 * np.abs(self.optimal_transport_.pd.integrals() - y) / y
        )
        self.max_percentage_error_ = percentage_errors.max()
        self.mean_percentage_error_ = percentage_errors.mean()

    def _ensure_fitted(self) -> NotFittedError | None:
        """Ensure generator is fitted before use."""

        if self.optimal_transport_ is None:
            raise NotFittedError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                f"Call 'fit' with appropriate arguments before using this estimator."
            )

        return None

    def fit(
        self,
        seeds: np.ndarray,
        volumes: np.ndarray,
        domain: np.ndarray,
        periodic: list[bool] | None = None,
        init_weights: np.ndarray | None = None,
    ) -> Self:
        """This function implements Algorithm 1 and 2 from the following paper:

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

        self

        """

        validate_generator_params(
            tol=self.tol,
            n_iter=self.n_iter,
            damp_param=self.damp_param,
            verbose=self.verbose,
        )

        validate_fit_args(
            seeds=seeds,
            volumes=volumes,
            domain=domain,
            periodic=periodic,
            init_weights=init_weights,
        )

        self.space_dim_ = domain.shape[0]

        # Build the domain
        omega = ConvexPolyhedraAssembly()

        mins = domain[:, 0].copy()
        maxs = domain[:, 1].copy()
        lens = domain[:, 1] - domain[:, 0]
        if periodic is not None:
            for k, p in enumerate(periodic):
                if p:
                    mins[k] = mins[k] - lens[k]
                    maxs[k] = maxs[k] + lens[k]
        omega.add_box(mins, maxs)

        # Turn the relative percentage error into an absolute error tolerance
        # by using the smallest volume
        err_tol = np.min(volumes) * self.tol / 100.0

        # If weight is None, then set the initial guess to zero
        if init_weights is None:
            init_weights = np.zeros(seeds.shape[0])

        # Set up the optimal transport problem
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
            periodic_dict = {True: [-1, 0, 1], False: [0]}
            periodic_list = [periodic_dict[p] for p in periodic]

            cartesian_periodic = list(itertools.product(*periodic_list))

            for rep in cartesian_periodic:
                if rep != (0, 0, 0):
                    optimal_transport.pd.add_replication(rep * lens)

        if self.n_iter == 0:
            # Algorithm 1
            # Solve the optimal transport problem (solve for the weights)
            optimal_transport.adjust_weights()

            self._update_optimal_transport(optimal_transport=optimal_transport)
            self._update_errors(y=volumes)

            return self

        # Algorithm 2
        MIN_VOL_TOL = 1e-10
        x_0 = seeds
        for k in range(self.n_iter):
            # Calculate the centroids: damped Lloyd step
            x_k = (
                1 - self.damp_param
            ) * x_0 + self.damp_param * optimal_transport.get_centroids()

            # Update the seed positions
            optimal_transport.set_positions(x_k)

            # Solve the optimal transport problem.
            # If moving the seeds and maintaining the weights gives a cell that is
            # empty, then we have a bad initial guess for the OT solver.
            # Check whether the smallest volume is bigger than some tolerance:
            # if so, then use the same weights; if not, reset the weights to zero.
            m = optimal_transport.pd.integrals()
            if np.min(m) > MIN_VOL_TOL:
                # Solve for the weights
                optimal_transport.adjust_weights()
            else:
                self._print_msg("Resetting weights to init_weights")
                optimal_transport.set_weights(init_weights)
                # Solve for the weights
                optimal_transport.adjust_weights()

            self._update_optimal_transport(optimal_transport=optimal_transport)
            self._update_errors(y=volumes)

            self._print_msg(
                f"iteration: {k + 1}/{self.n_iter}, max_percentage_error: {self.max_percentage_error_:.4f}%, "
                f"mean_percentage_error: {self.mean_percentage_error_:.4f}%"
            )

            x_0 = x_k

        return self

    def get_fitted_volumes(self) -> np.ndarray:
        """Get the fitted volumes after fitting generator."""
        self._ensure_fitted()

        return self.optimal_transport_.pd.integrals()

    def get_mesh(self) -> pv.PolyData | pv.UnstructuredGrid:
        """Get the underlying mesh as a pyvista PolyData or UnstructuredGrid data object."""

        self._ensure_fitted()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".vtk", delete=True
        ) as tmp_file:
            filename = tmp_file.name

            self.optimal_transport_.pd.display_vtk(
                filename, points=None, centroids=None
            )

            mesh = pv.read(filename)

        return mesh

    def get_positions(self) -> np.ndarray:
        """Get the final positions of initial seeds used for generating the laguerre diagram."""

        self._ensure_fitted()

        return self.optimal_transport_.pd.get_positions()

    def get_centroids(self) -> np.ndarray:
        """Get the centroids of the cells in the laguerre diagram."""

        self._ensure_fitted()

        return self.optimal_transport_.pd.centroids()

    def get_vertices(self) -> dict[int, list]:
        """Get the vertices of the cells in the laguerre diagram.

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

        self._ensure_fitted()

        res = {}

        if self.space_dim_ == 2:
            offsets, coords = self.optimal_transport_.pd.cell_polyhedra()

            for i in range(len(offsets) - 1):
                s, e = offsets[i : i + 2]
                res[i] = coords[s:e].tolist()

        elif self.space_dim_ == 3:
            offsets_polyhedra, offsets_polygon, coords = (
                self.optimal_transport_.pd.cell_polyhedra()
            )

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
        """Get the weights of the laguerre diagram."""

        self._ensure_fitted()

        return self.optimal_transport_.pd.get_weights()

    def diagram_to_vtk(self, filename: str | Path) -> None:
        """Write the generated diagram to .vkt file; filename must ends with .vtk."""

        self._ensure_fitted()

        self.optimal_transport_.pd.display_vtk(
            filename=filename, points=False, centroids=False
        )

        return None

    def get_params(self) -> dict[str, Any]:
        """Get the parameters of this instance as a dictionary."""

        return dict(
            tol=self.tol,
            n_iter=self.n_iter,
            damp_param=self.damp_param,
            verbose=self.verbose,
        )
