from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from synthetmic import _validate as vd
from synthetmic._validate import check_points
from synthetmic.typing import (
    BoolSequence,
    FloatArray,
    FloatSequence,
    IntArray,
    StrArray,
)


@dataclass(frozen=True, slots=True)
class GrainData:
    """
    Represents grain data.

    Attributes
    ----------
    orientations : FloatArray
        Grain orientations.
    phases : IntArray or StrArray
        Grain phases
    volumes : FloatArray
        Grain volumes.
    """

    orientations: FloatArray
    phases: IntArray | StrArray
    volumes: FloatArray

    def __post_init__(self):
        lengths = {f.name: getattr(self, f.name).shape[0] for f in fields(self)}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"All arrays must have the same length, got {lengths}.")

    def to_damask_configmaterial_params(self, file: Path | str) -> None:
        """
        Write grain data to damask.ConfigMaterial class
        parameters. This will be written to a numpy .npz
        file with the following field names:
        "O": orientations, "phase": phases.

        Parameters
        ----------
        file: file, str, pathlib.Path
            Either the filename (string) or an open file (file-like object) where
            the data will be saved. If file is a string or a Path,
            the .npz extension will be appended to the filename if it is not already there.
        """
        np.savez(
            file, **{"O": self.orientations, "phase": self.phases}, allow_pickle=True
        )


@dataclass(frozen=True, slots=True)
class VoxelGrid:
    """
    Represents the results of diagram voxelisation.

    Attributes
    ----------
    voxels : IntArray, shape (:, :) or (:, :, :)
    origin : FloatSequence, len 2 or 3
    size : FloatSequence, len 2 or 3
    """

    voxels: IntArray
    origin: FloatSequence
    size: FloatSequence

    def __post_init__(self) -> None:
        if not all(len(x) == self.voxels.ndim for x in (self.size, self.origin)):
            raise ValueError("All space dimensions must match.")

    def to_damask_geomgrid_params(self, file: Path | str) -> None:
        """
        Write fields to damask.GeomGrid class
        parameters. This will be written to a numpy .npz
        file with the following field names:
        "material": voxels, "origin": origin, "size": size.

        Calling this method on a 2D VoxelGrid will thrown an error,
        as damask.GeomGrid only works on 3D microstructures.

        Parameters
        ----------
        file: file, str, pathlib.Path
            Either the filename (string) or an open file (file-like object) where
            the data will be saved. If file is a string or a Path,
            the .npz extension will be appended to the filename if it is not already there.
        """

        if self.voxels.ndim != 3:
            raise ValueError(
                "Damask GeomGrid args can only be written for 3D microstructure."
            )
        np.savez(
            file,
            **{"material": self.voxels, "origin": self.origin, "size": self.size},
            allow_pickle=True,
        )


@dataclass(slots=True, init=False)
class DiagramConfig:
    """
    Configuration data class for both Voronoi and Laguerre
    diagrams.

    Parameters
    ----------
    domain : FloatArray, shape (d,2)
        minimum and maximum coordinates of the box in each of the d dimensions
        (d=2,3).
    seeds : FloatArray, shape (N,d)
        Locations of the N seeds.
    phases : IntArray or StrArray shape (N,) or None, optional
        Phases of the N cells. If None, will be initialised to
        an array of zeros.
    periodic : BoolSequence or None, optional, length d
        Sequence of bool indicating whether or not the domain is periodic in
        the different directions. If None, will be initialised to fully non-periodic.
    volumes : FloatArray shape (N,) or None
        Target volumes or areas of the N cells. This should be set to None
        for Voronoi diagram configuration and must be set to a valid 1D array
        for Laguerre diagram configuration.
    initial_weights : FloatArray or None, optional, shape (N,)
        Diagram weights. If None, will be initialised to an array of
        zeros.
    """

    domain: FloatArray
    seeds: FloatArray
    phases: IntArray | StrArray
    periodic: BoolSequence
    volumes: FloatArray | None
    initial_weights: FloatArray

    def __init__(
        self,
        domain: FloatArray,
        seeds: FloatArray,
        phases: IntArray | StrArray | None = None,
        periodic: BoolSequence | None = None,
        volumes: FloatArray | None = None,
        initial_weights: FloatArray | None = None,
    ) -> None:
        vd.compose_rules(
            vd.is_instance(np.ndarray),
            vd.check_array(allowed_types=[float, int], allowed_ndims=[2, 3]),
        )(seeds, "seeds")
        check_points(seeds)

        n_grains, space_dim = seeds.shape

        if phases is None:
            phases = np.zeros(n_grains, dtype=int)
        vd.compose_rules(
            vd.is_instance(np.ndarray),
            vd.check_array(allowed_types=[int, str], allowed_ndims=[1]),
        )(phases, "phases")

        if initial_weights is None:
            initial_weights = np.zeros(n_grains, dtype=float)
        vd.compose_rules(
            vd.is_instance(np.ndarray),
            vd.check_array(allowed_types=[float, int], allowed_ndims=[1]),
        )(initial_weights, "initial_weights")

        vd.compose_rules(
            vd.is_instance(np.ndarray),
            vd.check_array(allowed_types=[float, int], allowed_shapes=[(2, 2), (3, 2)]),
        )(domain, "domain")

        vd.is_instance(np.ndarray, allow_none=True)(volumes, "volumes")
        if volumes is not None:
            vd.check_array(allowed_types=[float, int], allowed_ndims=[1])(
                volumes, "volumes"
            )

        if periodic is None:
            periodic = create_periodicity(space_dim=space_dim, is_periodic=False)
        vd.compose_rules(vd.is_instance(list, tuple), vd.check_periodic())(
            periodic, "periodic"
        )

        # check if the number of samples match
        num_samples = [
            seeds.shape[0],
            phases.shape[0],
            initial_weights.shape[0],
        ]
        if volumes is not None:
            num_samples.append(volumes.shape[0])

        if len(set(num_samples)) > 1:
            raise ValueError(
                f"one or more of seeds, volumes, and initial_weights have inconsistent number of samples: {num_samples}."
            )

        # check if space dimensions match
        space_dims = [space_dim, domain.shape[0], len(periodic)]
        if len(set(space_dims)) > 1:
            raise ValueError(
                f"one or more of seeds, domain, and periodic have inconsistent space dimension: {space_dims}."
            )
        if not set(space_dims).issubset({2, 3}):
            raise ValueError(f"""one or more of seeds, domain, and periodic have wrong space dimension: {space_dims}.
                Supported space dimensions are 2 and 3.""")

        self.domain = domain
        self.seeds = seeds
        self.phases = phases
        self.periodic = tuple(periodic)
        self.volumes = volumes
        self.initial_weights = initial_weights


def sample_random_seeds(
    domain: FloatArray, n_grains: int, random_state: int | None = None
) -> FloatArray:
    """
    Sample random seeds from a domain or box.

    Parameters
    ----------
    domain : FloatArray
        Represents the minimum and maximum coordinates of the box
        in each of the d dimensions (d = 2 or 3).
    n_grains : int
        Number of grains.
    random_state : int or None, optional, default=None
        Pass an int for reproducibility.

    Returns
    -------
    FloatArray, shape (n_grains, d)
    """
    np.random.seed(random_state)

    return np.random.uniform(
        low=domain[:, 0], high=domain[:, 1], size=(n_grains, domain.shape[0])
    )


def create_periodicity(space_dim: int, is_periodic: bool) -> BoolSequence:
    """
    Create `periodic` field for a fully periodic or
    fully non-period diagram configuration.

    Parameters
    ----------
    space_dim : int
       Space dimension of the diagram box or domain.
    is_periodic : bool
        If True, produces field for fully periodic diagram.

    Returns
    -------
    BoolSequence, len d
    """
    return (is_periodic,) * space_dim


def create_constant_volumes(
    n_grains: int,
    domain_volume: float,
) -> FloatArray:
    """
    Create a 1D array of constant volumes.

    Parameters
    ----------
    n_grains : int
        Number of grains.
    domain_volume : float
       Volume of the domain or box.

    Returns
    -------
    FloatArray, shape (n_grains,)
    """
    return (np.ones(n_grains) / n_grains) * domain_volume
