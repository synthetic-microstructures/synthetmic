from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from synthetmic import _validate as vd
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

    def to_damask_config_material_args(self, file: Path | str) -> None:
        """
        Write grain data to damask.ConfigMaterial class
        arguments. This will be written to a numpy .npz
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

    def to_damask_geom_grid_args(self, file: Path | str) -> None:
        """
        Write fields to damask.GeomGrid class
        arguments. This will be written to a numpy .npz
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


@dataclass(slots=True)
class DiagramConfig:
    """
    Configuration data class for both Voronoi and Laguerre
    diagrams.

    Attributes
    ----------
    seeds : FloatArray, shape (N,d)
        Locations of the N seeds.
    volumes : FloatArray, shape (N,)
        Target volumes or areas of the N cells.
    domain : FloatArray, shape (d,2)
        minimum and maximum coordinates of the box in each of the d dimensions
        (d=2,3).
    phases : IntArray or StrArray, shape (N,)
        Phases of the N cells.
    periodic : sequence of bool or None, optional, length d
        Sequence of bool indicating whether or not the domain is periodic in
        the different directions. None indicates no periodicity in any direction.
    initial_weights : FloatArray or None, optional, shape (N,)
        Initial guess for the weights.
    """

    seeds: FloatArray
    domain: FloatArray
    volumes: FloatArray | None = None
    phases: IntArray | StrArray | None = None
    initial_weights: FloatArray | None = None
    periodic: BoolSequence | None = None

    def __post_init__(self) -> None:
        vd.compose_rules(
            vd.is_instance(np.ndarray), vd.check_array(allowed_types=[float, int])
        )(self.seeds, "seeds")

        if self.volumes is not None:
            vd.compose_rules(
                vd.is_instance(np.ndarray), vd.check_array(allowed_types=[float, int])
            )(self.volumes, "volumes")

        if self.phases is not None:
            vd.compose_rules(
                vd.is_instance(np.ndarray),
                vd.check_array(allowed_types=[int, str]),
            )(self.phases, "phases")

        vd.compose_rules(
            vd.is_instance(np.ndarray),
            vd.check_array(allowed_types=[float, int], allowed_shapes=[(2, 2), (3, 2)]),
        )(self.domain, "domain")

        vd.is_instance(list, tuple, allow_none=True)(self.periodic, "periodic")
        if self.periodic is not None:
            vd.check_periodic(self.periodic, "periodic")

        vd.is_instance(np.ndarray, allow_none=True)(
            self.initial_weights, "initial_weights"
        )
        if self.initial_weights is not None:
            vd.check_array(allowed_types=[float, int])(
                self.initial_weights, "initial_weights"
            )

        # check if the number of samples match
        num_samples = []
        for x in (self.seeds, self.volumes, self.initial_weights, self.phases):
            if x is not None:
                num_samples.append(x.shape[0])

        if len(set(num_samples)) > 1:
            raise ValueError(
                f"one or more of seeds, volumes, and initial_weights have inconsistent number of samples: {num_samples}."
            )

        # check if space dimensions match
        space_dims = [self.seeds.shape[1], self.domain.shape[0]]
        if self.periodic is not None:
            space_dims.append(len(self.periodic))

        if len(set(space_dims)) > 1:
            raise ValueError(
                f"one or more of seeds, domain, and periodic have inconsistent space dimension: {space_dims}."
            )

        if not set(space_dims).issubset({2, 3}):
            raise ValueError(f"""one or more of seeds, domain, and periodic have wrong space dimension: {space_dims}.
                Supported space dimensions are 2 and 3.""")


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


def create_periodicity(space_dim: int, is_periodic: bool) -> list[bool] | None:
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
    list[bool] or None
    """
    return [True] * space_dim if is_periodic else None


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
