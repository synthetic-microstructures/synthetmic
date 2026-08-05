from dataclasses import asdict, dataclass, fields
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
        Target volumes or areas of the N cells. If None, each volume will be set
        to total volume / N.
    initial_weights : FloatArray or None, optional, shape (N,)
        Diagram weights. If None, will be initialised to an array of
        zeros.
    """

    domain: FloatArray
    seeds: FloatArray
    phases: IntArray | StrArray
    periodic: BoolSequence
    volumes: FloatArray
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
        vd.check_domain(domain)

        vd.check_seeds(seeds=seeds, domain=domain)

        n_grains, space_dim = seeds.shape

        phases = vd.check_phases(phases=phases, n_grains=n_grains)

        periodic = vd.check_periodic(periodic=periodic, space_dim=space_dim)

        volumes = vd.check_volumes(
            volumes=volumes,
            domain=domain,
            n_grains=n_grains,
        )

        initial_weights = vd.check_initial_weights(
            initial_weights=initial_weights, n_grains=n_grains
        )

        vd.check_num_samples(
            seeds=seeds, phases=phases, initial_weights=initial_weights, volumes=volumes
        )
        vd.check_space_dim(seeds=seeds, domain=domain, periodic=periodic)

        self.domain = domain
        self.seeds = seeds
        self.phases = phases
        self.periodic = periodic
        self.volumes = volumes
        self.initial_weights = initial_weights

    def set_domain(self, domain: FloatArray) -> None:
        """
        Set to a new domain.
        """
        vd.check_domain(domain)
        vd.check_seeds(seeds=self.seeds, domain=domain)
        _ = vd.check_volumes(
            volumes=self.volumes, domain=domain, n_grains=self.seeds.shape[0]
        )
        vd.check_space_dim(seeds=self.seeds, domain=domain, periodic=self.periodic)

        self.domain = domain

    def set_seeds(self, seeds: FloatArray) -> None:
        """
        Set to new seeds.
        """
        vd.check_seeds(seeds=seeds, domain=self.domain)
        vd.check_num_samples(
            seeds=seeds,
            phases=self.phases,
            initial_weights=self.initial_weights,
            volumes=self.volumes,
        )
        vd.check_space_dim(seeds=seeds, domain=self.domain, periodic=self.periodic)

        self.seeds = seeds

    def set_phases(self, phases: IntArray | StrArray | None) -> None:
        """
        Set to new phases.
        """
        phases = vd.check_phases(phases=phases, n_grains=self.seeds.shape[0])
        vd.check_num_samples(
            seeds=self.seeds,
            phases=phases,
            initial_weights=self.initial_weights,
            volumes=self.volumes,
        )

        self.phases = phases

    def set_periodic(self, periodic: BoolSequence | None) -> None:
        """
        Set to new periodic.
        """
        periodic = vd.check_periodic(periodic=periodic, space_dim=self.seeds.shape[1])
        vd.check_space_dim(seeds=self.seeds, domain=self.domain, periodic=periodic)

        self.periodic = periodic

    def set_volumes(self, volumes: FloatArray | None) -> None:
        """
        Set to new volumes.
        """
        volumes = vd.check_volumes(
            volumes=volumes,
            domain=self.domain,
            n_grains=self.seeds.shape[0],
        )
        vd.check_num_samples(
            seeds=self.seeds,
            phases=self.phases,
            initial_weights=self.initial_weights,
            volumes=volumes,
        )

        self.volumes = volumes

    def set_initial_weights(self, initial_weights: FloatArray | None) -> None:
        """
        Set to new weights.
        """
        initial_weights = vd.check_initial_weights(
            initial_weights=initial_weights, n_grains=self.seeds.shape[0]
        )
        vd.check_num_samples(
            seeds=self.seeds,
            phases=self.phases,
            initial_weights=initial_weights,
            volumes=self.volumes,
        )

        self.initial_weights = initial_weights

    def to_npz(self, file: Path | str) -> None:
        """
        Export config to npz file.

        Parameters
        ----------
        file: file, str, pathlib.Path
            Either the filename (string) or an open file (file-like object) where
            the data will be saved. If file is a string or a Path,
            the .npz extension will be appended to the filename if it is not already there.
        """

        np.savez(file, **asdict(self), allow_pickle=True)

    @staticmethod
    def load(file: Path | str) -> "DiagramConfig":
        """
        Load diagram config from npz file.

        Parameters
        ----------
        file : file, str, pathlib.Path
            File to load configuration from.

        Returns
        -------
        loaded : synthetmic.DiagramConfig
            Loaded diagram config.
        """
        params = dict(np.load(file=file, allow_pickle=True))
        params["periodic"] = tuple(map(bool, params["periodic"]))

        return DiagramConfig(**params)


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
    seeds : FloatArray, shape (n_grains, d)
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
    periodic : BoolSequence, len d
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
    volumes : FloatArray, shape (n_grains,)
    """
    return (np.ones(n_grains) / n_grains) * domain_volume
