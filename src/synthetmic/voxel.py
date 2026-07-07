from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

import numpy as np
from pysdot import PowerDiagram

from synthetmic.types import IntSequence
from synthetmic.utils import assign_points_to_grains, compute_cell_centers


@dataclass(frozen=True, slots=True)
class DamaskGeomGridArgs:
    material: np.ndarray
    size: np.ndarray
    origin: np.ndarray

    def __post_init__(self) -> None:
        if not all(len(x) == 3 for x in (self.material.shape, self.size, self.origin)):
            raise ValueError("all space dimensions must match")

    def to_npz(self, file: Path) -> None:
        np.savez(file, **asdict(self), allow_pickle=True)


class Voxeliser:
    def __init__(self, pd: PowerDiagram, domain: np.ndarray | None = None) -> None:

        if domain is None:
            origin = pd.get_domain().min_position()
            size = pd.get_domain().max_position() - origin

        else:
            domain = np.asarray(domain)
            origin = domain[:, 0]
            size = domain[:, 1] - origin

        self.pd = pd
        self.domain = domain

        self._origin = origin
        self._size = size

    def fit(self, points_per_dim: IntSequence, n_jobs: int = -1) -> Self:
        space_dim = len(points_per_dim)
        if not all(len(x) == space_dim for x in (self._size, self._origin)):
            raise ValueError("all space dimensions must match")

        centers = compute_cell_centers(
            origin=self._origin, size=self._size, points_per_dim=points_per_dim
        )
        centers = centers.reshape(-1, space_dim)
        grain_indices = assign_points_to_grains(
            points=centers, pd=self.pd, domain=self.domain, n_jobs=n_jobs
        )
        self.voxels_ = grain_indices.reshape(points_per_dim, order="F")

        return self

    def to_damask_geomgrid_args(self, file: Path | str) -> None:

        file = Path(file)
        DamaskGeomGridArgs(
            material=self.voxels_, size=self._size, origin=self._origin
        ).to_npz(file)
