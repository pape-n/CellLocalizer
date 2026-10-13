"""Cell crops: a box in voxel coordinates cut from the original grid without resampling."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .image import ImageStack


@dataclass(frozen=True)
class CropBox:
    """Half-open voxel ranges [z0, z1) x [y0, y1) x [x0, x1) in source-image coordinates."""

    z0: int
    z1: int
    y0: int
    y1: int
    x0: int
    x1: int

    def clipped(self, shape_zyx: tuple[int, int, int]) -> "CropBox":
        nz, ny, nx = shape_zyx
        box = CropBox(
            max(0, self.z0), min(nz, self.z1),
            max(0, self.y0), min(ny, self.y1),
            max(0, self.x0), min(nx, self.x1),
        )
        if box.z1 <= box.z0 or box.y1 <= box.y0 or box.x1 <= box.x0:
            raise ValueError("The box lies outside the image or is empty.")
        return box

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.z1 - self.z0, self.y1 - self.y0, self.x1 - self.x0)

    def touches_border(self, shape_zyx: tuple[int, int, int]) -> bool:
        _, ny, nx = shape_zyx
        return self.y0 == 0 or self.x0 == 0 or self.y1 == ny or self.x1 == nx

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CropBox":
        return cls(**{k: int(d[k]) for k in ("z0", "z1", "y0", "y1", "x0", "x1")})


def box_from_rectangle(vertices_yx: np.ndarray, shape_zyx: tuple[int, int, int],
                       z_range: tuple[int, int] | None = None) -> CropBox:
    """Box from the corner points of a drawn rectangle (voxel coordinates, y and x).

    Fractional corners are widened to whole voxels so the drawn area is fully included.
    z_range is inclusive (first, last plane). None means the full stack.
    """
    v = np.asarray(vertices_yx, dtype=float)
    y0, x0 = np.floor(v.min(axis=0)).astype(int)
    y1, x1 = np.ceil(v.max(axis=0)).astype(int)
    if z_range is None:
        z0, z1 = 0, shape_zyx[0]
    else:
        z0, z1 = int(z_range[0]), int(z_range[1]) + 1
    return CropBox(z0, z1, int(y0), int(y1), int(x0), int(x1)).clipped(shape_zyx)


def crop(stack: ImageStack, box: CropBox) -> ImageStack:
    """Cut a box out of a full image. Voxel values are copied unchanged."""
    if stack.is_crop:
        raise ValueError("crop() expects the full image, not a crop.")
    box = box.clipped(stack.shape_zyx)
    data = stack.data[:, box.z0:box.z1, box.y0:box.y1, box.x0:box.x1].copy()
    return ImageStack(
        data=data,
        voxel_um=stack.voxel_um,
        channels=stack.channels,
        source=stack.source,
        origin=(box.z0, box.y0, box.x0),
        calibrated=stack.calibrated,
        info=dict(stack.info),
        is_crop=True,
    )
