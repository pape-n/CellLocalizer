"""Data model for a loaded multichannel 3D image."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ChannelInfo:
    index: int
    name: str                          # name in the file, e.g. "AF647-T8"
    fluor: str | None = None           # e.g. "Alexa Fluor 647"
    excitation_nm: float | None = None
    emission_nm: float | None = None
    color: str | None = None           # display colour from the file, "#RRGGBB"

    @property
    def short(self) -> str:
        """Short label for tables: fluorophore if known, else the channel name."""
        return self.fluor or self.name

    def describe(self) -> str:
        parts = [self.short]
        if self.excitation_nm and self.emission_nm:
            parts.append(f"ex {self.excitation_nm:.0f} / em {self.emission_nm:.0f} nm")
        return ", ".join(parts)


@dataclass
class ImageStack:
    """Image data as (C, Z, Y, X) with voxel size in µm as (dz, dy, dx).

    origin is the voxel offset (z, y, x) of this stack inside the source image.
    A full image has origin (0, 0, 0). A cell crop keeps the offset, so positions
    stay those of the original image.
    """

    data: np.ndarray
    voxel_um: tuple[float, float, float]
    channels: list[ChannelInfo]
    source: Path | None = None
    origin: tuple[int, int, int] = (0, 0, 0)
    calibrated: bool = True
    info: dict = field(default_factory=dict)
    is_crop: bool = False

    def __post_init__(self) -> None:
        if self.data.ndim != 4:
            raise ValueError(f"expected (C, Z, Y, X) data, got shape {self.data.shape}")
        if len(self.channels) != self.data.shape[0]:
            raise ValueError("number of channel descriptions does not match the data")

    @property
    def n_channels(self) -> int:
        return self.data.shape[0]

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return tuple(self.data.shape[1:])

    @property
    def size_um(self) -> tuple[float, float, float]:
        return tuple(n * s for n, s in zip(self.shape_zyx, self.voxel_um))

    @property
    def nbytes_mb(self) -> float:
        return self.data.nbytes / 1e6

    @property
    def translate_um(self) -> tuple[float, float, float]:
        """Physical position of voxel (0, 0, 0) of this stack in the source image."""
        return tuple(o * s for o, s in zip(self.origin, self.voxel_um))

    def channel_signature(self) -> tuple[str, ...]:
        """Identifies the channel layout, used to reuse a saved panel."""
        return tuple(c.short for c in self.channels)

    @property
    def display_name(self) -> str:
        if self.source:
            return self.source.name
        return "Synthetic example" if self.info.get("format") == "synthetic" else "unnamed image"
