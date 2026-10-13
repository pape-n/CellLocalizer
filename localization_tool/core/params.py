"""Analysis parameters with their defaults (METRICS.md). All values are user settings and saved with the analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

THRESHOLD_METHODS = ("otsu", "triangle", "li", "halfmax", "manual")
ACTIN_DEFINITIONS = ("threshold", "shell")
Z_REFERENCES = ("auto", "first_plane", "last_plane")
MASK_KEYS = ("cell", "nucleus", "actin", "actin_rich", "target", "comparison")
TARGET_SIGNALS = ("mask", "all")             # D89: count the target signal inside its mask, or all of it (D30)
POLARITY_FROM = ("actin_rich", "actin_signal")   # D88: front end of the axis


@dataclass
class MaskParams:
    """One automatic mask: Gaussian smoothing, a global threshold, a factor on that threshold."""

    sigma_um: float
    method: str = "otsu"
    factor: float = 1.0                 # threshold multiplier, < 1 widens the mask, > 1 narrows it
    manual_threshold: float | None = None
    min_volume_um3: float = 0.0         # components below this volume are dropped

    def __post_init__(self) -> None:
        if self.method not in THRESHOLD_METHODS:
            raise ValueError(f"unknown threshold method {self.method!r}")
        if self.method == "manual" and self.manual_threshold is None:
            raise ValueError("method 'manual' needs manual_threshold")
        if self.sigma_um < 0 or self.factor <= 0:
            raise ValueError("sigma must be >= 0 and factor > 0")


def _cell_default() -> MaskParams:
    # Li again since 1.2.1 (D69): the global half maximum of 1.2.0 (D54) cut through the actin cortex in
    # the upper planes of flat cells, where the stain is dimmer than at the coverslip. Li lies at the
    # outer side of the cortex. "halfmax" stays selectable.
    return MaskParams(sigma_um=0.1, method="li", min_volume_um3=5.0)


def _nucleus_default() -> MaskParams:
    # half maximum puts the edge where the DAPI has fallen halfway to the cytoplasm, also in compact
    # cells; 0.2 µm smoothing is then enough and keeps thin lobes in shape (D51, replaces the 0.3 µm
    # of D40 that Otsu needed against holes in the chromatin)
    return MaskParams(sigma_um=0.2, method="halfmax", min_volume_um3=0.5)


def _actin_default() -> MaskParams:
    return MaskParams(sigma_um=0.1, method="li")


def _actin_rich_default() -> MaskParams:
    return MaskParams(sigma_um=0.1, method="otsu")


def _target_default() -> MaskParams:
    # Li marks all visible target signal, Otsu only the brightest part (D40)
    return MaskParams(sigma_um=0.15, method="li")


def _comparison_default() -> MaskParams:
    # a second stained protein (D83), its area found like the target
    return MaskParams(sigma_um=0.15, method="li")


@dataclass
class AnalysisParams:
    cell: MaskParams = field(default_factory=_cell_default)
    nucleus: MaskParams = field(default_factory=_nucleus_default)
    actin: MaskParams = field(default_factory=_actin_default)            # all actin, compartment (D28)
    actin_rich: MaskParams = field(default_factory=_actin_rich_default)  # brightest actin, zone (D28)
    target: MaskParams = field(default_factory=_target_default)          # stained target signal (D30, D89)
    comparison: MaskParams = field(default_factory=_comparison_default)  # area of a comparison channel (D83)
    target_signal: str = "mask"              # D89: "mask" counts the target signal inside its mask, "all" all of it
    polarity_from: str = "actin_rich"        # D88: axis towards the actin-rich mask, or the whole actin signal
    polarity_clear_index: float = 0.25       # D84: polarity index from which the front direction counts as clear
    envelope_lateral_only: bool = False      # D86 (c): nuclear envelope zones only at the side of the nucleus
    nucleus_max_hole_um2: float = 0.25      # holes per plane up to this area count as nucleus
    nucleus_dim_hole_fraction: float = 0.0  # > 0: holes of any size with DAPI >= this x threshold count as nucleus (D51, off)
    actin_definition: str = "threshold"      # D16: actin compartment by threshold, or "shell" under the surface
    background: dict[str, float] = field(default_factory=dict)   # channel label -> value, default 0
    envelope_um: float = 0.3                 # half width of the nuclear envelope zone (D13)
    cortex_um: float = 0.5                   # cortex shell thickness (D13, D16)
    basal_um: float = 0.5                    # basal layer thickness (D13)
    profile_bin_um: float = 0.2
    z_reference: str = "auto"                # which plane lies at the coverslip (D14, D20)
    axial_scale: float = 1.0                 # correction of the nominal z step (refractive index, D23)
    truncation_fraction: float = 0.2         # QC: cell area in the first or last plane relative to the widest plane
    bottom_fraction: float = 0.6             # QC: coverslip plane this bright relative to the brightest plane: stack starts inside the cell (D67)
    saturation_fraction: float = 0.001       # QC: share of cell voxels at the maximum of the data type
    colocalization: bool = True              # Pearson and Manders with Costes threshold (D35)
    polarity: bool = True                    # front-rear axis from the actin channel (D36)
    crosstalk: list = field(default_factory=list)   # [{"into": label, "from": label, "k": 0.06}], off by default (D27, D38)
    channel_offsets: list = field(default_factory=list)  # [{"channel": label, "dz_um", "dy_um", "dx_um", "fluor"?}] from beads (D55)
    sensitivity: bool = True                 # shares again with cell and nucleus edge factors 0.8 and 1.2 (D56)
    actin_per_plane: bool = True             # actin also above a half-maximum threshold of its own plane (D70)

    def __post_init__(self) -> None:
        if self.actin_definition not in ACTIN_DEFINITIONS:
            raise ValueError(f"unknown actin definition {self.actin_definition!r}")
        if self.z_reference not in Z_REFERENCES:
            raise ValueError(f"unknown z reference {self.z_reference!r}")
        if not 0.0 <= self.bottom_fraction <= 1.0:
            raise ValueError("bottom_fraction must be between 0 and 1")
        if self.axial_scale <= 0:
            raise ValueError("axial_scale must be > 0")
        if self.target_signal not in TARGET_SIGNALS:
            raise ValueError(f"unknown target signal {self.target_signal!r}")
        if self.polarity_from not in POLARITY_FROM:
            raise ValueError(f"unknown polarity axis {self.polarity_from!r}")
        if self.polarity_clear_index < 0:
            raise ValueError("polarity_clear_index must be >= 0")
        for key in MASK_KEYS:
            if key not in ("nucleus", "cell") and getattr(self, key).method == "halfmax":
                raise ValueError(f"the half-maximum threshold is only available for nucleus and cell edge, not {key!r}")
        if not 0.0 <= self.nucleus_dim_hole_fraction < 1.0:
            raise ValueError("nucleus_dim_hole_fraction must be between 0 and 1")
        for name in ("envelope_um", "cortex_um", "basal_um", "profile_bin_um"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        clean = []
        for c in self.crosstalk or []:
            c = dict(c)
            if not c.get("into") or not c.get("from") or c["into"] == c["from"]:
                raise ValueError("crosstalk needs two different channel labels ('into' and 'from')")
            k = float(c.get("k", 0.0))
            if not 0.0 <= k < 1.0:
                raise ValueError("crosstalk coefficient k must be between 0 and 1")
            clean.append({"into": str(c["into"]), "from": str(c["from"]), "k": k})
        self.crosstalk = clean
        offs = []
        for o in self.channel_offsets or []:
            o = dict(o)
            if not o.get("channel"):
                raise ValueError("a channel offset needs the channel label")
            vals = {k: float(o.get(k, 0.0)) for k in ("dz_um", "dy_um", "dx_um")}
            if abs(vals["dz_um"]) > 5.0 or abs(vals["dy_um"]) > 1.0 or abs(vals["dx_um"]) > 1.0:
                raise ValueError("channel offsets above 5 µm in z or 1 µm in x and y are not plausible "
                                 "(bead measurement?)")
            entry = {"channel": str(o["channel"]), **vals}
            if o.get("fluor"):
                entry["fluor"] = str(o["fluor"])
            offs.append(entry)
        self.channel_offsets = offs

    def background_for(self, label: str) -> float:
        return float(self.background.get(label, 0.0))

    def to_dict(self) -> dict:
        return asdict(self)

    def with_overrides(self, overrides: dict | None) -> "AnalysisParams":
        """Copy with some values replaced, e.g. {"nucleus": {"factor": 0.9}} for one cell."""
        d = self.to_dict()
        for key, value in (overrides or {}).items():
            if isinstance(value, dict) and isinstance(d.get(key), dict):
                d[key] = {**d[key], **value}
            else:
                d[key] = value
        return AnalysisParams.from_dict(d)

    @classmethod
    def from_dict(cls, d: dict) -> "AnalysisParams":
        d = dict(d)
        d.pop("spots", None)                     # puncta of 0.4.0a1, removed (D34 rejected)
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown parameters: {', '.join(sorted(unknown))} (file of a newer version?)")
        try:
            for key in MASK_KEYS:
                if key in d and isinstance(d[key], dict):
                    base = asdict(getattr(cls(), key))
                    base.update(d[key])
                    d[key] = MaskParams(**base)
            return cls(**d)
        except TypeError as exc:                 # unknown key inside a group
            raise ValueError(f"invalid parameters: {exc}") from exc
