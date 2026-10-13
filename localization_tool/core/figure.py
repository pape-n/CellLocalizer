"""Figures of single cells for the cell viewer (D71): a plane, a vertical section at any angle, with
channels, masks and zones switched on and off, scale bar and legend, saved as PNG or as TIFF with the
unchanged voxel values.

Only for looking and showing. Nothing here changes a measured value. Rules that keep the pictures
honest: the TIFF export holds the recorded values of the shown plane or section (nearest voxel, no
interpolation, D18); the PNG of a section is stretched in z to the true aspect ratio (as in the control
images); the batch export uses one contrast for all cells unless asked otherwise, so cells stay
comparable, and it writes the settings used next to the pictures (figure_settings.json).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from . import distances
from .channels import ChannelAssignment
from .qc_images import _font, _hex

MODES = ("plane", "section")
# masks and zones the viewer can show, in drawing order (later ones on top)
MASK_NAMES = ("cytoplasm", "actin", "actin_rich", "nucleus", "envelope_zone", "cortex_shell", "basal_layer", "cell")
MASK_TITLES = {"cell": "Cell", "nucleus": "Nucleus", "cytoplasm": "Cytoplasm", "actin": "Actin",
               "actin_rich": "Actin-rich", "envelope_zone": "Nuclear envelope zone", "cortex_shell": "Cortex shell",
               "basal_layer": "Basal layer"}
# order in the viewer panel: masks of the target first, then the compartments, then the zones
PANEL_MASKS = ("cell", "nucleus", "cytoplasm", "actin", "actin_rich")
PANEL_ZONES = ("envelope_zone", "cortex_shell", "basal_layer")
MASK_COLORS = {"cell": "#FFFFFF", "nucleus": "#00C8FF", "cytoplasm": "#8A929C", "actin": "#3CFF5A",
               "actin_rich": "#FFD60A", "envelope_zone": "#FF9F1C", "cortex_shell": "#C77DFF",
               "basal_layer": "#FF5C8A", "target": "#FFFFFF"}
NICE_BARS_UM = (0.5, 1, 2, 5, 10, 20, 50, 100)


@dataclass
class ChannelView:
    label: str
    color: str
    visible: bool = True
    low: float = 0.0
    high: float = 1.0
    gamma: float = 1.0


@dataclass
class MaskView:
    name: str
    visible: bool = False
    style: str = "outline"          # "outline" or "fill"
    color: str = "#FFFFFF"


@dataclass
class FigureSettings:
    mode: str = "plane"
    z: int = -1                     # plane index in the crop, -1: the default plane (middle of the target)
    angle_deg: float = 0.0          # section direction in the xy plane, 0 = along x
    offset_um: float = 0.0          # section line shifted perpendicular to its direction from the cell centre
    scale: int = 2                  # pixels per voxel in the PNG (nearest neighbour, no invented detail)
    line_px: int = 2                # outline width in the PNG
    fill_alpha: float = 0.35
    scale_bar_um: float = 0.0       # 0: chosen automatically
    show_scale_bar: bool = True
    show_scale_text: bool = True
    show_legend: bool = False
    crop_to_cell: bool = True       # picture only around the cell (plus crop_margin_um), not the whole analysed box
    crop_margin_um: float = 1.0
    title: str = ""
    channels: list[ChannelView] = field(default_factory=list)
    masks: list[MaskView] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FigureSettings":
        d = dict(d)
        chans = [ChannelView(**c) for c in d.pop("channels", [])]
        masks = [MaskView(**m) for m in d.pop("masks", [])]
        known = set(cls.__dataclass_fields__)
        s = cls(**{k: v for k, v in d.items() if k in known})
        s.channels, s.masks = chans, masks
        if s.mode not in MODES:
            raise ValueError(f"unknown view mode {s.mode!r}")
        return s

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FigureSettings":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ------------------------------------------------------------------------------------------- masks
def view_masks(masks: dict, voxel_um, envelope_um: float, cortex_um: float, basal_um: float,
               coverslip_first: bool) -> dict[str, np.ndarray]:
    """Masks and zones for the viewer from the masks of an analysis (crop grid): the compartments as
    analysed (labels 1 nucleus, 2 actin, 3 cytoplasm without actin, 4 actin-rich) and the zones with
    the widths of the analysis settings, computed as in metrics.measure_target."""
    cell = np.asarray(masks["cell"], bool)
    lab = masks.get("labels")
    nucleus = (lab == 1) if lab is not None else np.asarray(masks.get("nucleus", np.zeros_like(cell)), bool)
    actin = np.isin(lab, (2, 4)) if lab is not None else np.asarray(masks.get("actin", np.zeros_like(cell)), bool)
    out = {"cell": cell, "nucleus": nucleus & cell, "cytoplasm": cell & ~nucleus, "actin": actin & cell,
           "actin_rich": np.asarray(masks.get("actin_rich", np.zeros_like(cell)), bool) & cell}
    if cell.any():
        if nucleus.any():
            dn = distances.signed_distance(nucleus, voxel_um)
            out["envelope_zone"] = cell & (np.abs(dn) <= envelope_um)
        else:
            out["envelope_zone"] = np.zeros_like(cell)
        out["cortex_shell"] = cell & (distances.depth_below_surface(cell, voxel_um) <= cortex_um)
        out["basal_layer"] = cell & (distances.height_above_substrate(cell, voxel_um[0], coverslip_first) <= basal_um)
    else:
        for k in ("envelope_zone", "cortex_shell", "basal_layer"):
            out[k] = np.zeros_like(cell)
    for k, v in masks.items():
        if k.startswith("target_"):
            out[k] = np.asarray(v, bool) & cell
    return out


def default_mask_views(names) -> list[MaskView]:
    """Cell and nucleus outlined, the rest off."""
    views = []
    for n in names:
        color = MASK_COLORS.get(n, MASK_COLORS["target"])
        views.append(MaskView(n, visible=n in ("cell", "nucleus"), style="outline", color=color))
    return views


def mask_title(name: str) -> str:
    return f"{name[7:]} mask" if name.startswith("target_") else MASK_TITLES.get(name, name)


# ------------------------------------------------------------------------------------------- geometry
def default_plane(data: np.ndarray, assignments: list[ChannelAssignment], masks: dict) -> int:
    """The plane of the control image: median height of the first target's signal in the cell."""
    from .qc_images import _middle_plane

    return _middle_plane(data, assignments, masks)[0]


def cell_centre(cell: np.ndarray) -> tuple[float, float]:
    if not cell.any():
        return (cell.shape[1] - 1) / 2, (cell.shape[2] - 1) / 2
    yy, xx = np.nonzero(cell.any(axis=0))
    return float(yy.mean()), float(xx.mean())


def section_line(shape_zyx, voxel_um, angle_deg: float, centre_yx, offset_um: float = 0.0):
    """Sample points (row, column) of a vertical section through the stack: a straight line in the
    coverslip plane at angle_deg to the x axis, shifted by offset_um perpendicular to it, one sample per
    lateral pixel, clipped to the image. Returns (rows, columns, step_um)."""
    _, ny, nx = shape_zyx
    dy, dx = voxel_um[1], voxel_um[2]
    step = min(dy, dx)
    a = np.deg2rad(angle_deg)
    ux, uy = np.cos(a), np.sin(a)                       # direction in µm
    cy, cx = centre_yx[0] * dy - offset_um * ux, centre_yx[1] * dx + offset_um * uy
    reach = np.hypot(ny * dy, nx * dx)
    t = np.arange(-reach, reach + step, step)
    ys, xs = (cy + t * uy) / dy, (cx + t * ux) / dx
    inside = (ys >= 0) & (ys <= ny - 1) & (xs >= 0) & (xs <= nx - 1)
    if not inside.any():
        raise ValueError("the section line does not cross the image")
    k = np.flatnonzero(inside)
    return ys[k[0]:k[-1] + 1], xs[k[0]:k[-1] + 1], step


def section(volume: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """(z, t) values along the line in every plane, nearest voxel (the recorded values, D18)."""
    r = np.clip(np.rint(rows).astype(int), 0, volume.shape[1] - 1)
    c = np.clip(np.rint(cols).astype(int), 0, volume.shape[2] - 1)
    return np.asarray(volume)[:, r, c]


# ------------------------------------------------------------------------------------------- contrast
def auto_contrast(volume: np.ndarray, cell: np.ndarray | None = None) -> tuple[float, float]:
    """0 to the 99.7th percentile inside the cell (as the control images)."""
    v = np.asarray(volume)
    sample = v[cell] if cell is not None and cell.any() else v
    hi = float(np.percentile(sample, 99.7)) if sample.size else 1.0
    if hi <= 0:
        hi = float(v.max()) or 1.0
    return 0.0, hi


def default_channel_views(data: np.ndarray, assignments: list[ChannelAssignment], cell=None) -> list[ChannelView]:
    views = []
    for i, a in enumerate(assignments):
        lo, hi = auto_contrast(data[i], cell)
        views.append(ChannelView(a.label, a.color, visible=a.role != "Ignore", low=lo, high=hi))
    return views


# ------------------------------------------------------------------------------------------- render
@dataclass
class View:
    """What one figure shows: the 2D images per channel (recorded values), the 2D masks, the pixel
    size of the raw grid (rows, columns) in µm and how strongly rows are stretched for the picture."""
    raw: np.ndarray                     # (C, rows, columns)
    masks: dict[str, np.ndarray]
    pixel_um: tuple[float, float]
    row_stretch: float = 1.0
    caption: str = ""


def make_view(data: np.ndarray, masks3d: dict[str, np.ndarray], voxel_um, settings: FigureSettings,
              coverslip_first: bool = True, default_z: int | None = None) -> View:
    """Cut the plane or section out of the crop. Sections are turned so the coverslip is at the bottom."""
    cell = masks3d["cell"]
    if settings.mode == "plane":
        z = settings.z if 0 <= settings.z < data.shape[1] else (default_z if default_z is not None
                                                                 else data.shape[1] // 2)
        ys, xs = slice(None), slice(None)
        if settings.crop_to_cell and cell.any():     # same frame in every plane: the cell in all planes
            ys, xs = _span(cell.any(axis=(0, 2)), settings.crop_margin_um / voxel_um[1]), \
                _span(cell.any(axis=(0, 1)), settings.crop_margin_um / voxel_um[2])
        return View(np.asarray(data[:, z, ys, xs]), {k: m[z, ys, xs] for k, m in masks3d.items()},
                    (voxel_um[1], voxel_um[2]),
                    caption=f"plane {z + 1} of {data.shape[1]}, pixel {voxel_um[2]:.4g} µm")
    rows, cols, step = section_line(data.shape[1:], voxel_um, settings.angle_deg, cell_centre(cell),
                                    settings.offset_um)
    raw = np.stack([section(ch, rows, cols) for ch in data])
    ms = {k: section(m, rows, cols) for k, m in masks3d.items()}
    if coverslip_first:                     # z index 0 at the coverslip: put it at the bottom
        raw, ms = raw[:, ::-1], {k: v[::-1] for k, v in ms.items()}
    if settings.crop_to_cell and ms["cell"].any():
        cs = _span(ms["cell"].any(axis=0), settings.crop_margin_um / step)
        raw, ms = raw[:, :, cs], {k: v[:, cs] for k, v in ms.items()}
    return View(raw, ms, (voxel_um[0], step), row_stretch=voxel_um[0] / step,
                caption=f"vertical section at {settings.angle_deg:g}°, offset {settings.offset_um:g} µm, "
                        f"pixel {step:.4g} µm, z step {voxel_um[0]:.4g} µm")


def _span(present: np.ndarray, margin_px: float) -> slice:
    """Slice from the first to the last True plus a margin, clipped to the array."""
    idx = np.flatnonzero(present)
    m = int(np.ceil(max(margin_px, 0)))
    return slice(max(0, idx[0] - m), min(present.size, idx[-1] + m + 1))


def _stretch_rows(img: np.ndarray, f: float, order: int) -> np.ndarray:
    if abs(f - 1.0) < 0.01:
        return img
    return ndi.zoom(img.astype(np.float32), (f, 1), order=order, prefilter=False)


def _nice_bar(width_um: float) -> float:
    target = width_um * 0.2
    return min(NICE_BARS_UM, key=lambda v: abs(np.log(v / target)))


def render(view: View, settings: FigureSettings) -> np.ndarray:
    """RGB picture (uint8) of a view with the channel and mask settings, scale bar and legend."""
    k = max(1, int(settings.scale))
    rows, cols = view.raw.shape[1:]
    out_rows = int(round(rows * view.row_stretch))
    rgb = np.zeros((out_rows, cols, 3), np.float32)
    for i, cv in enumerate(settings.channels):
        if not cv.visible or i >= view.raw.shape[0]:
            continue
        img = _stretch_rows(view.raw[i].astype(np.float32), view.row_stretch, 1)[:out_rows]
        span = max(cv.high - cv.low, 1e-9)
        v = np.clip((img - cv.low) / span, 0, 1) ** max(cv.gamma, 1e-3)
        rgb += v[..., None] * (np.array(_hex(cv.color), np.float32) / 255.0)
    rgb = np.clip(rgb, 0, 1)
    if k > 1:
        rgb = np.repeat(np.repeat(rgb, k, axis=0), k, axis=1)
    vis_masks = [m for m in settings.masks if m.visible and m.name in view.masks]
    order = {n: i for i, n in enumerate(MASK_NAMES)}
    vis_masks.sort(key=lambda m: order.get(m.name, -1))
    for mv in vis_masks:
        m = _stretch_rows(view.masks[mv.name].astype(np.float32), view.row_stretch, 1)[:out_rows] > 0.5
        if k > 1:
            m = np.repeat(np.repeat(m, k, axis=0), k, axis=1)
        if not m.any():
            continue
        color = np.array(_hex(mv.color), np.float32) / 255.0
        if mv.style == "fill":
            a = float(settings.fill_alpha)
            rgb[m] = rgb[m] * (1 - a) + color * a
        else:
            edge = m & ~ndi.binary_erosion(m, iterations=max(1, int(settings.line_px)), border_value=0)
            rgb[edge] = color
    img = Image.fromarray((rgb * 255).astype(np.uint8))
    return np.asarray(_decorate(img, view, settings, vis_masks, k))


def _decorate(img: Image.Image, view: View, settings: FigureSettings, vis_masks, k: int) -> Image.Image:
    """Title above, scale bar below and legend right of the picture, in black margins, so nothing covers
    the cell."""
    w, h = img.size
    px_um = view.pixel_um[1] / k                       # µm per output pixel along a row
    size = max(12, min(28, int(round(max(w, h) / 32))))
    font = _font(size)
    pad = size // 2
    probe = ImageDraw.Draw(img)
    top = int(size * 1.6) if settings.title else 0
    bottom = int(size * 2.6) if settings.show_scale_bar and w > 20 else 0
    items = []
    if settings.show_legend:
        items = [(c.label, c.color) for c in settings.channels if c.visible]
        items += [(mask_title(m.name) + (" (area)" if m.style == "fill" else " (outline)"), m.color) for m in vis_masks]
    right = (int(max(probe.textlength(t, font=font) for t, _ in items)) + size * 2 + pad) if items else 0
    canvas = Image.new("RGB", (w + right, max(h + top + bottom, top + len(items) * int(size * 1.3) + pad)), (0, 0, 0))
    canvas.paste(img, (0, top))
    draw = ImageDraw.Draw(canvas)
    if settings.title:
        draw.text((pad, (top - size) // 2), settings.title, fill=(255, 255, 255), font=font)
    if bottom:
        bar = settings.scale_bar_um or _nice_bar(w * px_um)
        length = int(round(bar / px_um))
        thick = max(3, size // 4)
        x1 = w - pad
        y0 = top + h + pad
        draw.rectangle((x1 - length, y0, x1, y0 + thick), fill=(255, 255, 255))
        if settings.show_scale_text:
            text = f"{bar:g} µm"
            tw = draw.textlength(text, font=font)
            draw.text((x1 - length - tw - pad, y0 + thick / 2 - size * 0.6), text, fill=(255, 255, 255), font=font)
    y = top + pad
    for text, color in items:
        x = w + pad
        draw.rectangle((x, y + 3, x + size - 4, y + size - 1), fill=_hex(color))
        draw.text((x + size + 2, y), text, fill=(255, 255, 255), font=font)
        y += int(size * 1.3)
    return canvas


# ------------------------------------------------------------------------------------------- files
def save_png(rgb: np.ndarray, path: str | Path, dpi: int = 300) -> Path:
    path = Path(path)
    Image.fromarray(rgb).save(path, dpi=(dpi, dpi))
    return path


def save_tiff(view: View, labels: list[str], path: str | Path) -> Path:
    """The recorded values of the shown plane or section, one channel per page, calibrated in µm
    (ImageJ hyperstack). Sections are not stretched: rows are planes, the row height is the z step."""
    import tifffile

    path = Path(path)
    data = np.ascontiguousarray(view.raw)
    if data.dtype not in (np.uint8, np.uint16, np.float32):
        data = data.astype(np.float32)
    py, px = view.pixel_um
    tifffile.imwrite(path, data, imagej=True, resolution=(1.0 / px, 1.0 / py),
                     metadata={"axes": "CYX" if data.ndim == 3 else "YX", "unit": "um", "Labels": list(labels)})
    return path


def file_stem(text: str) -> str:
    keep = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)
    return "_".join(p for p in keep.split("_") if p)[:120] or "cell"
