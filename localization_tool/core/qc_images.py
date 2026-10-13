"""Control images per cell (SPEC §4): one column per channel plus a column with the compartments.

Rows: top view (maximum projection), the plane at the median height of the target signal, and a
side section (x, z) through the cell centre, coverslip at the bottom. Only for looking: the side
section is stretched in z to the true aspect ratio with linear interpolation, and outlines are
drawn from the interpolated masks, so they are smooth instead of stepped. Nothing here changes
the values used for calculations.

Outlines per column, to keep the picture readable: cell (white) and nucleus (cyan) everywhere,
actin (green) and actin-rich (yellow) in the actin column, the target mask (white dashes) in the
target column. The compartment column shows nucleus, actin, actin-rich and the rest of the
cytoplasm filled.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries

from .channels import ChannelAssignment

OUTLINE = {"cell": (255, 255, 255), "nucleus": (0, 200, 255), "actin": (60, 255, 90),
           "actin_rich": (255, 214, 10), "target": (255, 255, 255)}
FILL = {1: (61, 125, 255), 2: (43, 212, 107), 3: (120, 128, 138), 4: (255, 214, 10)}
PAD = 8
SECTION_HALF_UM = 0.5           # the side section averages +-0.5 µm around the cell centre in y
MARGIN_Z_UM = 0.6               # black space above and below the side section


def _font(size: int = 13):
    for name in ("Arial.ttf", "arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _norm(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    hi = float(np.percentile(ref, 99.7)) if ref.size else 1.0
    if hi <= 0:
        hi = float(img.max()) or 1.0
    return np.clip(img / hi, 0, 1)


def _hex(color: str) -> tuple[int, int, int]:
    c = (color or "#FFFFFF").lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def _stretch(img2d: np.ndarray, zfac: float, order: int) -> np.ndarray:
    """(z, x) image stretched in z to the true aspect ratio."""
    if zfac <= 1.01:
        return img2d
    return ndi.zoom(img2d.astype(np.float32), (zfac, 1), order=order, prefilter=False)


def _outline(rgb: np.ndarray, mask: np.ndarray | None, color, dashed: bool = False) -> None:
    if mask is None or not mask.any():
        return
    b = find_boundaries(mask, mode="inner")
    if dashed:
        yy, xx = np.indices(b.shape)
        b &= ((yy + xx) // 3) % 2 == 0
    rgb[b] = color


def _views(vol: np.ndarray, z_mid: int, y_c: int, half: int, zfac: float) -> list[np.ndarray]:
    top = vol.max(axis=0)
    plane = vol[z_mid]
    y0, y1 = max(0, y_c - half), min(vol.shape[1], y_c + half + 1)
    side = vol[:, y0:y1].astype(np.float32).mean(axis=1)[::-1]          # coverslip at the bottom
    return [top, plane, _stretch(side, zfac, 1)]


def _mask_views(m: np.ndarray, z_mid: int, y_c: int, zfac: float) -> list[np.ndarray]:
    top = m.any(axis=0)
    plane = m[z_mid]
    side = _stretch(m[:, y_c].astype(np.float32)[::-1], zfac, 1) > 0.5
    return [top, plane, side]


def _pad_side(img: np.ndarray, margin: int) -> np.ndarray:
    pad = [(margin, margin)] + [(0, 0)] * (img.ndim - 1)
    return np.pad(img, pad)


def control_image(data: np.ndarray, assignments: list[ChannelAssignment], masks: dict[str, np.ndarray],
                  voxel_um: tuple[float, float, float], title: str, lines: list[str], path,
                  coverslip_first: bool = True, overlays: dict | None = None):
    """Write a PNG to a path or an open binary file. overlays (crop coordinates, optional): "polarity"
    arrow on the top views. Each channel column names the intensity shown as full brightness, so a
    control without staining does not look stained."""
    overlays = overlays or {}
    cell = masks["cell"]
    oy = ox = 0
    if cell.any():
        # show the cell with a margin of 1 µm instead of the whole field
        yy, xx = np.nonzero(cell.any(axis=0))
        my, mx = int(round(1.0 / voxel_um[1])), int(round(1.0 / voxel_um[2]))
        y0, y1 = max(0, yy.min() - my), min(cell.shape[1], yy.max() + my + 1)
        x0, x1 = max(0, xx.min() - mx), min(cell.shape[2], xx.max() + mx + 1)
        data = data[:, :, y0:y1, x0:x1]
        oy, ox = y0, x0
        masks = {k: v[:, y0:y1, x0:x1] for k, v in masks.items() if v is not None and v.ndim == 3}
        cell = masks["cell"]
    flip = (lambda a: a) if coverslip_first else (lambda a: a[::-1])
    data = data if coverslip_first else data[:, ::-1]
    masks = {k: flip(v) for k, v in masks.items()}
    z_mid, mid_text = _middle_plane(data, assignments, masks)
    y_c = int(round(np.nonzero(cell)[1].mean())) if cell.any() else data.shape[2] // 2
    half = max(0, int(round(SECTION_HALF_UM / voxel_um[1])))
    zfac = voxel_um[0] / voxel_um[2]
    margin = int(round(MARGIN_Z_UM / voxel_um[2]))
    cell_any = cell.any(axis=0)

    mviews = {k: _mask_views(masks[k], z_mid, y_c, zfac) for k in ("cell", "nucleus", "actin", "actin_rich")
              if k in masks}
    columns = []
    for i, a in enumerate(assignments):
        if a.role == "Ignore":
            continue
        color = np.array(_hex(a.color) if a.color else (255, 255, 255), np.float32)
        vol = data[i]
        ref_vals = vol[cell] if cell.any() else vol
        hi = float(np.percentile(ref_vals, 99.7)) if ref_vals.size else 0.0
        views = _views(vol, z_mid, y_c, half, zfac)
        tmask = masks.get(f"target_{a.label}")
        tviews = _mask_views(tmask, z_mid, y_c, zfac) if tmask is not None else None
        rows = []
        for k, img in enumerate(views):
            rgb = _norm(img, ref_vals)[..., None] * color[None, None, :]
            for name in ("cell", "nucleus") + (("actin", "actin_rich") if a.role == "Actin" else ()):
                if name in mviews:
                    _outline(rgb, mviews[name][k], OUTLINE[name])
            if tviews is not None:
                _outline(rgb, tviews[k], OUTLINE["target"], dashed=True)
            if k == 0 and a.role in ("Actin", "Target"):
                rgb = _arrow(rgb, overlays.get("polarity"), oy, ox)
            if k == 2:
                rgb = _pad_side(rgb, margin)
                rgb[-margin:-margin + 1] = (90, 90, 90)          # coverslip
            rows.append(rgb)
        columns.append((f"{a.label} (full = {hi:.0f})", rows))

    if "labels" in masks:
        lab = masks["labels"].astype(np.uint8)
        if "actin_rich" in masks:
            lab = np.where((lab == 2) & masks["actin_rich"], 4, lab).astype(np.uint8)
        side = ndi.zoom(lab[:, y_c][::-1], (zfac, 1), order=0) if zfac > 1.01 else lab[:, y_c][::-1]
        lviews = [lab[int(np.argmax(cell.reshape(cell.shape[0], -1).sum(axis=1)))], lab[z_mid], side]
        rows = []
        for k, lv in enumerate(lviews):
            rgb = np.zeros(lv.shape + (3,), np.float32)
            for code, col in FILL.items():
                rgb[lv == code] = col
            if k == 2:
                rgb = _pad_side(rgb, margin)
                rgb[-margin:-margin + 1] = (90, 90, 90)
            rows.append(rgb)
        columns.append(("Compartments", rows))

    w = max(r.shape[1] for _, rows in columns for r in rows)
    heights = [max(rows[k].shape[0] for _, rows in columns) for k in range(3)]
    font = _font(13)
    header = 24 + 16 * len(lines)
    label_h = 36                         # row title and column names
    row_titles = ["top view (max projection)", f"plane {z_mid + 1} of {data.shape[1]} ({mid_text})",
                  "side section through the cell centre"]
    W = PAD + len(columns) * (w + PAD)
    bar = int(round(2.0 / voxel_um[2]))
    legend_parts = ["outlines: white cell, cyan nucleus, green actin, yellow actin-rich, white dashes target mask",
                    "compartments (top row: widest plane): blue nucleus, green actin, yellow actin-rich, "
                    "grey rest of cytoplasm"]
    if overlays.get("polarity"):
        legend_parts.append("orange arrow: front")
    legend = _wrap(legend_parts, font, max(W - 2 * PAD, 200))
    W = max(W, 2 * PAD + bar + 60)
    H = header + sum(label_h + h + PAD for h in heights) + 26 + 16 * len(legend)
    canvas = Image.new("RGB", (W, H), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((PAD, 4), title, fill=(230, 233, 237), font=_font(15))
    for k, line in enumerate(lines):
        draw.text((PAD, 24 + 16 * k), line, fill=(154, 164, 175), font=font)
    y = header
    for k in range(3):
        for c, (lab_text, rows) in enumerate(columns):
            x0 = PAD + c * (w + PAD)
            if c == 0:
                draw.text((x0, y), row_titles[k], fill=(230, 233, 237), font=_font(14))
            draw.text((x0, y + 17), lab_text, fill=(154, 164, 175), font=font)
            canvas.paste(Image.fromarray(np.clip(rows[k], 0, 255).astype(np.uint8)), (x0, y + label_h))
        y += label_h + heights[k] + PAD
    yb = y + 8
    draw.rectangle([PAD, yb, PAD + bar, yb + 3], fill=(230, 233, 237))
    draw.text((PAD + bar + 6, yb - 6), "2 µm", fill=(230, 233, 237), font=font)
    for k, line in enumerate(legend):
        draw.text((PAD, yb + 12 + 16 * k), line, fill=(154, 164, 175), font=font)
    if hasattr(path, "write"):
        canvas.save(path, format="PNG")
        return path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


def _middle_plane(data: np.ndarray, assignments, masks: dict) -> tuple[int, str]:
    """The plane of the middle row: the median height of the first target's signal in the cell, so
    the row shows where the target is (phase 5 review: the widest nucleus plane of a stack cut off at
    the top had no target at all). Without target signal: the median plane of the nucleus."""
    cell = masks["cell"]
    targets = [i for i, a in enumerate(assignments) if a.role == "Target"]
    if targets and cell.any():
        w = np.where(cell, np.asarray(data[targets[0]], np.float64), 0.0)
        w = np.maximum(w - np.median(w[cell]), 0).sum(axis=(1, 2))
        if w.sum() > 0:
            c = np.cumsum(w) / w.sum()
            return int(np.searchsorted(c, 0.5)), f"middle of the {assignments[targets[0]].label} signal"
    nuc = masks.get("nucleus")
    ref = nuc if nuc is not None and nuc.any() else cell
    if ref.any():
        return int(np.median(np.nonzero(ref)[0])), "middle of the nucleus" if ref is nuc else "middle of the cell"
    return data.shape[1] // 2, "middle of the stack"


def _wrap(parts: list[str], font, width: int) -> list[str]:
    """Legend lines that fit the picture width (narrow cells cut it off otherwise)."""
    words = "   ".join(parts).split(" ")
    lines, cur = [], ""
    for wd in words:
        trial = (cur + " " + wd) if cur else wd
        if cur and font.getlength(trial) > width:
            lines.append(cur.strip())
            cur = wd
        else:
            cur = trial
    if cur.strip():
        lines.append(cur.strip())
    return lines


def _arrow(rgb: np.ndarray, pol: dict | None, oy: int, ox: int) -> np.ndarray:
    if not pol:
        return rgb
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    (y0, x0), (y1, x1) = pol["start"], pol["end"]
    a, b = (x0 - ox, y0 - oy), (x1 - ox, y1 - oy)
    clear = pol.get("clear", True)            # D84: a thin, dimmer arrow when the polarity is not clear
    col = (255, 140, 0) if clear else (150, 90, 20)
    width = 3 if clear else 1
    head = 12 if clear else 7
    d.line([a, b], fill=col, width=width)
    v = np.array([b[0] - a[0], b[1] - a[1]], float)
    n = float(np.hypot(*v))
    if n > 0:
        v /= n
        w = np.array([-v[1], v[0]])
        tip = np.array(b, float)
        d.polygon([tuple(tip), tuple(tip - head * v + head * 0.6 * w), tuple(tip - head * v - head * 0.6 * w)], fill=col)
    return np.asarray(img, np.float32)
