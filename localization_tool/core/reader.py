"""Reading of Zeiss CZI, Nikon ND2 and TIFF files into an ImageStack."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .image import ChannelInfo, ImageStack

SUPPORTED_SUFFIXES = (".czi", ".nd2", ".tif", ".tiff")
DATA_AXES = "CZYX"


class ReaderError(Exception):
    """Problem with an input file. The message is shown to the user as it is."""


def read_image(path: str | Path) -> ImageStack:
    path = Path(path)
    if not path.exists():
        raise ReaderError(f"File not found: {path}")
    suffix = path.suffix.lower()
    readers = {".czi": _read_czi, ".nd2": _read_nd2, ".tif": _read_tiff, ".tiff": _read_tiff}
    if suffix in readers:
        try:
            return readers[suffix](path)
        except ReaderError:
            raise
        except MemoryError as exc:
            raise ReaderError(f"{path.name} is too large for the memory of this computer.") from exc
        except Exception as exc:  # damaged files raise all kinds of errors in the libraries
            raise ReaderError(f"{path.name} could not be read ({exc.__class__.__name__}: {exc}). "
                              "The file may be damaged or incomplete.") from exc
    raise ReaderError(
        f"{path.name}: file type '{path.suffix}' is not supported. "
        "Supported are Zeiss .czi, Nikon .nd2 and .tif/.tiff (ImageJ or OME)."
    )


# --------------------------------------------------------------------------- helpers

def _to_czyx(data: np.ndarray, axes: str, filename: str) -> np.ndarray:
    """Bring data with the given axis letters into (C, Z, Y, X)."""
    axes = axes.upper()
    for ax, n in zip(axes, data.shape):
        if ax in DATA_AXES:
            continue
        if n == 1:
            continue
        if ax == "H":
            raise ReaderError(
                f"{filename} contains unprocessed Airyscan raw data. "
                "Please run Airyscan Processing in ZEN first and load the processed file."
            )
        if ax == "T":
            raise ReaderError(f"{filename} is a time series ({n} time points). Time series are not supported.")
        if ax in ("S", "A") or ax == "0":
            raise ReaderError(f"{filename} contains RGB or multi-sample pixels, which are not supported.")
        if ax in ("M", "B", "R", "I", "V", "Q"):
            raise ReaderError(f"{filename} contains several tiles or positions (axis {ax} = {n}). Please export single positions.")
        raise ReaderError(f"{filename}: unexpected image dimension '{ax}' with {n} entries.")
    # drop singleton foreign axes
    index = tuple(slice(None) if ax in DATA_AXES else 0 for ax in axes)
    data = data[index]
    kept = "".join(ax for ax in axes if ax in DATA_AXES)
    for ax in DATA_AXES:
        if ax not in kept:
            data = data[np.newaxis, ...]
            kept = ax + kept
    order = [kept.index(ax) for ax in DATA_AXES]
    return np.ascontiguousarray(np.transpose(data, order))


def _hex_rgb(color: str | None) -> str | None:
    """'#AARRGGBB' or '#RRGGBB' to '#RRGGBB'. White and black count as 'no colour'."""
    if not color or not isinstance(color, str):
        return None
    c = color.strip().lstrip("#")
    if len(c) == 8:
        c = c[2:]
    if len(c) != 6:
        return None
    c = c.upper()
    if c in ("FFFFFF", "000000"):
        return None
    return "#" + c


def _num(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


# --------------------------------------------------------------------------- CZI

def _read_czi(path: Path) -> ImageStack:
    import czifile

    try:
        czi = czifile.CziFile(path)
    except Exception as exc:  # broken or incomplete file
        raise ReaderError(f"{path.name} could not be opened as CZI file ({exc}). The file may be incomplete.") from exc
    with czi:
        scenes = czi.scenes
        if not scenes:
            raise ReaderError(f"{path.name} contains no image data.")
        if len(scenes) > 1:
            raise ReaderError(f"{path.name} contains {len(scenes)} scenes. Please save single scenes in ZEN.")
        img = scenes[0]
        axes = "".join(img.dims)
        try:
            raw = img.asarray()
        except Exception as exc:
            raise ReaderError(f"{path.name}: image data could not be read ({exc}). The file may be incomplete.") from exc
        data = _to_czyx(np.asarray(raw), axes, path.name)

        scales = getattr(img, "coord_scales", {}) or {}
        def um(ax):
            v = scales.get(ax)
            return float(v) * 1e6 if v else None
        dx, dy, dz = um("X"), um("Y"), um("Z")
        calibrated = bool(dx and dy)
        dx = dx or 1.0
        dy = dy or dx
        if dz is None:
            dz = dx if data.shape[1] == 1 else 1.0
            calibrated = calibrated and data.shape[1] == 1

        channels = []
        meta_channels = list((getattr(img, "channels", None) or {}).items())
        for i in range(data.shape[0]):
            if i < len(meta_channels):
                name, m = meta_channels[i]
                m = m or {}
                channels.append(ChannelInfo(
                    index=i,
                    name=str(name),
                    fluor=m.get("Fluor") or m.get("DyeName"),
                    excitation_nm=_num(m.get("ExcitationWavelength")),
                    emission_nm=_num(m.get("EmissionWavelength")),
                    color=_hex_rgb(m.get("Color")),
                ))
            else:
                channels.append(ChannelInfo(index=i, name=f"C{i + 1}"))

        info = {"format": "CZI"}
        objective = getattr(img, "objective", None) or {}
        if isinstance(objective, dict) and objective.get("Name"):
            info["objective"] = objective["Name"]
        when = getattr(img, "datetime", None)
        if when:
            info["acquired"] = str(when)
        try:
            xml = czi.metadata()
        except Exception:
            xml = ""
        info["airyscan_processed"] = "SuperResolutionParameter" in (xml or "")
        info.update(z_orientation(xml or ""))
        acq = acquisition_settings(xml or "")
        if acq:
            info["acquisition"] = acq

    return ImageStack(data=data, voxel_um=(dz, dy, dx), channels=channels, source=path,
                      calibrated=calibrated, info=info)


def z_orientation(xml: str) -> dict:
    """Which plane lies at the coverslip, from the Zeiss metadata (D20).

    Plane 0 is nearest to the objective when the z axis points from the objective into the
    specimen and the z positions increase with the plane index. The coverslip faces the objective
    for oil and water immersion, so plane 0 is the coverslip side. Returns
    {"coverslip_first": bool, "z_orientation_source": text} or {} when the metadata are missing.
    """
    import re

    m = re.search(r"<Dimensions>.*?<Z>(.*?)</Z>", xml, re.S)
    if not m:
        return {}
    block = m.group(1)
    direction = re.search(r"<ZAxisDirection>\s*(\w+)\s*</ZAxisDirection>", block)
    increment = re.search(r"<Increment>\s*([-+0-9.eE]+)\s*</Increment>", block)
    if not direction:
        return {}
    toward = direction.group(1) == "FromObjectiveToSpecimen"
    if direction.group(1) not in ("FromObjectiveToSpecimen", "FromSpecimenToObjective"):
        return {}
    step = float(increment.group(1)) if increment else 1.0
    first = toward == (step > 0)
    stand = re.search(r"<Type>\s*(Inverted|Upright)\s*</Type>", xml)
    source = f"CZI metadata: {direction.group(1)}, step {step:+g} um"
    if stand:
        source += f", {stand.group(1).lower()} stand"
    return {"coverslip_first": first, "z_orientation_source": source}


def acquisition_settings(xml: str) -> dict:
    """Settings that should be equal within an experiment (D63), from the Zeiss metadata.

    Returns {"objective", "immersion", "refractive_index", "channels": [per channel in file order:
    {"fluor", "detector_voltage", "detector_offset", "laser_transmission_pct", "airyscan_strength",
    "airyscan_estimation"}]}. Missing values are left out. Never raises: metadata vary between ZEN versions.
    """
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(xml)
    except Exception:
        return {}

    def num(el, tag):
        x = el.find(tag) if el is not None else None
        try:
            return float(x.text) if x is not None and x.text is not None else None
        except ValueError:
            return None

    out: dict = {}
    obj = root.find("./Metadata/Information/Image/ObjectiveSettings")
    if obj is not None:
        if obj.findtext("Medium"):
            out["immersion"] = obj.findtext("Medium")
        ri = num(obj, "RefractiveIndex")
        if ri:
            out["refractive_index"] = ri
    objectives = root.findall("./Metadata/Information/Instrument/Objectives/Objective")
    ref = obj.find("ObjectiveRef") if obj is not None else None
    used = next((o for o in objectives if ref is not None and o.get("Id") == ref.get("Id")), None)
    objective = used if used is not None else (objectives[0] if len(objectives) == 1 else None)
    if objective is not None and objective.get("Name"):
        out["objective"] = objective.get("Name")    # the one referenced by the image, not the first listed
    strengths = {}                       # by channel Id ("Channel:0", "Channel:2", ...), not by position
    for ch in root.findall("./Metadata/Information/Processing/Airyscan/Channels/Channel"):
        v = num(ch, "SuperResolutionParameter")
        if ch.get("Id") and v is not None:
            strengths[ch.get("Id")] = (v, ch.findtext("ParameterEstimationType") or "")
    channels = []
    for ch in root.findall("./Metadata/Information/Image/Dimensions/Channels/Channel"):
        i = ch.get("Id")
        d: dict = {}
        if ch.findtext("Fluor"):
            d["fluor"] = ch.findtext("Fluor")
        det = ch.find("DetectorSettings")
        for tag, key in (("Voltage", "detector_voltage"), ("Offset", "detector_offset")):
            v = num(det, tag)
            if v is not None:
                d[key] = round(v, 1)
        att = num(ch.find("LightSourcesSettings/LightSourceSettings"), "Attenuation")
        if att is not None:
            d["laser_transmission_pct"] = round(100.0 * (1.0 - att), 2)
        if i in strengths:
            d["airyscan_strength"] = round(strengths[i][0], 2)
            d["airyscan_estimation"] = strengths[i][1]
        channels.append(d)
    if channels:
        out["channels"] = channels
    return out


# --------------------------------------------------------------------------- ND2 (D90)

ND2_AXES = {"T": "T", "P": "M", "Z": "Z", "C": "C", "Y": "Y", "X": "X", "S": "S"}
INVERTED_STANDS = ("ti2", "ti-e", "ti-u", "ti-s", "eclipse ti", "te2000", "te300", "diaphot")
UPRIGHT_STANDS = ("ni-e", "ni-u", "eclipse ni", "fn1", "fn-s", "80i", "90i", "e600", "e800", "e1000")


def _nd2_color(value) -> str | None:
    """nd2 >= 0.11 gives a Color (r, g, b); older versions and the file store 0x00BBGGRR."""
    if all(hasattr(value, k) for k in ("r", "g", "b")):
        return _hex_rgb(f"#{int(value.r) & 0xFF:02X}{int(value.g) & 0xFF:02X}{int(value.b) & 0xFF:02X}")
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    return _hex_rgb(f"#{v & 0xFF:02X}{(v >> 8) & 0xFF:02X}{(v >> 16) & 0xFF:02X}")


def nd2_z_orientation(experiment, text: str) -> dict:
    """Which plane lies at the coverslip, from the ND2 metadata (D90, like D20 for CZI).

    The z loop says whether planes were taken from bottom to top. On an inverted stand (Ti2, Ti,
    TE2000) the bottom is the coverslip side, on an upright stand (Ni, FN1) the top faces the
    objective and the coverslip. Without a recognised stand nothing is returned and the setting
    "Coverslip side" decides (default: first plane)."""
    loop = next((e for e in (experiment or []) if getattr(e, "type", "") == "ZStackLoop"), None)
    params = getattr(loop, "parameters", None)
    bottom_to_top = getattr(params, "bottomToTop", None)
    if bottom_to_top is None:
        return {}
    low = (text or "").lower()
    if any(s in low for s in INVERTED_STANDS):
        stand, first = "inverted", bool(bottom_to_top)
    elif any(s in low for s in UPRIGHT_STANDS):
        stand, first = "upright", not bool(bottom_to_top)
    else:
        return {}
    order = "bottom to top" if bottom_to_top else "top to bottom"
    return {"coverslip_first": first, "z_orientation_source": f"ND2 metadata: z stack {order}, {stand} stand"}


def _read_nd2(path: Path) -> ImageStack:
    import nd2

    try:
        f = nd2.ND2File(path)
    except Exception as exc:
        raise ReaderError(f"{path.name} could not be opened as ND2 file ({exc}). The file may be incomplete.") from exc
    with f:
        if f.is_rgb:
            raise ReaderError(f"{path.name} is an RGB colour image, which is not supported.")
        sizes = dict(f.sizes)
        axes = "".join(ND2_AXES.get(k, k) for k in sizes)
        try:
            raw = np.asarray(f.asarray())
        except Exception as exc:
            raise ReaderError(f"{path.name}: image data could not be read ({exc}). The file may be incomplete.") from exc
        data = _to_czyx(raw, axes, path.name)
        try:
            vs = f.voxel_size()
            dx, dy, dz = _num(vs.x), _num(vs.y), _num(vs.z)
        except Exception:
            dx = dy = dz = None
        calibrated = bool(dx and dy and (dz or data.shape[1] == 1))
        if "Z" not in sizes:
            dz = None
        dx = dx or 1.0
        dy = dy or dx
        dz = dz or (dx if data.shape[1] == 1 else 1.0)
        meta_channels = list(getattr(getattr(f, "metadata", None), "channels", None) or [])
        channels = []
        for i in range(data.shape[0]):
            ch = getattr(meta_channels[i], "channel", None) if i < len(meta_channels) else None
            if ch is None:
                channels.append(ChannelInfo(index=i, name=f"C{i + 1}"))
                continue
            name = str(getattr(ch, "name", "") or f"C{i + 1}")
            channels.append(ChannelInfo(index=i, name=name, fluor=None,
                                        excitation_nm=_num(getattr(ch, "excitationLambdaNm", None)),
                                        emission_nm=_num(getattr(ch, "emissionLambdaNm", None)),
                                        color=_nd2_color(getattr(ch, "color", None) or getattr(ch, "colorRGB", None))))
        info: dict = {"format": "ND2"}
        scope = getattr(meta_channels[0], "microscope", None) if meta_channels else None
        acq: dict = {}
        if scope is not None and getattr(scope, "objectiveName", None):
            info["objective"] = str(scope.objectiveName)
            acq["objective"] = str(scope.objectiveName)
        if scope is not None and _num(getattr(scope, "immersionRefractiveIndex", None)):
            acq["refractive_index"] = float(scope.immersionRefractiveIndex)
        if acq:
            acq["channels"] = [{} for _ in channels]
            info["acquisition"] = acq
        try:
            text = " ".join(str(v) for v in (f.text_info or {}).values())
        except Exception:
            text = ""
        try:
            experiment = f.experiment
        except Exception:
            experiment = []
        info.update(nd2_z_orientation(experiment, text))
        date = (f.text_info or {}).get("date") if isinstance(getattr(f, "text_info", None), dict) else None
        if date:
            info["acquired"] = str(date)
    return ImageStack(data=data, voxel_um=(dz, dy, dx), channels=channels, source=path,
                      calibrated=calibrated, info=info)


ACQUISITION_KEYS = ("airyscan_strength", "laser_transmission_pct", "detector_voltage", "detector_offset")
# --------------------------------------------------------------------------- TIFF

def _read_tiff(path: Path) -> ImageStack:
    import tifffile

    try:
        tif = tifffile.TiffFile(path)
    except Exception as exc:
        raise ReaderError(f"{path.name} could not be opened as TIFF ({exc}).") from exc
    with tif:
        series = tif.series[0]
        axes = series.axes
        if not (tif.is_imagej or tif.is_ome):
            # plain multipage TIFF: a single unknown leading axis is taken as Z
            unknown = [ax for ax in axes if ax in "QI"]
            if len(unknown) == 1 and "Z" not in axes:
                axes = axes.replace(unknown[0], "Z")
        data = _to_czyx(series.asarray(), axes, path.name)
        expected = (tif.imagej_metadata or {}).get("images") if tif.is_imagej else None
        if expected and int(expected) > data.shape[0] * data.shape[1]:
            raise ReaderError(f"{path.name} holds {data.shape[0] * data.shape[1]} of {int(expected)} planes. "
                              "The file is incomplete.")
        dz = dy = dx = None
        names: list[str] = []

        if tif.is_ome and tif.ome_metadata:
            ome = tifffile.xml2dict(tif.ome_metadata)
            image = ome.get("OME", {}).get("Image", {})
            if isinstance(image, list):
                image = image[0]
            pixels = image.get("Pixels", {})
            dx = _ome_um(pixels.get("PhysicalSizeX"), pixels.get("PhysicalSizeXUnit"))
            dy = _ome_um(pixels.get("PhysicalSizeY"), pixels.get("PhysicalSizeYUnit"))
            dz = _ome_um(pixels.get("PhysicalSizeZ"), pixels.get("PhysicalSizeZUnit"))
            chans = pixels.get("Channel", [])
            if isinstance(chans, dict):
                chans = [chans]
            names = [str(c.get("Name") or f"C{i + 1}") for i, c in enumerate(chans)]
        elif tif.is_imagej:
            meta = tif.imagej_metadata or {}
            page = tif.pages[0]
            unit = str(meta.get("unit", "")).lower().replace("\\u00b5", "µ").replace("\u03bc", "µ")
            factor = {"micron": 1.0, "µm": 1.0, "um": 1.0, "nm": 1e-3, "mm": 1e3}.get(unit)
            xres = page.tags.get("XResolution")
            yres = page.tags.get("YResolution")
            if factor and xres:
                num, den = xres.value
                dx = den / num * factor if num else None
            if factor and yres:
                num, den = yres.value
                dy = den / num * factor if num else None
            if factor and meta.get("spacing"):
                dz = float(meta["spacing"]) * factor
            labels = meta.get("Labels")
            if isinstance(labels, (list, tuple)) and len(labels) >= data.shape[0]:
                names = [str(x) for x in labels[: data.shape[0]]]
                if any(n.startswith(("c:", "z:")) or len(n) > 40 for n in names):
                    names = []   # Fiji slice labels ("c:1/3 z:1/32 - file.czi"), not channel names

    calibrated = bool(dx and dy and (dz or data.shape[1] == 1))
    dx = dx or 1.0
    dy = dy or dx
    dz = dz or (dx if data.shape[1] == 1 else 1.0)
    if len(names) != data.shape[0]:
        names = [f"C{i + 1}" for i in range(data.shape[0])]
    channels = [ChannelInfo(index=i, name=n) for i, n in enumerate(names)]
    info = {"format": "OME-TIFF" if names and tif.is_ome else "TIFF"}
    return ImageStack(data=data, voxel_um=(dz, dy, dx), channels=channels, source=path,
                      calibrated=calibrated, info=info)


def _ome_um(value, unit) -> float | None:
    v = _num(value)
    if v is None:
        return None
    unit = (unit or "µm").replace("\u03bc", "µ")
    return v * {"µm": 1.0, "um": 1.0, "nm": 1e-3, "mm": 1e3, "m": 1e6}.get(unit, 1.0)


def write_imagej_tiff(stack: ImageStack, path: str | Path) -> None:
    """Write a stack as calibrated ImageJ hyperstack (used for examples and tests)."""
    import tifffile

    dz, dy, dx = stack.voxel_um
    data = np.transpose(stack.data, (1, 0, 2, 3))  # ZCYX
    tifffile.imwrite(
        path,
        data,
        imagej=True,
        resolution=(1.0 / dx, 1.0 / dy),
        metadata={"axes": "ZCYX", "spacing": dz, "unit": "micron",
                  "Labels": [c.short for _ in range(data.shape[0]) for c in stack.channels]},
    )
