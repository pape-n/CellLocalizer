"""Synthetic example cell. Used for the self-test, the built-in example and tests.

Never derived from real experiments. Geometry (all in µm, centred in the stack):
- cell: flattened ellipsoid resting on plane 0 (coverslip side)
- nucleus: ellipsoid inside the cell, slightly off centre
- actin: cortical shell plus a dense lamellipodium on one side
- target: puncta, a share of them in the nucleus, at the actin front and in the cytoplasm
"""

from __future__ import annotations

import numpy as np

from .image import ChannelInfo, ImageStack


def make_synthetic_cell(shape_zyx=(30, 192, 192), voxel_um=(0.13, 0.05, 0.05), seed: int = 1,
                        with_vimentin: bool = False) -> ImageStack:
    rng = np.random.default_rng(seed)
    nz, ny, nx = shape_zyx
    dz, dy, dx = voxel_um
    z = (np.arange(nz) * dz)[:, None, None]
    y = (np.arange(ny) * dy - ny * dy / 2)[None, :, None]
    x = (np.arange(nx) * dx - nx * dx / 2)[None, None, :]

    cell_h = min(3.0, nz * dz * 0.85)
    cell = ((x / 3.6) ** 2 + (y / 3.0) ** 2 + ((z - 0.0) / cell_h) ** 2) <= 1.0
    nucleus = (((x + 0.6) / 2.0) ** 2 + (y / 1.7) ** 2 + ((z - 1.2) / 1.0) ** 2) <= 1.0
    nucleus &= cell

    # distance-like measure from the surface for the cortex shell
    r = np.sqrt((x / 3.6) ** 2 + (y / 3.0) ** 2 + (z / cell_h) ** 2)
    cortex = cell & (r > 0.86)
    front = cell & (x > 2.4) & (z < 1.2)

    def noise(img, scale):
        return rng.poisson(np.clip(img, 0, None) * scale) / scale

    dapi = np.where(nucleus, 800.0, 0.0)
    actin = np.where(cortex, 500.0, 0.0) + np.where(front, 1200.0, 0.0) + np.where(cell, 60.0, 0.0)

    target = np.zeros(shape_zyx)
    n_puncta = 160
    regions = [(nucleus, 0.25), (front | cortex, 0.35), (cell & ~nucleus & ~cortex & ~front, 0.40)]
    for mask, share in regions:
        idx = np.argwhere(mask)
        if len(idx) == 0:
            continue
        pick = idx[rng.choice(len(idx), size=int(n_puncta * share), replace=True)]
        for zz, yy, xx in pick:
            target[zz, max(0, yy - 1):yy + 2, max(0, xx - 1):xx + 2] += 1500.0
    target += np.where(cell, 40.0, 0.0)

    channels_data = [noise(target, 0.05), noise(actin, 0.05), noise(dapi, 0.05)]
    channels = [
        ChannelInfo(0, "AF647-T1", "Alexa Fluor 647", 653, 668, None),
        ChannelInfo(1, "AF488-T2", "Alexa Fluor 488", 493, 517, "#00FF33"),
        ChannelInfo(2, "DAPI-T3", "DAPI", 353, 465, "#00A0FF"),
    ]
    if with_vimentin:
        vim = np.where(cell & ~nucleus, 200.0, 0.0) * (1 + 0.5 * np.sin(x * 4) * np.cos(y * 3))
        channels_data.insert(1, noise(vim, 0.05))
        channels.insert(1, ChannelInfo(1, "AF555-T2", "Alexa Fluor 555", 553, 568, "#FF7E00"))
        for i, ch in enumerate(channels):
            ch.index = i

    data = np.stack([np.clip(c, 0, 65535).astype(np.uint16) for c in channels_data])
    return ImageStack(data=data, voxel_um=tuple(voxel_um), channels=channels, source=None,
                      info={"format": "synthetic", "airyscan_processed": False})



def make_practice_cell() -> ImageStack:
    """The cell behind "Practice with a synthetic cell": the synthetic cell with two targets and four
    empty planes below it, as in a stack that starts in the glass under the cell (INSTALL checklist), so
    the practice cell has a low bottom_plane_ratio (D67)."""
    st = make_synthetic_cell(with_vimentin=True)
    pad = np.zeros((st.data.shape[0], 4) + st.data.shape[2:], st.data.dtype)
    return ImageStack(data=np.concatenate([pad, st.data], axis=1), voxel_um=st.voxel_um, channels=st.channels,
                      source=None, info=dict(st.info))

# --------------------------------------------------------------------------- golden cells for tests

GOLDEN_DENSITY = {"nucleus": 2.0, "actin": 5.0, "cytoplasm": 1.0}


def make_golden_cell(voxel_um=(0.13, 0.05, 0.05), cell_axes=(5.0, 4.0, 3.0), nucleus_radius=2.0,
                     nucleus_shift_x=-1.0, front_x=3.0, ring_nucleus=False, scale=100.0, margin_um=1.0):
    """Synthetic cell with analytically known compartments, without blur and noise.

    Geometry in µm: the cell is an ellipsoid with semi-axes (a, b, c) = cell_axes along (x, y, z),
    its lowest point 0.2 µm above the centre of plane 0. The nucleus is a sphere (or a torus with
    ring_nucleus=True) shifted along x. The actin-rich region is the cap of the cell with x > front_x.
    The target has a constant density per compartment (GOLDEN_DENSITY, times `scale`).

    Returns (stack, truth). truth holds the voxel masks and the analytic expectations:
    volumes in µm³, signal shares and enrichment per compartment.
    """
    a, b, c = cell_axes
    dz, dy, dx = voxel_um
    margin = margin_um
    nz = int(np.ceil((2 * c + 0.2 + margin) / dz))
    ny = int(np.ceil((2 * b + 2 * margin) / dy))
    nx = int(np.ceil((2 * a + 2 * margin) / dx))
    # voxel centres sit half a voxel off round numbers in x and y, so no boundary of the
    # analytic shapes falls exactly on a row of voxel centres
    z = (np.arange(nz) * dz)[:, None, None]
    y = ((np.arange(ny) + 0.5) * dy - ny * dy / 2)[None, :, None]
    x = ((np.arange(nx) + 0.5) * dx - nx * dx / 2)[None, None, :]
    zc = 0.2 + c
    cell = ((x / a) ** 2 + (y / b) ** 2 + ((z - zc) / c) ** 2) <= 1.0
    xn = x - nucleus_shift_x
    if ring_nucleus:
        # torus around the z axis: tube radius r, ring radius R; the hole has radius R - r
        R, r = nucleus_radius, nucleus_radius * 0.45
        nucleus = (np.sqrt(xn ** 2 + y ** 2) - R) ** 2 + (z - zc) ** 2 <= r ** 2
        v_nucleus = 2 * np.pi ** 2 * R * r ** 2
    else:
        nucleus = xn ** 2 + y ** 2 + (z - zc) ** 2 <= nucleus_radius ** 2
        v_nucleus = 4 / 3 * np.pi * nucleus_radius ** 3
    nucleus = nucleus & cell
    front = cell & (x > front_x)
    labels = np.zeros(cell.shape, np.uint8)
    labels[cell] = 3
    labels[front] = 2
    labels[nucleus] = 1

    v_cell = 4 / 3 * np.pi * a * b * c
    v_front = (b * c / a ** 2) * np.pi * (a - front_x) ** 2 * (2 * a + front_x) / 3
    v_cyto = v_cell - v_nucleus - v_front
    vols = {"nucleus": v_nucleus, "actin": v_front, "cytoplasm": v_cyto}      # region types of GOLDEN_DENSITY
    sig = {k: GOLDEN_DENSITY[k] * v for k, v in vols.items()}
    tot = sum(sig.values())
    # measured regions (D39): cytoplasm = actin + cytoplasm without actin
    m_vol = {"nucleus": v_nucleus, "actin": v_front, "cytoplasm_without_actin": v_cyto,
             "cytoplasm": v_front + v_cyto}
    m_sig = {"nucleus": sig["nucleus"], "actin": sig["actin"], "cytoplasm_without_actin": sig["cytoplasm"],
             "cytoplasm": sig["actin"] + sig["cytoplasm"]}
    expected = {}
    for k in m_vol:
        f = 100 * m_sig[k] / tot
        phi = 100 * m_vol[k] / v_cell
        expected[f"signal_{k}_pct"] = f
        expected[f"volume_{k}_pct"] = phi
        expected[f"enrichment_{k}"] = f / phi

    target = np.zeros(cell.shape)
    target[labels == 1] = GOLDEN_DENSITY["nucleus"] * scale
    target[labels == 2] = GOLDEN_DENSITY["actin"] * scale
    target[labels == 3] = GOLDEN_DENSITY["cytoplasm"] * scale
    actin = np.where(front, 1000.0, 0.0) + np.where(cell, 100.0, 0.0)
    dapi = np.where(nucleus, 800.0, 0.0)
    data = np.stack([target, actin, dapi]).astype(np.uint16)
    channels = [
        ChannelInfo(0, "AF647-T1", "Alexa Fluor 647", 653, 668, None),
        ChannelInfo(1, "AF488-T2", "Alexa Fluor 488", 493, 517, "#00FF33"),
        ChannelInfo(2, "DAPI-T3", "DAPI", 353, 465, "#00A0FF"),
    ]
    stack = ImageStack(data=data, voxel_um=tuple(voxel_um), channels=channels, source=None,
                       info={"format": "synthetic", "airyscan_processed": False, "coverslip_first": True,
                             "z_orientation_source": "synthetic"})
    truth = {
        "cell": cell, "nucleus": nucleus, "actin": front, "labels": labels,
        "volumes_um3": {"cell": v_cell, **vols}, "expected": expected,
        "nucleus_centre_um": (zc, 0.0, nucleus_shift_x), "nucleus_radius_um": nucleus_radius,
        "cell_axes_um": cell_axes, "cell_centre_z_um": zc,
        "grid_um": (z, y, x),
    }
    return stack, truth


def blur_and_noise(stack: ImageStack, sigma_xy_um=0.06, sigma_z_um=0.15, photons_per_unit=0.5, seed=0) -> ImageStack:
    """Gaussian approximation of the Airyscan PSF plus Poisson noise, for robustness tests."""
    from scipy import ndimage as ndi

    rng = np.random.default_rng(seed)
    dz, dy, dx = stack.voxel_um
    out = []
    for ch in stack.data.astype(np.float64):
        b = ndi.gaussian_filter(ch, sigma=(sigma_z_um / dz, sigma_xy_um / dy, sigma_xy_um / dx))
        n = rng.poisson(np.clip(b, 0, None) * photons_per_unit) / photons_per_unit
        out.append(np.clip(n, 0, 65535).astype(np.uint16))
    return ImageStack(data=np.stack(out), voxel_um=stack.voxel_um, channels=stack.channels,
                      source=None, info=dict(stack.info))
