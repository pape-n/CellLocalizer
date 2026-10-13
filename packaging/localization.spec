# PyInstaller spec for CellLocalizer (Windows, macOS, Linux).
# Build from the repository root:  pyinstaller packaging/localization.spec --noconfirm
# Result: dist/CellLocalizer/ (folder with CellLocalizer.exe) or dist/CellLocalizer.app on macOS.
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_all, collect_submodules, copy_metadata

ROOT = Path(SPECPATH).parent
sys.path.insert(0, str(ROOT))
from localization_tool import __version__  # noqa: E402

NAME = "CellLocalizer"
icon_file = ROOT / "packaging" / ("icon.icns" if sys.platform == "darwin" else "icon.ico")
ICON = str(icon_file) if icon_file.exists() else None

datas, binaries, hiddenimports = [], [], []
# napari and its helpers load resources, themes, icons and plugin manifests at runtime
for pkg in ("napari", "vispy", "napari_svg", "npe2", "app_model", "magicgui", "superqt",
            "in_n_out", "psygnal", "pydantic_compat", "napari_builtins"):
    try:
        d, b, h = collect_all(pkg)
    except Exception:
        continue
    datas += d
    binaries += b
    hiddenimports += h
# plugin discovery uses package metadata (entry points)
for dist in ("napari", "napari-svg", "npe2", "app-model", "magicgui", "imageio", "numpy", "tifffile",
             "scikit-image", "pydantic"):
    try:
        datas += copy_metadata(dist)
    except Exception:
        pass
# program resources (icon) and the marching cubes module for the 3D surfaces
datas += [(str(ROOT / "localization_tool" / "resources"), "localization_tool/resources")]
hiddenimports += collect_submodules("czifile") + collect_submodules("imagecodecs")
# Nikon ND2 (D90): the reader and its metadata models
hiddenimports += collect_submodules("nd2") + collect_submodules("ome_types")
for dist in ("nd2", "ome-types"):
    try:
        datas += copy_metadata(dist)
    except Exception:
        pass
# python-docx loads its Word template from package data (methods text, D58)
datas += collect_data_files("docx")
hiddenimports += collect_submodules("docx")
hiddenimports += ["skimage.measure._marching_cubes_lewiner", "skimage.measure._marching_cubes_lewiner_cy"]
# scipy 1.18 vendors array_api_compat under scipy._external, not yet known to the PyInstaller hooks
hiddenimports += collect_submodules("scipy._external")
hiddenimports += ["qtpy", "PySide6.QtOpenGL", "PySide6.QtOpenGLWidgets", "PySide6.QtSvg"]

a = Analysis(
    [str(ROOT / "packaging" / "launch.py")],
    pathex=[str(ROOT)],
    datas=datas,
    binaries=binaries,
    hiddenimports=hiddenimports,
    excludes=[
        "tkinter", "_tkinter", "IPython", "ipykernel", "qtconsole", "jupyter_client", "jupyter_core",
        "napari_console", "jedi", "debugpy", "sphinx", "babel", "pytest",
        "PyQt5", "PyQt6", "PySide2", "matplotlib", "torch", "tensorflow",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=NAME,
    console=False,
    icon=ICON,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name=NAME, upx=False)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=f"{NAME}.app",
        icon=ICON,
        bundle_identifier="de.uni-muenster.physiologie2.localizationtool",
        version=__version__,
        info_plist={
            "CFBundleName": "CellLocalizer",
            "CFBundleDisplayName": "CellLocalizer",
            "CFBundleShortVersionString": __version__,
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "12.0",
            "NSHumanReadableCopyright": "Copyright (c) 2026 Nils Pape, BSD-3-Clause",
        },
    )
