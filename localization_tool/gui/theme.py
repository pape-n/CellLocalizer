"""Colours, fonts, style sheet and napari theme (docs/DESIGN.md, D61). All colours go through TOKENS.

Dark surroundings, because fluorescence images are viewed on black, with the component system of the
Chemotaxis Tool: Arial, one accent colour (#2A6496) for the primary button and the active step,
secondary buttons with a frame, radius 4 px for buttons and 8 px for cards, no shadows. Buttons are
larger than usual (at least 32 px, step buttons 40 px) for lab screens. The text size can be
scaled to 115 or 130 % in the settings; every size below is multiplied by that factor.
"""

from __future__ import annotations

FONT_FAMILY = "Arial"
FONT_FALLBACKS = ("Arial", "Liberation Sans", "Helvetica")

TOKENS = {
    "canvas": "#000000",
    "bg": "#1B1F24",            # window
    "surface": "#242A31",       # panels, input fields, cards
    "surface_alt": "#2E353D",   # secondary buttons, table headers
    "border": "#3B434D",
    "border_strong": "#58626E",
    "text": "#E8EBEF",
    "text_muted": "#A7B1BC",
    "accent": "#2A6496",        # primary button, active step (the Chemotaxis Tool accent)
    "accent_hover": "#3473AA",
    "accent_text": "#7FB2E0",   # accent as text on the dark background (links, current step label)
    "accent_soft": "#22384D",   # selection
    "ok": "#5CC08C",
    "warn": "#E3B451",
    "error": "#EE7A73",
}

# text sizes in pt before scaling: title, page title, section, body, hint
SIZES = {"title": 24, "page": 20, "section": 14, "body": 11, "hint": 10}

THEME_ID = "localization"


def _luminance(hex_color: str) -> float:
    c = hex_color.lstrip("#")
    rgb = [int(c[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(fg: str, bg: str) -> float:
    """WCAG contrast ratio of two colours (1 to 21)."""
    a, b = sorted((_luminance(fg), _luminance(bg)), reverse=True)
    return (a + 0.05) / (b + 0.05)


def readable(color: str, bg: str | None = None, minimum: float = 4.5) -> str:
    """The colour, lightened towards white until it reaches the contrast ratio on the background (for
    coloured labels such as the compartment check boxes)."""
    bg = bg or TOKENS["bg"]
    c = color.lstrip("#")
    rgb = [int(c[i:i + 2], 16) for i in (0, 2, 4)]
    for _ in range(20):
        out = "#" + "".join(f"{v:02X}" for v in rgb)
        if contrast(out, bg) >= minimum:
            return out
        rgb = [min(255, int(round(v + 0.12 * (255 - v)))) for v in rgb]
    return "#FFFFFF"


def _icon(name: str) -> str:
    """Path of an icon in the resources for url() in a style sheet (forward slashes, also on Windows)."""
    from localization_tool.paths import resource

    return str(resource(name)).replace("\\", "/")


def fit_to_screen(widget, width: int, height: int) -> None:
    """Resize, but never larger than the free area of the screen (1280 x 720 at 125 % leaves 1024 x 497)."""
    screen = widget.screen() if hasattr(widget, "screen") and widget.screen() is not None else None
    if screen is None:
        from qtpy.QtWidgets import QApplication

        screen = QApplication.primaryScreen()
    if screen is not None:
        geo = screen.availableGeometry()
        width, height = min(width, geo.width() - 40), min(height, geo.height() - 60)
    widget.resize(max(320, width), max(240, height))


def pt(kind: str, scale: float = 1.0) -> float:
    return round(SIZES[kind] * scale, 1)


def register_napari_theme() -> str:
    """Register the program theme with napari and return its id."""
    from napari.utils.theme import available_themes, get_theme, register_theme

    if THEME_ID in available_themes():
        return THEME_ID
    theme = get_theme("dark")
    theme.id = THEME_ID
    theme.label = "CellLocalizer"
    theme.canvas = TOKENS["canvas"]
    theme.background = TOKENS["bg"]
    theme.foreground = TOKENS["surface"]
    theme.primary = TOKENS["surface_alt"]
    theme.secondary = TOKENS["border"]
    theme.highlight = TOKENS["accent"]
    theme.current = TOKENS["accent_text"]
    theme.text = TOKENS["text"]
    theme.icon = TOKENS["text_muted"]
    theme.warning = TOKENS["warn"]
    theme.error = TOKENS["error"]
    register_theme(THEME_ID, theme, "localization_tool")
    return THEME_ID


def stylesheet(scale: float = 1.0) -> str:
    """Style sheet for the program's own widgets: start window, workflow panel, dialogs."""
    t = TOKENS
    body, hint, section, page, title = (pt(k, scale) for k in ("body", "hint", "section", "page", "title"))
    pad = int(round(8 * scale))
    up, down, check = _icon("arrow_up.svg"), _icon("arrow_down.svg"), _icon("check.svg")
    return f"""
    QWidget {{ font-family: {FONT_FAMILY}; font-size: {body}pt; color: {t['text']}; }}
    QDialog, QWidget#page, QWidget#start {{ background: {t['bg']}; }}
    QLabel {{ background: transparent; }}
    QLabel[role="apptitle"] {{ font-size: {title}pt; font-weight: bold; }}
    QLabel[role="page"] {{ font-size: {page}pt; font-weight: bold; }}
    QLabel[role="title"] {{ font-size: {section}pt; font-weight: bold; }}
    QLabel[role="hint"] {{ color: {t['text_muted']}; font-size: {hint}pt; }}
    QLabel[role="warn"] {{ color: {t['warn']}; }}
    QLabel[role="error"] {{ color: {t['error']}; }}
    QLabel[role="ok"] {{ color: {t['ok']}; }}
    QLabel[role="instruction"] {{ font-size: {section}pt; font-weight: bold; padding: 4px 0; }}

    QFrame#card {{ background: {t['surface']}; border: 1px solid {t['border']}; border-radius: 8px; }}
    QFrame#card:hover {{ border-color: {t['accent_text']}; }}
    QFrame#separator {{ background: {t['border']}; max-height: 1px; min-height: 1px; }}

    QPushButton {{ background: {t['surface_alt']}; color: {t['text']}; border: 1px solid {t['border_strong']};
                   border-radius: 4px; padding: {pad}px {2 * pad}px; min-height: {int(16 * scale)}px; }}
    QPushButton:hover {{ border-color: {t['accent_text']}; }}
    QPushButton:focus {{ border: 2px solid {t['accent_text']}; }}
    QPushButton:pressed {{ background: {t['border']}; }}
    QPushButton:disabled {{ color: #6E7884; border-color: {t['border']}; background: {t['surface']}; }}
    QPushButton[role="primary"] {{ background: {t['accent']}; color: white; border: 1px solid {t['accent']}; }}
    QPushButton[role="primary"]:hover {{ background: {t['accent_hover']}; border-color: {t['accent_hover']}; }}
    QPushButton[role="primary"]:focus {{ border: 2px solid {t['accent_text']}; }}
    QPushButton[role="primary"]:disabled {{ background: {t['border']}; border-color: {t['border']};
                                           color: #8A949F; }}
    QPushButton[size="large"] {{ min-height: {int(26 * scale)}px; padding: {pad}px {3 * pad}px; font-weight: bold; }}
    QPushButton[size="compact"] {{ padding: 4px 10px; min-height: {int(14 * scale)}px; }}
    QFrame#viewbar {{ background: rgba(27, 31, 36, 215); border: 1px solid {t['border']}; border-radius: 8px; }}
    QPushButton[role="flat"] {{ background: transparent; border: none; color: {t['text_muted']};
                                padding: 4px 8px; }}
    QPushButton[role="flat"]:hover {{ color: {t['accent_text']}; }}
    QPushButton:checked {{ background: {t['accent_soft']}; color: {t['text']}; border: 1px solid {t['accent_text']}; }}

    QPushButton[step] {{ background: transparent; border: 1px solid transparent; border-radius: 4px;
                         padding: 4px 6px; min-height: 0px; color: {t['text_muted']}; font-weight: normal; }}
    QPushButton[step]:hover {{ color: {t['accent_text']}; border-color: {t['border_strong']}; }}
    QPushButton[step="current"] {{ color: {t['accent_text']}; font-weight: bold; border-radius: 0px;
                                   border-bottom: 2px solid {t['accent_text']}; }}
    QPushButton[step="done"] {{ color: {t['text']}; }}
    QPushButton[step="open"] {{ color: {t['text_muted']}; }}
    QPushButton[step="locked"], QPushButton[step]:disabled {{ color: #5E6873; background: transparent;
                                                             border-color: transparent; }}
    QFrame#stepline {{ background: {t['border']}; border: none; }}
    QFrame#stepline[done="true"] {{ background: {t['text_muted']}; }}

    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit, QTextBrowser {{
        background: {t['surface']}; color: {t['text']}; border: 1px solid {t['border_strong']}; border-radius: 4px;
        padding: 4px 6px; min-height: {int(18 * scale)}px; selection-background-color: {t['accent']};
        selection-color: white; }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus {{
        border-color: {t['accent_text']}; }}
    QComboBox QAbstractItemView {{ background: {t['surface']}; color: {t['text']};
                                   selection-background-color: {t['accent']}; selection-color: white; }}
    QCheckBox {{ spacing: 6px; }}
    QCheckBox::indicator {{ width: {int(14 * scale)}px; height: {int(14 * scale)}px; border: 1px solid {t['text_muted']};
                            border-radius: 3px; background: {t['surface']}; }}
    QCheckBox::indicator:hover {{ border-color: {t['accent_text']}; }}
    QCheckBox::indicator:checked {{ background: {t['accent']}; border-color: {t['accent_text']}; image: url("{check}"); }}
    QCheckBox::indicator:disabled {{ border-color: {t['border']}; background: {t['bg']}; }}
    QSpinBox, QDoubleSpinBox {{ padding-right: 22px; }}
    QSpinBox::up-button, QDoubleSpinBox::up-button {{ subcontrol-origin: border; subcontrol-position: top right;
        width: 20px; border-left: 1px solid {t['border_strong']}; border-bottom: 1px solid {t['border']};
        background: {t['surface_alt']}; border-top-right-radius: 4px; }}
    QSpinBox::down-button, QDoubleSpinBox::down-button {{ subcontrol-origin: border; subcontrol-position: bottom right;
        width: 20px; border-left: 1px solid {t['border_strong']}; background: {t['surface_alt']};
        border-bottom-right-radius: 4px; }}
    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover, QSpinBox::down-button:hover,
    QDoubleSpinBox::down-button:hover {{ background: {t['border_strong']}; }}
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url("{up}"); width: 9px; height: 9px; }}
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url("{down}"); width: 9px; height: 9px; }}
    QComboBox {{ padding-right: 24px; }}
    QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: center right; width: 22px;
        border-left: 1px solid {t['border_strong']}; }}
    QComboBox::down-arrow {{ image: url("{down}"); width: 10px; height: 10px; }}
    QListWidget, QTableWidget, QTableView {{ background: {t['surface']}; color: {t['text']};
        border: 1px solid {t['border']}; border-radius: 6px; gridline-color: {t['border']};
        selection-background-color: {t['accent']}; selection-color: white; }}
    QListWidget::item {{ padding: 3px 4px; }}
    QHeaderView::section {{ background: {t['surface_alt']}; color: {t['text_muted']}; border: none;
        border-bottom: 1px solid {t['border']}; padding: 5px; font-weight: bold; }}
    QTabWidget::pane {{ border: 1px solid {t['border']}; border-radius: 8px; background: {t['surface']}; }}
    QTabBar::tab {{ background: transparent; color: {t['text_muted']}; padding: 8px 16px;
                    border-bottom: 2px solid transparent; }}
    QTabBar::tab:selected {{ color: {t['text']}; border-bottom: 2px solid {t['accent_text']}; }}
    QGroupBox {{ border: 1px solid {t['border']}; border-radius: 8px; margin-top: 14px; padding: 12px;
                 background: {t['surface']}; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 4px; color: {t['text_muted']};
                        font-weight: bold; }}
    QProgressBar {{ background: {t['surface']}; border: 1px solid {t['border']}; border-radius: 6px;
                    text-align: center; min-height: 12px; }}
    QProgressBar::chunk {{ background: {t['accent']}; border-radius: 5px; }}
    QScrollArea {{ background: transparent; border: none; }}
    QToolTip {{ background: {t['surface_alt']}; color: {t['text']}; border: 1px solid {t['border_strong']};
                padding: 4px; }}

    /* keyboard focus visible on every clickable thing (after the rules above, which would override it) */
    QFrame#card:focus {{ border: 2px solid {t['accent_text']}; }}
    QPushButton[step]:focus, QPushButton[role="flat"]:focus {{ border: 2px solid {t['accent_text']}; }}
    """


def style(widget, scale: float | None = None) -> None:
    """Apply the program style to a top-level widget or dialog (not to napari's own window)."""
    if scale is None:
        from localization_tool.prefs import font_scale

        scale = font_scale()
    widget.setStyleSheet(stylesheet(scale))


def dock_stylesheet(scale: float | None = None) -> str:
    """Kept for callers of version 1.1: the panel uses the same style sheet as everything else."""
    if scale is None:
        from localization_tool.prefs import font_scale

        scale = font_scale()
    return stylesheet(scale)
