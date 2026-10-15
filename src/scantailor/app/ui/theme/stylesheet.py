"""QSS generation for the light and dark looks."""

from PySide6 import QtGui

from .fonts import DISPLAY_RATIO, PANEL_HEADING_RATIO, SEMIBOLD, SMALL_RATIO, pixel_size
from .tokens import Tokens

CONTROL_HEIGHT = 32
# Spacing scale: every margin, padding and gap is one of these.
SPACE_XS = 4
SPACE_S = 8
SPACE_M = 12
SPACE_L = 16
SPACE_XL = 24
CHEVRON = 16
# Alpha of the text colour behind a hovered header row.
HOVER_ALPHA = 18
# Alpha of the text colour behind the selected stage. The stylesheet style paints a
# selected table cell's background twice, so it shows at about double this.
SELECTED_ALPHA = 16
# Width of the accent bar that marks the selected stage.
SELECTED_BAR = 3
RADIUS_PANEL = 12
RADIUS_CONTROL = 8
FIELD_BORDER = 1
FIELD_PADDING_X = 8


def text_tint(t: Tokens, alpha: int) -> str:
    """Return the text colour at an alpha, a neutral fill that follows the theme.

    Args:
        t (Tokens): Colours for the look.
        alpha (int): Alpha from 0 to 255.

    Returns:
        str: An rgba() colour for QSS.
    """
    text = QtGui.QColor(t.text)
    return f"rgba({text.red()}, {text.green()}, {text.blue()}, {alpha})"


def header_height(font: QtGui.QFont, nested: bool = False) -> int:
    """Return the height of a group header row, which grows with the font.

    Args:
        font (QtGui.QFont): Font of the group title.
        nested (bool): True for a group inside another group, which is tighter.

    Returns:
        int: Height in logical pixels; 32 for a card and 28 nested at 13 px Inter.
    """
    line = QtGui.QFontMetrics(font).height()
    return line + (3 * SPACE_XS if nested else 2 * SPACE_S)


def neutral_hover(t: Tokens) -> str:
    """Return the text colour at a low alpha, a hover fill that reads as no selection.

    Args:
        t (Tokens): Colours for the look.

    Returns:
        str: An rgba() colour for QSS.
    """
    text = QtGui.QColor(t.text)
    return f"rgba({text.red()}, {text.green()}, {text.blue()}, {HOVER_ALPHA})"


def build_stylesheet(t: Tokens, glyphs: dict[str, str], font: QtGui.QFont) -> str:
    """Build the application stylesheet for a token set.

    Body text takes the application font and is never sized here; the other
    sizes are ratios of it, so the stylesheet is rebuilt when that font changes.

    Args:
        t (Tokens): Colours for the look.
        glyphs (dict[str, str]): Glyph name to SVG path, from write_assets.
        font (QtGui.QFont): The application font, the body size.

    Returns:
        str: QSS text. The page image viewports are left untinted.
    """
    body = pixel_size(font)
    small_px = round(body * SMALL_RATIO)
    panel_heading_px = round(body * PANEL_HEADING_RATIO)
    display_px = round(body * DISPLAY_RATIO)
    line = QtGui.QFontMetrics(font).height()
    header = header_height(font)
    nested_header = header_height(font, nested=True)
    hover = neutral_hover(t)
    handle_fill = t.accent_text if t.dark else "#ffffff"
    control = CONTROL_HEIGHT - 2
    window_gradient = (
        f"qlineargradient(x1:0, y1:0, x2:0, y2:1, "
        f"stop:0 {t.bg_top}, stop:1 {t.bg_bottom})"
    )
    chevron_down = glyphs["chevron_down"]
    chevron_up = glyphs["chevron_up"]
    chevron_right = glyphs["chevron_right"]
    return f"""
QMainWindow, QDialog {{
    background: {window_gradient};
    color: {t.text};
}}
QWidget {{ color: {t.text}; }}
QToolTip {{
    background: {t.bg_bottom}; color: {t.text};
    border: 1px solid {t.field_border}; border-radius: 6px; padding: 4px 8px;
}}

QDockWidget {{
    font-size: {panel_heading_px}px; font-weight: {SEMIBOLD}; color: {t.text};
    titlebar-close-icon: url({glyphs["close"]});
    titlebar-normal-icon: url({glyphs["float"]});
}}
QDockWidget::close-button, QDockWidget::float-button {{
    width: 24px; height: 24px; border: none; border-radius: 6px;
    background: transparent;
}}
QDockWidget::close-button:hover, QDockWidget::float-button:hover {{
    background: {t.field_hover};
}}
QDockWidget::title {{
    background: transparent; text-align: left;
    padding: {SPACE_M}px {SPACE_S}px {SPACE_XS}px {SPACE_M + SPACE_S}px;
}}
QWidget#dockWidgetContents_5, QWidget#dockWidgetContents_2,
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QMainWindow::separator {{
    background: transparent; width: {SPACE_S}px; height: {SPACE_S}px;
    image: url({glyphs["grip"]});
}}
QMainWindow::separator:hover {{ background: {hover}; border-radius: {SPACE_XS}px; }}

QGroupBox {{
    background: {t.panel};
    border: 1px solid {t.panel_border};
    border-top: 1px solid {t.panel_highlight};
    border-radius: {RADIUS_PANEL}px;
    margin-top: 0;
    padding: {SPACE_XS + header + SPACE_S}px {SPACE_M}px {SPACE_M}px {SPACE_M}px;
    font-weight: {SEMIBOLD};
}}
QGroupBox[untitled="true"] {{ padding-top: {SPACE_M}px; }}
QGroupBox QGroupBox[untitled="true"] {{ padding-top: 0; }}
QGroupBox::title {{
    subcontrol-origin: padding; subcontrol-position: top left;
    left: {SPACE_M}px; top: {SPACE_XS + (header - line) // 2}px; padding: 0;
    color: {t.text};
}}
QGroupBox::indicator {{
    width: 16px; height: 16px; background: transparent; border: none;
    subcontrol-origin: padding; left: 8px; top: 8px;
}}
QGroupBox::indicator:unchecked {{ image: url({chevron_right}); }}
QGroupBox::indicator:checked {{ image: url({chevron_down}); }}
QGroupBox QGroupBox {{
    background: transparent; border: none; margin: 0;
    padding: {nested_header + SPACE_XS}px 0 0 0;
}}
QGroupBox QGroupBox::title {{
    left: 0; top: {(nested_header - line) // 2}px; color: {t.text_dim};
}}
QLabel {{ background: transparent; }}
QFrame[frameShape="4"] {{
    background: {t.field_border}; border: none; max-height: 1px;
}}
QFrame[frameShape="5"] {{
    background: {t.field_border}; border: none; max-width: 1px;
}}

QPushButton, QToolButton {{
    min-height: {control}px; min-width: 0; padding: 0 6px;
    background: {t.field}; border: 1px solid {t.field_border};
    border-radius: {RADIUS_CONTROL}px;
}}
QToolButton {{ min-width: {control}px; padding: 0; icon-size: 20px; }}
QPushButton:hover, QToolButton:hover {{ background: {t.field_hover}; }}
QPushButton:pressed, QToolButton:pressed {{ background: {t.accent_tint}; }}
QPushButton:checked, QToolButton:checked {{
    background: {t.accent_tint}; color: {t.text}; border: 1px solid {t.accent};
}}
QPushButton:default {{
    background: {t.accent}; color: {t.accent_text}; border-color: {t.accent};
}}
QPushButton:default:hover {{ background: {t.accent_hover}; }}
QPushButton:disabled, QToolButton:disabled {{
    color: {t.text_disabled}; background: {t.field_disabled};
    border-color: transparent;
}}
QGroupBox > QToolButton#groupHeader {{
    min-width: 0; min-height: 0; padding: 0 {SPACE_S}px; border: none;
    border-radius: {RADIUS_CONTROL}px; background: transparent;
    icon-size: {CHEVRON}px; font-weight: {SEMIBOLD}; color: {t.text};
}}
QGroupBox QGroupBox > QToolButton#groupHeader {{
    padding: 0 {SPACE_XS}px; border-radius: 6px; color: {t.text_dim};
}}
QGroupBox > QToolButton#groupHeader:hover,
QGroupBox > QToolButton#groupHeader:pressed {{ background: {hover}; }}
QGroupBox > QToolButton#groupHeader:focus {{ border: 2px solid {t.accent}; }}
QGroupBox > QToolButton#groupHeader:disabled {{
    background: transparent; color: {t.text_disabled};
}}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
    min-height: {control}px; padding: 0 {FIELD_PADDING_X}px;
    background: {t.field}; border: {FIELD_BORDER}px solid {t.field_border};
    border-radius: {RADIUS_CONTROL}px;
    selection-background-color: {t.accent}; selection-color: {t.accent_text};
}}
QPlainTextEdit, QTextEdit {{ padding: 6px 8px; }}
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{
    background: {t.field_hover};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
QPushButton:focus, QPlainTextEdit:focus, QTextEdit:focus {{
    border: 2px solid {t.accent};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
QComboBox:disabled, QPlainTextEdit:disabled, QTextEdit:disabled {{
    color: {t.text_disabled}; background: {t.field_disabled};
    border-color: transparent;
}}
QComboBox {{ padding-right: 28px; }}
QComboBox::drop-down {{
    subcontrol-origin: padding; subcontrol-position: center right;
    border: none; width: 24px;
}}
QComboBox::down-arrow {{ image: url({chevron_down}); width: 16px; height: 16px; }}
QComboBox QAbstractItemView {{
    background: {t.bg_top}; border: 1px solid {t.field_border};
    border-radius: {RADIUS_CONTROL}px; padding: 4px; outline: 0;
    selection-background-color: {t.accent_tint}; selection-color: {t.text};
}}
QSpinBox, QDoubleSpinBox {{ padding-right: 24px; }}
QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border; subcontrol-position: top right;
    width: 22px; height: 15px; border: none; background: transparent;
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border; subcontrol-position: bottom right;
    width: 22px; height: 15px; border: none; background: transparent;
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: url({chevron_up}); width: 14px; height: 14px;
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url({chevron_down}); width: 14px; height: 14px;
}}

QCheckBox, QRadioButton {{ spacing: 8px; min-height: 24px; background: transparent; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 18px; height: 18px; background: {t.field};
    border: 1px solid {t.indicator_border};
}}
QCheckBox::indicator {{ border-radius: 4px; }}
QRadioButton::indicator {{ border-radius: 10px; }}
QCheckBox:disabled, QRadioButton:disabled, QLabel:disabled,
QGroupBox:disabled {{ color: {t.text_disabled}; }}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
    background: {t.field_disabled}; border-color: {t.field_border};
}}
QCheckBox::indicator:checked:disabled {{
    background: {t.text_disabled}; border-color: {t.text_disabled};
}}
QRadioButton::indicator:checked:disabled {{
    background: {t.text_disabled}; border-color: {t.field_disabled};
}}
QCheckBox::indicator:checked {{
    background: {t.accent}; border-color: {t.accent}; image: url({glyphs["check"]});
}}
QRadioButton::indicator:checked {{
    background: {t.accent}; border: 5px solid {t.field_hover};
    border-color: {t.field_hover}; width: 8px; height: 8px;
}}

QSlider::groove:horizontal {{
    height: 4px; background: {t.field_border}; border-radius: 2px;
}}
QSlider::sub-page:horizontal {{ background: {t.accent}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    width: 16px; height: 16px; margin: -7px 0; border-radius: 9px;
    background: {handle_fill}; border: 2px solid {t.accent};
}}

QSlider::groove:horizontal:disabled {{ background: {t.field_disabled}; }}
QSlider::sub-page:horizontal:disabled {{ background: {t.text_disabled}; }}
QSlider::handle:horizontal:disabled {{
    background: {t.field_disabled}; border-color: {t.text_disabled};
}}

QListView, QListWidget, QTreeView, QTableView {{
    background: transparent; border: none; outline: 0;
    alternate-background-color: transparent;
}}
QListView::item, QListWidget::item, QTreeView::item, QTableView::item {{
    border-radius: 6px; padding: 4px 8px; margin: 1px 4px;
}}
QListView::item:hover, QListWidget::item:hover,
QTreeView::item:hover, QTableView::item:hover {{
    background: {t.field};
}}
QListView::item:selected, QListWidget::item:selected, QTreeView::item:selected {{
    background: {t.accent_tint}; color: {t.text};
}}
QTableView::item {{ border: 0; border-radius: 0; margin: 0; padding: 4px 8px; }}
QTableView::item:selected {{ background: {t.accent_tint}; color: {t.text}; }}
QTableView QPushButton, QTableView QToolButton {{
    background: transparent; border: none; min-height: 0; padding: 0;
}}
QTableView QHeaderView, QTableCornerButton::section {{
    background: transparent; border: none;
}}
QTableView#filterList {{
    background: {t.panel}; border: 1px solid {t.panel_border};
    border-top: 1px solid {t.panel_highlight}; border-radius: {RADIUS_PANEL}px;
    padding: {SPACE_XS}px;
    selection-background-color: transparent;
}}
QTableView#filterList:focus {{ border: 1px solid {t.accent}; }}
QTableView#filterList::item:selected {{
    background: {text_tint(t, SELECTED_ALPHA)}; color: {t.text};
}}
QTableView#filterList::item:first {{
    border-left: {SELECTED_BAR}px solid transparent;
}}
QTableView#filterList::item:selected:first {{
    border-left: {SELECTED_BAR}px solid {t.accent};
}}
QHeaderView::section {{
    background: transparent; border: none; padding: 6px 8px; color: {t.text_dim};
}}

QGraphicsView {{ background: {t.viewport}; border: none; }}
QGraphicsView#thumbView {{
    background: {t.panel}; border: 1px solid {t.panel_border};
    border-radius: {RADIUS_PANEL}px;
}}
QFrame#imageViewFrame {{
    background: {t.viewport}; border: 1px solid {t.panel_border};
    border-radius: 10px; padding: 2px;
}}
QLabel#newProjectLabel, QLabel#openProjectLabel, QLabel[textRole="display"] {{
    font-size: {display_px}px; font-weight: {SEMIBOLD};
}}
QLabel[textRole="small"] {{ font-size: {small_px}px; color: {t.text_dim}; }}

QMenuBar {{ background: transparent; padding: 2px 8px; }}
QMenuBar::item {{ padding: 6px 10px; border-radius: 6px; background: transparent; }}
QMenuBar::item:selected {{ background: {t.accent_tint}; }}
QMenu {{
    background: {t.bg_top}; border: 1px solid {t.field_border};
    border-radius: 10px; padding: 6px;
}}
QMenu::item {{ padding: 6px 24px 6px 16px; border-radius: 6px; }}
QMenu::item:selected {{ background: {t.accent_tint}; color: {t.text}; }}
QMenu::item:disabled {{ color: {t.text_dim}; }}
QMenu::separator {{ height: 1px; background: {t.field_border}; margin: 6px 8px; }}

QToolBar {{ background: transparent; border: none; spacing: 4px; padding: 4px; }}
QStatusBar {{
    background: transparent; color: {t.text_dim}; font-size: {small_px}px;
}}
QStatusBar::item {{ border: none; }}

QTabWidget::pane {{
    border: 1px solid {t.panel_border}; border-radius: {RADIUS_PANEL}px;
}}
QTabBar {{ qproperty-drawBase: 0; }}
QTabBar::tab {{
    padding: 6px 14px; border-radius: 8px; margin: 2px; background: transparent;
}}
QTabBar::tab:selected {{ background: {t.accent_tint}; }}
QTabBar::tab:hover:!selected {{ background: {t.field}; }}

QProgressBar {{
    min-height: 8px; max-height: 8px; text-align: center; color: transparent;
    background: {t.field_border}; border: none; border-radius: 4px;
}}
QProgressBar::chunk {{ background: {t.accent}; border-radius: 4px; }}

QScrollArea {{ background: transparent; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 12px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 12px; margin: 2px; }}
QScrollBar::handle:vertical {{
    background: {t.scroll_handle}; border-radius: 3px; min-height: 32px; margin: 0 1px;
}}
QScrollBar::handle:horizontal {{
    background: {t.scroll_handle}; border-radius: 3px; min-width: 32px; margin: 1px 0;
}}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{
    margin: 0; border-radius: 5px; background: {t.accent};
}}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
"""
