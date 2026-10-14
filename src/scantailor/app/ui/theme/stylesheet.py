"""QSS generation for the light and dark looks."""

from .fonts import BASE_PX, CAPTION_PX, HEADING_PX, SEMIBOLD
from .tokens import Tokens

CONTROL_HEIGHT = 32
RADIUS_PANEL = 12
RADIUS_CONTROL = 8
FIELD_BORDER = 1
FIELD_PADDING_X = 8


def build_stylesheet(t: Tokens, glyphs: dict[str, str]) -> str:
    """Build the application stylesheet for a token set.

    Args:
        t (Tokens): Colours for the look.
        glyphs (dict[str, str]): Glyph name to SVG path, from write_assets.

    Returns:
        str: QSS text. The page image viewports are left untinted.
    """
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
    font-size: {CAPTION_PX}px; font-weight: {SEMIBOLD}; color: {t.text_dim};
    titlebar-close-icon: url({glyphs["close"]});
    titlebar-normal-icon: url({glyphs["float"]});
}}
QDockWidget::close-button, QDockWidget::float-button {{
    width: 20px; height: 20px; border: none; border-radius: 6px;
    background: transparent;
}}
QDockWidget::close-button:hover, QDockWidget::float-button:hover {{
    background: {t.field_hover};
}}
QDockWidget::title {{
    background: transparent; padding: 8px 12px; text-align: left;
}}
QWidget#dockWidgetContents, QWidget#dockWidgetContents_2,
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QMainWindow::separator {{ background: transparent; width: 8px; height: 8px; }}

QGroupBox {{
    background: {t.panel};
    border: 1px solid {t.panel_border};
    border-top: 1px solid {t.panel_highlight};
    border-radius: {RADIUS_PANEL}px;
    margin-top: 0; padding: 36px 8px 8px 8px;
    font-size: {BASE_PX}px; font-weight: {SEMIBOLD};
}}
QGroupBox::title {{
    subcontrol-origin: padding; subcontrol-position: top left;
    left: 12px; top: 10px; padding: 0; color: {t.text};
}}
QGroupBox::indicator {{
    width: 16px; height: 16px; background: transparent; border: none;
    subcontrol-origin: padding; left: 8px; top: 8px;
}}
QGroupBox::indicator:unchecked {{ image: url({chevron_right}); }}
QGroupBox::indicator:checked {{ image: url({chevron_down}); }}
QGroupBox QGroupBox {{
    background: transparent; border: none; margin: 0; padding: 20px 0 0 0;
}}
QGroupBox QGroupBox::title {{ left: 0; top: 0; color: {t.text_dim}; }}
QToolButton#collapseButton {{
    min-width: 0; min-height: 0; padding: 0; border: none; border-radius: 6px;
    background: transparent; icon-size: 14px;
}}
QToolButton#collapseButton:hover {{ background: {t.accent_tint}; }}
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
QLabel#newProjectLabel, QLabel#openProjectLabel {{
    font-size: {HEADING_PX}px; font-weight: {SEMIBOLD};
}}

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
    background: transparent; color: {t.text_dim}; font-size: {CAPTION_PX}px;
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
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px 4px 2px 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{
    background: {t.scroll_handle}; border-radius: 3px; min-height: 32px; margin: 0 2px;
}}
QScrollBar::handle:horizontal {{
    background: {t.scroll_handle}; border-radius: 3px; min-width: 32px; margin: 2px 0;
}}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{
    margin: 0; border-radius: 5px; background: {t.accent};
}}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
"""
