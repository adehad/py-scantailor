"""The options widget and image view that belong to a filter stage."""

import numpy as np
from numpy.typing import NDArray
from PySide6 import QtCore, QtWidgets

from scantailor.app.ui.image_views.base import ImageViewBase
from scantailor.app.ui.image_views.deskew import DeskewImageView
from scantailor.app.ui.image_views.fix_orientation import FixOrientationImageView
from scantailor.app.ui.image_views.output import OutputImageView
from scantailor.app.ui.image_views.page_layout import PageLayoutImageView
from scantailor.app.ui.image_views.page_split import PageSplitImageView
from scantailor.app.ui.image_views.select_content import SelectContentImageView
from scantailor.app.ui.options_widgets import (
    deskew,
    fix_orientation,
    output,
    page_layout,
    page_split,
    select_content,
)
from scantailor.core.models import PageId
from scantailor.core.stage_sequence import FilterStage, StageSequence

_VIEW_CLASSES: dict[FilterStage, type[ImageViewBase]] = {
    FilterStage.FIX_ORIENTATION: FixOrientationImageView,
    FilterStage.PAGE_SPLIT: PageSplitImageView,
    FilterStage.DESKEW: DeskewImageView,
    FilterStage.SELECT_CONTENT: SelectContentImageView,
    FilterStage.PAGE_LAYOUT: PageLayoutImageView,
    FilterStage.OUTPUT: OutputImageView,
}


class StagePanels(QtCore.QObject):
    """Options widget and image view of one stage, bound to a stage sequence.

    The options widgets write their edits to the stage's settings themselves;
    this class reports that an edit happened.

    Signals:
        settings_changed: The user edited an option of the current page.
    """

    settings_changed = QtCore.Signal()

    def __init__(
        self,
        stage: FilterStage,
        sequence: StageSequence,
        parent: QtCore.QObject | None = None,
    ) -> None:
        """Build the widgets of ``stage``.

        Args:
            stage (FilterStage): The stage to show.
            sequence (StageSequence): Stages whose settings the options edit.
            parent (QtCore.QObject | None): Owner of this object.
        """
        super().__init__(parent)
        self.stage = stage
        self.sequence = sequence
        self.options: QtWidgets.QWidget = self._build_options(stage, sequence)
        self.view: ImageViewBase = _VIEW_CLASSES[stage]()

    def _build_options(
        self, stage: FilterStage, sequence: StageSequence
    ) -> QtWidgets.QWidget:
        forward = self._on_option_edited
        match stage:
            case FilterStage.FIX_ORIENTATION:
                widget = fix_orientation.OptionsWidget(sequence.fix_orientation_filter)
                widget.rotation_changed.connect(forward)
            case FilterStage.PAGE_SPLIT:
                widget = page_split.OptionsWidget(sequence.page_split_filter)
                widget.layout_type_changed.connect(forward)
                widget.split_mode_changed.connect(forward)
            case FilterStage.DESKEW:
                widget = deskew.OptionsWidget(sequence.deskew_filter)
                widget.angle_changed.connect(forward)
            case FilterStage.SELECT_CONTENT:
                widget = select_content.OptionsWidget(sequence.select_content_filter)
                widget.page_mode_changed.connect(forward)
                widget.content_mode_changed.connect(forward)
                widget.fine_tune_changed.connect(forward)
            case FilterStage.PAGE_LAYOUT:
                widget = page_layout.OptionsWidget(sequence.page_layout_filter)
                widget.margins_changed.connect(forward)
                widget.alignment_changed.connect(forward)
            case FilterStage.OUTPUT:
                widget = output.OptionsWidget(sequence.output_filter)
                widget.params_changed.connect(forward)
        return widget

    def _on_option_edited(self, *_args: object) -> None:
        # The widgets have already stored the new values, so the signal
        # arguments are not needed.
        self.settings_changed.emit()

    def set_page(self, page_id: PageId | None) -> None:
        """Make the options edit the settings of a page.

        Args:
            page_id (PageId | None): The page, or None when there is none.
        """
        self.options.setEnabled(page_id is not None)
        if page_id is None:
            return
        if isinstance(self.options, fix_orientation.OptionsWidget):
            self.options.set_current_image(page_id.image_id)
        else:
            self.options.set_current_page(page_id)  # type: ignore[attr-defined]

    def show_image(self, image: NDArray[np.uint8]) -> None:
        """Display a rendered page.

        Args:
            image (NDArray[np.uint8]): ``(H, W)`` gray or ``(H, W, 3)`` RGB pixels.
        """
        self.view.set_image(image)
