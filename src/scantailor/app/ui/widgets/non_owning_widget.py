"""Non-owning widget that doesn't delete its children.

A QWidget that calls setParent(None) on its children instead of deleting them.
This is useful for container widgets where the children may be managed elsewhere.
"""

from PySide6.QtWidgets import QWidget


class NonOwningWidget(QWidget):
    """A QWidget that doesn't delete its children when destroyed.

    Instead of deleting child widgets, this widget calls setParent(None) on them,
    allowing the children to be managed elsewhere or continue to exist.

    This is useful for dynamic layouts where widgets are frequently moved between
    different containers.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the non-owning widget.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)

    def __del__(self) -> None:
        """Clean up by unparenting children instead of deleting them."""
        try:
            # Unparent all child widgets before this widget is destroyed
            for child in self.findChildren(QWidget):
                if child.parent() is self:
                    child.setParent(None)
        except (RuntimeError, AttributeError):
            # Widget may already be deleted in C++ layer
            pass
