"""Container widget whose children outlive it."""

from PySide6.QtWidgets import QWidget


class NonOwningWidget(QWidget):
    """A QWidget that hands its child widgets back instead of deleting them.

    Qt deletes children inside the C++ destructor, before any Python hook can
    run, so this cannot intercept a parent that is destroyed implicitly. It
    releases the children when the container is disposed of with ``deleteLater``
    or ``release_children``, which is how dynamic layouts that move widgets
    between containers dispose of one.
    """

    def release_children(self) -> None:
        """Unparent every direct child widget so deleting the container spares it."""
        for child in self.findChildren(QWidget):
            if child.parent() is self:
                child.hide()
                child.setParent(None)

    def deleteLater(self) -> None:
        """Release the children, then schedule this widget for deletion."""
        self.release_children()
        super().deleteLater()
