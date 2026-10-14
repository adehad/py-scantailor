"""Application-wide current unit, observable by the widgets that show lengths."""

from PySide6 import QtCore

from scantailor.core.settings import ApplicationSettings, Units


class UnitsProvider(QtCore.QObject):
    """Holds the unit lengths are shown in and announces changes.

    Signals:
        units_changed: The unit changed. Args: units (Units)
    """

    units_changed = QtCore.Signal(object)

    _instance: UnitsProvider | None = None

    def __init__(self, units: Units = Units.MILLIMETERS) -> None:
        """Create a provider; use ``get_instance`` for the shared one.

        Args:
            units (Units): The initial unit.
        """
        super().__init__()
        self._units = units

    @classmethod
    def get_instance(cls) -> UnitsProvider:
        """Return the shared provider, starting from the stored setting.

        Returns:
            UnitsProvider: The application-wide provider.
        """
        if cls._instance is None:
            cls._instance = cls(ApplicationSettings.get_instance().units)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Forget the shared provider (mainly for testing)."""
        cls._instance = None

    @property
    def units(self) -> Units:
        """The unit lengths are shown in."""
        return self._units

    def set_units(self, units: Units) -> None:
        """Change the unit and tell the listeners if it differs.

        Args:
            units (Units): The new unit.
        """
        if units is self._units:
            return
        self._units = units
        self.units_changed.emit(units)
