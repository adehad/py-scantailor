"""Conversion between the measurement units the user can choose.

Margins and page sizes are stored in millimetres; other units exist only for
display and entry.
"""

from scantailor.core.settings import Units

_MM_PER_INCH = 25.4
_MM_PER_PHYSICAL_UNIT = {
    Units.MILLIMETERS: 1.0,
    Units.CENTIMETERS: 10.0,
    Units.INCHES: _MM_PER_INCH,
}
# Short names for labels; the same codes C++ writes to its settings.
_SUFFIX = {
    Units.PIXELS: "px",
    Units.MILLIMETERS: "mm",
    Units.CENTIMETERS: "cm",
    Units.INCHES: "in",
}


def units_suffix(units: Units) -> str:
    """Return the abbreviation shown next to a value.

    Args:
        units (Units): The unit.

    Returns:
        str: "px", "mm", "cm" or "in".
    """
    return _SUFFIX[units]


def spin_box_precision(units: Units) -> tuple[int, float]:
    """Return how finely a value in ``units`` is shown and stepped.

    Args:
        units (Units): The unit.

    Returns:
        tuple[int, float]: Decimals and single step; whole steps for the small
            units, hundredths for centimetres and inches.
    """
    if units in (Units.PIXELS, Units.MILLIMETERS):
        return 1, 1.0
    return 2, 0.01


def convert_length(
    value: float,
    from_units: Units,
    to_units: Units,
    dpi: float,
) -> float:
    """Convert one length along an axis.

    Args:
        value (float): The length in ``from_units``.
        from_units (Units): Unit of ``value``.
        to_units (Units): Unit to convert to.
        dpi (float): Resolution along the axis; only pixels need it, and a value
            of 1 or less means unknown.

    Returns:
        float: The length in ``to_units``. A conversion that needs an unknown
            resolution returns ``value`` unchanged, as C++ does for every
            conversion in that case.
    """
    if from_units is to_units:
        return value
    if Units.PIXELS in (from_units, to_units) and dpi <= 1:
        return value
    return _from_millimetres(_to_millimetres(value, from_units, dpi), to_units, dpi)


def _to_millimetres(value: float, units: Units, dpi: float) -> float:
    if units is Units.PIXELS:
        return value / dpi * _MM_PER_INCH
    return value * _MM_PER_PHYSICAL_UNIT[units]


def _from_millimetres(value: float, units: Units, dpi: float) -> float:
    if units is Units.PIXELS:
        return value / _MM_PER_INCH * dpi
    return value / _MM_PER_PHYSICAL_UNIT[units]
