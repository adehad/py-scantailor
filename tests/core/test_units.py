"""Tests for the unit conversions behind the margin display."""

import pytest

from scantailor.core.settings import Units
from scantailor.core.units import convert_length, spin_box_precision, units_suffix


class TestConvertLength:
    @pytest.mark.parametrize(
        ("value", "source", "target", "expected"),
        [
            (25.4, Units.MILLIMETERS, Units.INCHES, 1.0),
            (25.4, Units.MILLIMETERS, Units.CENTIMETERS, 2.54),
            (1.0, Units.INCHES, Units.MILLIMETERS, 25.4),
            (2.54, Units.CENTIMETERS, Units.INCHES, 1.0),
            (10.0, Units.MILLIMETERS, Units.MILLIMETERS, 10.0),
        ],
    )
    def test_physical_units_convert_without_a_resolution(
        self, value: float, source: Units, target: Units, expected: float
    ) -> None:
        assert convert_length(value, source, target, dpi=0) == pytest.approx(expected)

    def test_pixels_use_the_resolution_of_the_axis(self) -> None:
        assert convert_length(300, Units.PIXELS, Units.INCHES, dpi=300) == (
            pytest.approx(1.0)
        )
        assert convert_length(25.4, Units.MILLIMETERS, Units.PIXELS, dpi=600) == (
            pytest.approx(600.0)
        )

    def test_pixels_without_a_resolution_leave_the_value_unchanged(self) -> None:
        assert convert_length(12.0, Units.MILLIMETERS, Units.PIXELS, dpi=0) == 12.0
        assert convert_length(12.0, Units.PIXELS, Units.INCHES, dpi=1) == 12.0

    @pytest.mark.parametrize("source", list(Units))
    @pytest.mark.parametrize("target", list(Units))
    def test_round_trip_returns_the_start_value(
        self, source: Units, target: Units
    ) -> None:
        there = convert_length(37.5, source, target, dpi=400)

        assert convert_length(there, target, source, dpi=400) == pytest.approx(37.5)


class TestPresentation:
    def test_suffixes_are_the_cpp_unit_codes(self) -> None:
        assert {units: units_suffix(units) for units in Units} == {
            Units.PIXELS: "px",
            Units.MILLIMETERS: "mm",
            Units.CENTIMETERS: "cm",
            Units.INCHES: "in",
        }

    @pytest.mark.parametrize("units", [Units.PIXELS, Units.MILLIMETERS])
    def test_small_units_step_by_whole_numbers(self, units: Units) -> None:
        assert spin_box_precision(units) == (1, 1.0)

    @pytest.mark.parametrize("units", [Units.CENTIMETERS, Units.INCHES])
    def test_large_units_step_by_hundredths(self, units: Units) -> None:
        assert spin_box_precision(units) == (2, 0.01)
