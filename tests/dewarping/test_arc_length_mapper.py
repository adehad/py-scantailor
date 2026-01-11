"""Tests for arc length mapper."""

import numpy as np
import pytest

from scantailor.dewarping import ArcLengthMapper


class TestArcLengthMapper:
    """Tests for ArcLengthMapper class."""

    def test_empty_mapper(self):
        """Empty mapper returns zero."""
        mapper = ArcLengthMapper()
        assert mapper.total_arc_length() == 0.0
        assert mapper.arc_len_to_x(0.5) == 0.0
        assert mapper.x_to_arc_len(0.5) == 0.0
        assert mapper.num_samples == 0

    def test_single_sample(self):
        """Single sample returns that sample's values."""
        mapper = ArcLengthMapper()
        mapper.add_sample(5.0, 0.0)

        assert mapper.num_samples == 1
        assert mapper.total_arc_length() == 0.0
        assert mapper.arc_len_to_x(0.0) == 5.0
        assert mapper.x_to_arc_len(5.0) == 0.0

    def test_two_samples_horizontal(self):
        """Two samples with constant f(x)."""
        mapper = ArcLengthMapper()
        mapper.add_sample(0.0, 0.0)
        mapper.add_sample(10.0, 0.0)

        # Arc length equals x distance when f(x) is constant
        assert mapper.total_arc_length() == pytest.approx(10.0)
        assert mapper.arc_len_to_x(5.0) == pytest.approx(5.0)
        assert mapper.x_to_arc_len(5.0) == pytest.approx(5.0)

    def test_two_samples_diagonal(self):
        """Two samples on a diagonal."""
        mapper = ArcLengthMapper()
        mapper.add_sample(0.0, 0.0)
        mapper.add_sample(3.0, 4.0)

        # Arc length is sqrt(3^2 + 4^2) = 5
        assert mapper.total_arc_length() == pytest.approx(5.0)

    def test_normalize_range(self):
        """Normalizing arc length range."""
        mapper = ArcLengthMapper()
        mapper.add_sample(0.0, 0.0)
        mapper.add_sample(10.0, 0.0)
        mapper.normalize_range(1.0)

        assert mapper.total_arc_length() == pytest.approx(1.0)
        assert mapper.arc_len_to_x(0.5) == pytest.approx(5.0)
        assert mapper.x_to_arc_len(5.0) == pytest.approx(0.5)

    def test_multiple_samples(self):
        """Multiple samples with varying f(x)."""
        mapper = ArcLengthMapper()
        # Create a simple zigzag
        mapper.add_sample(0.0, 0.0)
        mapper.add_sample(1.0, 1.0)
        mapper.add_sample(2.0, 0.0)
        mapper.add_sample(3.0, 1.0)

        assert mapper.num_samples == 4
        # Each segment has length sqrt(2)
        expected_total = 3 * np.sqrt(2)
        assert mapper.total_arc_length() == pytest.approx(expected_total)

    def test_extrapolation_beyond_bounds(self):
        """Extrapolation works beyond sample bounds."""
        mapper = ArcLengthMapper()
        mapper.add_sample(0.0, 0.0)
        mapper.add_sample(10.0, 0.0)

        # Beyond bounds should extrapolate linearly
        assert mapper.arc_len_to_x(-5.0) == pytest.approx(-5.0)
        assert mapper.arc_len_to_x(15.0) == pytest.approx(15.0)
        assert mapper.x_to_arc_len(-5.0) == pytest.approx(-5.0)
        assert mapper.x_to_arc_len(15.0) == pytest.approx(15.0)

    def test_interpolation_midpoints(self):
        """Interpolation at segment midpoints."""
        mapper = ArcLengthMapper()
        mapper.add_sample(0.0, 0.0)
        mapper.add_sample(10.0, 0.0)
        mapper.add_sample(20.0, 0.0)

        # Midpoint of first segment
        assert mapper.arc_len_to_x(5.0) == pytest.approx(5.0)
        # Midpoint of second segment
        assert mapper.arc_len_to_x(15.0) == pytest.approx(15.0)

    def test_samples_property(self):
        """Samples property returns copies."""
        mapper = ArcLengthMapper()
        mapper.add_sample(0.0, 0.0)
        mapper.add_sample(10.0, 5.0)

        x_arr, arclen_arr = mapper.samples
        assert len(x_arr) == 2
        assert len(arclen_arr) == 2
        assert x_arr[0] == pytest.approx(0.0)
        assert x_arr[1] == pytest.approx(10.0)


class TestArcLengthMapperRoundTrip:
    """Round-trip tests for arc length mapping."""

    def test_roundtrip_x_to_arclen_to_x(self):
        """x -> arc_len -> x preserves value."""
        mapper = ArcLengthMapper()
        mapper.add_sample(0.0, 0.0)
        mapper.add_sample(5.0, 3.0)
        mapper.add_sample(10.0, 0.0)

        for x in [0.0, 2.5, 5.0, 7.5, 10.0]:
            arc_len = mapper.x_to_arc_len(x)
            x_back = mapper.arc_len_to_x(arc_len)
            assert x_back == pytest.approx(x, abs=1e-10)

    def test_roundtrip_arclen_to_x_to_arclen(self):
        """arc_len -> x -> arc_len preserves value."""
        mapper = ArcLengthMapper()
        mapper.add_sample(0.0, 0.0)
        mapper.add_sample(5.0, 3.0)
        mapper.add_sample(10.0, 0.0)
        mapper.normalize_range(1.0)

        for arc_len in [0.0, 0.25, 0.5, 0.75, 1.0]:
            x = mapper.arc_len_to_x(arc_len)
            arc_len_back = mapper.x_to_arc_len(x)
            assert arc_len_back == pytest.approx(arc_len, abs=1e-10)
