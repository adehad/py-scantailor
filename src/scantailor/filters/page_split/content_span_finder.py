"""Split a column histogram into spans of content separated by whitespace."""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    """A half-open range ``[begin, end)`` of columns."""

    begin: int
    end: int

    @property
    def width(self) -> int:
        """Return the number of columns in the span."""
        return self.end - self.begin

    @property
    def center(self) -> float:
        """Return the middle of the span."""
        return 0.5 * (self.begin + self.end)

    @staticmethod
    def between(left: "Span | int", right: "Span | int") -> "Span":
        """Return the gap between two spans or positions.

        A span stands for its inner edge: its end when it is on the left, its
        begin when it is on the right.

        Args:
            left (Span | int): Span or column where the gap starts.
            right (Span | int): Span or column where the gap ends.

        Returns:
            Span: The range from the left argument's end to the right one's begin.
        """
        begin = left.end if isinstance(left, Span) else left
        end = right.begin if isinstance(right, Span) else right
        return Span(begin, end)


def find_content_spans(
    histogram: Sequence[int],
    min_content_width: int = 1,
    min_whitespace_width: int = 1,
) -> list[Span]:
    """Find the runs of non-empty columns, merging runs that are close together.

    A content run narrower than ``min_content_width`` is dropped unless
    whitespace shorter than ``min_whitespace_width`` joins it to its
    neighbours.

    Args:
        histogram (Sequence[int]): Amount of content in each column; 0 is empty.
        min_content_width (int): Narrowest span worth reporting.
        min_whitespace_width (int): Narrowest gap that separates two spans.

    Returns:
        list[Span]: Spans from left to right.
    """
    size = len(histogram)
    spans: list[Span] = []
    i = 0
    # Starting the previous content end a whitespace before column 0 lets the
    # first run open a span without a special case.
    content_end = -min_whitespace_width
    content_begin = content_end

    while True:
        while i < size and histogram[i] == 0:
            i += 1
        if i - content_end >= min_whitespace_width:
            if content_end - content_begin >= min_content_width:
                spans.append(Span(content_begin, content_end))
            content_begin = i
        if i == size:
            break
        while i < size and histogram[i] != 0:
            i += 1
        content_end = i

    if content_end - content_begin >= min_content_width:
        spans.append(Span(content_begin, content_end))
    return spans
