"""The random numbers Qt's QRandomGenerator(seed) gives, checked against libstdc++."""

from scantailor.dewarping.qt_random import QtRandomGenerator, seed_sequence_words


class TestSeedSequence:
    def test_words_match_the_standard_library_for_a_five_value_seed(self) -> None:
        """Values produced by std::seed_seq{1, 2, 3, 4, 5}.generate on libstdc++ 12."""
        words = seed_sequence_words([1, 2, 3, 4, 5], 5)

        assert [int(word) for word in words] == [
            0x00F8538C,
            0x177846AB,
            0xF50C8C25,
            0x2EFCD072,
            0xAF7E7B91,
        ]

    def test_a_longer_request_starts_with_other_words_than_a_shorter_one(self) -> None:
        short = seed_sequence_words([0], 5)
        long = seed_sequence_words([0], 624)

        assert len(long) == 624
        assert list(long[:5]) != list(short)


class TestGenerator:
    def test_first_outputs_for_seed_zero_match_std_mt19937_seeded_by_seed_seq(
        self,
    ) -> None:
        """Values from a std::mt19937 built from std::seed_seq{0} on libstdc++ 12."""
        generator = QtRandomGenerator(0)

        assert [generator.generate() for _ in range(3)] == [
            1529278401,
            1377812627,
            18815962,
        ]

    def test_the_same_seed_gives_the_same_sequence(self) -> None:
        first = QtRandomGenerator(7)
        second = QtRandomGenerator(7)

        assert [first.generate() for _ in range(20)] == [
            second.generate() for _ in range(20)
        ]

    def test_different_seeds_give_different_sequences(self) -> None:
        assert QtRandomGenerator(1).generate() != QtRandomGenerator(2).generate()
