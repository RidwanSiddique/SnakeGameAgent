"""Tests for the shared PRNG.

This generator is half of the parity contract: the TypeScript port must produce
byte-identical sequences, so these tests pin down behaviour that a JS
implementation could plausibly get wrong (32-bit wrapping, modulo bias, seeding).
"""
import pytest

from snake.core.rng import Rng


def test_same_seed_gives_same_sequence():
    a = Rng(12345)
    b = Rng(12345)
    assert [a.next_u32() for _ in range(50)] == [b.next_u32() for _ in range(50)]


def test_different_seeds_diverge():
    rng_a, rng_b = Rng(1), Rng(2)
    a = [rng_a.next_u32() for _ in range(10)]
    b = [rng_b.next_u32() for _ in range(10)]
    assert a != b


def test_successive_draws_differ():
    """A fixed point in the state would make every draw identical."""
    rng = Rng(1)
    assert len({rng.next_u32() for _ in range(1000)}) > 900


def test_output_stays_in_uint32_range():
    rng = Rng(99)
    for _ in range(500):
        value = rng.next_u32()
        assert 0 <= value < 2**32


def test_seed_zero_does_not_collapse_to_constant():
    """A naive xorshift seeded with all-zero state emits zeros forever."""
    rng = Rng(0)
    values = [rng.next_u32() for _ in range(20)]
    assert len(set(values)) > 1


def test_randint_respects_inclusive_bounds():
    rng = Rng(7)
    for _ in range(500):
        assert 3 <= rng.randint(3, 9) <= 9


def test_randint_reaches_both_endpoints():
    rng = Rng(7)
    seen = {rng.randint(0, 3) for _ in range(500)}
    assert seen == {0, 1, 2, 3}


def test_randint_single_value_range():
    rng = Rng(7)
    assert rng.randint(5, 5) == 5


def test_randint_rejects_inverted_range():
    rng = Rng(7)
    with pytest.raises(ValueError):
        rng.randint(9, 3)


def test_randint_is_roughly_uniform():
    """Guards against modulo bias, which is the easy way to get this wrong."""
    rng = Rng(2024)
    counts = [0] * 6
    trials = 60_000
    for _ in range(trials):
        counts[rng.randint(0, 5)] += 1
    expected = trials / 6
    for count in counts:
        assert abs(count - expected) < expected * 0.05


def test_clone_preserves_position():
    rng = Rng(42)
    [rng.next_u32() for _ in range(10)]
    twin = rng.clone()
    assert [rng.next_u32() for _ in range(10)] == [twin.next_u32() for _ in range(10)]


def test_golden_sequence_is_frozen():
    """Frozen output. The TypeScript port must reproduce these exact values.

    If this test fails after a change to rng.py, the change broke the parity
    contract and every recorded golden trajectory is invalidated.
    """
    rng = Rng(1)
    assert [rng.next_u32() for _ in range(8)] == [
        2233660604,
        3039944688,
        311919074,
        3056116658,
        607987423,
        533246967,
        2986260861,
        1111009731,
    ]
