"""Deterministic PRNG shared by the Python trainer and the TypeScript client.

Python's `random` and JavaScript's `Math.random` cannot be made to agree, so the
game never uses either. Every random draw in the engine comes from this class,
which is specified tightly enough to reimplement exactly in TypeScript:

  - state is four 32-bit words, seeded through splitmix32
  - `next_u32` is Marsaglia's xorshift128
  - `randint` uses rejection sampling, not modulo, so both ports agree bit for bit

Keep this file free of imports. It is a specification as much as an
implementation, and `tests/test_rng.py::test_golden_sequence_is_frozen` pins the
output. Changing the algorithm invalidates every recorded golden trajectory.
"""

_U32 = 0xFFFFFFFF
_TWO32 = 0x100000000


def _splitmix32(z: int) -> tuple[int, int]:
    """Advance a splitmix32 state, returning (output, new_state).

    Used only for seeding: it turns a single user-supplied integer into four
    well-mixed words, so `Rng(0)` and `Rng(1)` start far apart.
    """
    z = (z + 0x9E3779B9) & _U32
    t = z
    t = ((t ^ (t >> 16)) * 0x21F0AAAD) & _U32
    t = ((t ^ (t >> 15)) * 0x735A2D97) & _U32
    return (t ^ (t >> 15)) & _U32, z


class Rng:
    """xorshift128, seeded from a single integer."""

    __slots__ = ("_x", "_y", "_z", "_w")

    def __init__(self, seed: int = 0):
        state = seed & _U32
        words = []
        for _ in range(4):
            value, state = _splitmix32(state)
            words.append(value)

        # An all-zero state is a fixed point of xorshift, so nudge it. splitmix32
        # makes this vanishingly unlikely, but the failure mode is silent and
        # total, so it is worth two lines to rule out.
        if not any(words):
            words = [0x9E3779B9, 0x243F6A88, 0xB7E15162, 0x85A308D3]

        self._x, self._y, self._z, self._w = words

    def next_u32(self) -> int:
        """Return the next pseudo-random 32-bit unsigned integer."""
        t = self._x
        t = (t ^ (t << 11)) & _U32
        t ^= t >> 8
        self._x, self._y, self._z = self._y, self._z, self._w
        w = self._w
        t ^= w
        t ^= w >> 19
        self._w = t & _U32
        return self._w

    def randint(self, low: int, high: int) -> int:
        """Return an integer in [low, high], inclusive and unbiased.

        Rejection sampling rather than `% span`: modulo would skew the low end of
        the range, and the skew would have to be reproduced exactly in the
        TypeScript port to keep races fair. Rejection is simpler to port.
        """
        if high < low:
            raise ValueError(f"empty range: [{low}, {high}]")

        span = high - low + 1
        if span == 1:
            return low

        limit = (_TWO32 // span) * span
        while True:
            value = self.next_u32()
            if value < limit:
                return low + (value % span)

    def choice(self, items):
        """Return a uniformly chosen element. Raises on an empty sequence."""
        if not items:
            raise ValueError("cannot choose from an empty sequence")
        return items[self.randint(0, len(items) - 1)]

    def clone(self) -> "Rng":
        """Return an independent generator positioned exactly here.

        Used to fork a food sequence for a replay without disturbing the original.
        """
        twin = Rng.__new__(Rng)
        twin._x, twin._y, twin._z, twin._w = self._x, self._y, self._z, self._w
        return twin
