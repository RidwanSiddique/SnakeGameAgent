/**
 * Port of snake/core/rng.py — xorshift128 seeded through splitmix32.
 *
 * Must produce byte-identical output to the Python original; `web/tests` pins it
 * against the frozen golden sequence. Every operation is forced back into
 * unsigned 32-bit range with `>>> 0`, because JavaScript's bitwise operators
 * produce *signed* 32-bit results and Math.imul is required for 32-bit
 * multiplication that wraps rather than losing precision to doubles.
 */

const TWO32 = 0x100000000;

function splitmix32(state: number): [number, number] {
  const z = (state + 0x9e3779b9) >>> 0;
  let t = z;
  t = Math.imul(t ^ (t >>> 16), 0x21f0aaad) >>> 0;
  t = Math.imul(t ^ (t >>> 15), 0x735a2d97) >>> 0;
  return [(t ^ (t >>> 15)) >>> 0, z];
}

export class Rng {
  private x: number;
  private y: number;
  private z: number;
  private w: number;

  constructor(seed = 0) {
    let state = seed >>> 0;
    const words: number[] = [];
    for (let i = 0; i < 4; i += 1) {
      const [value, nextState] = splitmix32(state);
      words.push(value);
      state = nextState;
    }

    if (words.every((word) => word === 0)) {
      words.splice(0, 4, 0x9e3779b9, 0x243f6a88, 0xb7e15162, 0x85a308d3);
    }

    [this.x, this.y, this.z, this.w] = words;
  }

  nextU32(): number {
    let t = this.x;
    t = (t ^ (t << 11)) >>> 0;
    t = (t ^ (t >>> 8)) >>> 0;
    this.x = this.y;
    this.y = this.z;
    this.z = this.w;
    const w = this.w;
    t = (t ^ w) >>> 0;
    t = (t ^ (w >>> 19)) >>> 0;
    this.w = t >>> 0;
    return this.w;
  }

  /**
   * Inclusive range, unbiased by rejection sampling.
   *
   * Modulo would bias the low end, and Python would have to reproduce the same
   * bias exactly for races to stay fair. Rejection is simpler to keep in sync.
   */
  randint(low: number, high: number): number {
    if (high < low) throw new Error(`empty range: [${low}, ${high}]`);
    const span = high - low + 1;
    if (span === 1) return low;

    const limit = Math.floor(TWO32 / span) * span;
    for (;;) {
      const value = this.nextU32();
      if (value < limit) return low + (value % span);
    }
  }

  clone(): Rng {
    const twin = new Rng(0);
    twin.x = this.x;
    twin.y = this.y;
    twin.z = this.z;
    twin.w = this.w;
    return twin;
  }
}
