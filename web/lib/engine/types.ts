/**
 * Value types mirroring snake/core/types.py.
 *
 * Ordering is part of the contract, not a style choice. `DELTA_ORDER` and
 * `CLOCKWISE` are iterated during breadth-first searches whose results are
 * truncated by a cap, so a different order yields a different truncation and the
 * feature vectors drift from Python's. Do not reorder them.
 */

export interface Point {
  x: number;
  y: number;
}

/**
 * Const objects rather than `enum`: TypeScript enums emit runtime code, so they
 * are not erasable, and Node's type-stripping loader rejects them outright.
 * This form keeps `Direction.RIGHT` usage identical while staying erasable.
 */
export const Direction = {
  RIGHT: 'RIGHT',
  LEFT: 'LEFT',
  UP: 'UP',
  DOWN: 'DOWN',
} as const;
export type Direction = (typeof Direction)[keyof typeof Direction];

export const Action = {
  STRAIGHT: 0,
  RIGHT: 1,
  LEFT: 2,
} as const;
export type Action = (typeof Action)[keyof typeof Action];

/** Insertion order of Python's DELTA dict. Drives neighbour iteration. */
export const DELTA_ORDER: readonly Direction[] = [
  Direction.RIGHT,
  Direction.LEFT,
  Direction.UP,
  Direction.DOWN,
];

export const DELTA: Record<Direction, readonly [number, number]> = {
  [Direction.RIGHT]: [1, 0],
  [Direction.LEFT]: [-1, 0],
  [Direction.UP]: [0, -1],
  [Direction.DOWN]: [0, 1],
};

/** Clockwise turn order, matching Python's CLOCKWISE tuple. */
export const CLOCKWISE: readonly Direction[] = [
  Direction.RIGHT,
  Direction.DOWN,
  Direction.LEFT,
  Direction.UP,
];

export interface StepResult {
  ate: boolean;
  died: boolean;
  score: number;
  steps: number;
}

export interface GridSize {
  cols: number;
  rows: number;
}

export function step(cell: Point, direction: Direction, times = 1): Point {
  const [dx, dy] = DELTA[direction];
  return { x: cell.x + dx * times, y: cell.y + dy * times };
}

export function turn(direction: Direction, action: Action): Direction {
  const index = CLOCKWISE.indexOf(direction);
  if (action === Action.RIGHT) return CLOCKWISE[(index + 1) % 4];
  if (action === Action.LEFT) return CLOCKWISE[(index + 3) % 4];
  return CLOCKWISE[index];
}

/** Cells are keyed as integers so Sets compare by value, as Python tuples do. */
export function cellKey(cell: Point, cols: number): number {
  return cell.y * cols + cell.x;
}

export function keyToCell(key: number, cols: number): Point {
  return { x: key % cols, y: Math.floor(key / cols) };
}

export function samePoint(a: Point | null, b: Point | null): boolean {
  if (a === null || b === null) return a === b;
  return a.x === b.x && a.y === b.y;
}
