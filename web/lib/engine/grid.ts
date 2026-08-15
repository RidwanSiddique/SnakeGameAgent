/**
 * Port of snake/core/grid.py.
 *
 * The neighbour iteration order is load-bearing: `floodFill` returns as soon as
 * it reaches `limit`, so a different visit order produces a different truncated
 * set, and the free-space and tail-reachability features drift from Python's.
 */
import { DELTA, DELTA_ORDER, type Point, cellKey } from './types.ts';

export function neighbours(cell: Point, cols: number, rows: number): Point[] {
  const out: Point[] = [];
  for (const direction of DELTA_ORDER) {
    const [dx, dy] = DELTA[direction];
    const nx = cell.x + dx;
    const ny = cell.y + dy;
    if (nx >= 0 && nx < cols && ny >= 0 && ny < rows) out.push({ x: nx, y: ny });
  }
  return out;
}

/**
 * Cells reachable from `start` without entering `blocked`, capped at `limit`.
 *
 * The cap is checked on insertion rather than once per dequeue: a single
 * dequeue can add four cells, so a per-dequeue check overshoots and callers that
 * normalise by `limit` would emit values above 1.0.
 */
export function floodFill(
  start: Point,
  blocked: Set<number>,
  cols: number,
  rows: number,
  limit: number | null = null,
): Set<number> {
  const startKey = cellKey(start, cols);
  if (blocked.has(startKey) || start.x < 0 || start.x >= cols || start.y < 0 || start.y >= rows) {
    return new Set();
  }

  const seen = new Set<number>([startKey]);
  if (limit !== null && seen.size >= limit) return seen;

  const queue: Point[] = [start];
  let head = 0;
  while (head < queue.length) {
    const cell = queue[head];
    head += 1;
    for (const neighbour of neighbours(cell, cols, rows)) {
      const key = cellKey(neighbour, cols);
      if (seen.has(key) || blocked.has(key)) continue;
      seen.add(key);
      if (limit !== null && seen.size >= limit) return seen;
      queue.push(neighbour);
    }
  }
  return seen;
}
