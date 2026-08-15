/**
 * Port of snake/core/state.py — the 25-feature encoding.
 *
 * Feature order and arithmetic must match Python exactly; the golden trajectory
 * suite compares every value of every frame. See the Python module for what each
 * index means and why the free-space and path features exist.
 */
import { floodFill, neighbours } from './grid.ts';
import type { SnakeEngine } from './engine.ts';
import {
  CLOCKWISE,
  Direction,
  type Point,
  cellKey,
  samePoint,
  step as stepCell,
} from './types.ts';

export const FEATURE_VERSION = 3;
export const FEATURE_COUNT = 25;
export const LOOKAHEAD_STEPS = 3;

function relativeDirections(direction: Direction): [Direction, Direction, Direction] {
  const index = CLOCKWISE.indexOf(direction);
  return [direction, CLOCKWISE[(index + 1) % 4], CLOCKWISE[(index + 3) % 4]];
}

function pathBlocked(engine: SnakeEngine, direction: Direction, steps = LOOKAHEAD_STEPS): boolean {
  for (let distance = 1; distance <= steps; distance += 1) {
    if (engine.isCollision(stepCell(engine.head, direction, distance))) return true;
  }
  return false;
}

function distanceToObstruction(engine: SnakeEngine, direction: Direction, cap: number): number {
  for (let distance = 1; distance <= cap; distance += 1) {
    if (engine.isCollision(stepCell(engine.head, direction, distance))) return (distance - 1) / cap;
  }
  return 1.0;
}

function freeSpace(
  engine: SnakeEngine,
  direction: Direction,
  blocked: Set<number>,
  budget: number,
): number {
  const reachable = floodFill(
    stepCell(engine.head, direction),
    blocked,
    engine.grid.cols,
    engine.grid.rows,
    budget,
  );
  return reachable.size / budget;
}

function tailReachable(engine: SnakeEngine, direction: Direction, blocked: Set<number>): boolean {
  if (engine.snake.length < 2) return true;
  const tail = engine.snake[engine.snake.length - 1];
  const start = stepCell(engine.head, direction);
  if (samePoint(start, tail)) return true;

  const budget = 2 * engine.snake.length + 8;
  const without = new Set(blocked);
  without.delete(cellKey(tail, engine.grid.cols));

  const reached = floodFill(start, without, engine.grid.cols, engine.grid.rows, budget);
  return reached.has(cellKey(tail, engine.grid.cols)) || reached.size >= budget;
}

/** Shortest route from head to food: [first step, length], or [null, null]. */
export function computePath(
  engine: SnakeEngine,
  blocked: Set<number>,
): [Direction | null, number | null] {
  const food = engine.food;
  if (!food) return [null, null];

  const { cols, rows } = engine.grid;
  const head = engine.head;

  const origin = new Map<number, Direction>();
  const distance = new Map<number, number>();
  const queue: Point[] = [];

  for (const direction of CLOCKWISE) {
    const cell = stepCell(head, direction);
    if (cell.x < 0 || cell.x >= cols || cell.y < 0 || cell.y >= rows) continue;
    const key = cellKey(cell, cols);
    if (blocked.has(key)) continue;
    if (samePoint(cell, food)) return [direction, 1];
    origin.set(key, direction);
    distance.set(key, 1);
    queue.push(cell);
  }

  let index = 0;
  while (index < queue.length) {
    const cell = queue[index];
    index += 1;
    const cellDistance = distance.get(cellKey(cell, cols))!;
    const cellOrigin = origin.get(cellKey(cell, cols))!;

    for (const neighbour of neighbours(cell, cols, rows)) {
      const key = cellKey(neighbour, cols);
      if (origin.has(key) || blocked.has(key) || samePoint(neighbour, head)) continue;
      if (samePoint(neighbour, food)) return [cellOrigin, cellDistance + 1];
      origin.set(key, cellOrigin);
      distance.set(key, cellDistance + 1);
      queue.push(neighbour);
    }
  }

  return [null, null];
}

export function blockedCells(engine: SnakeEngine): Set<number> {
  const blocked = new Set(engine.obstacles);
  for (let i = 0; i < engine.snake.length - 1; i += 1) {
    blocked.add(cellKey(engine.snake[i], engine.grid.cols));
  }
  return blocked;
}

export function pathToFood(engine: SnakeEngine): [Direction | null, number | null] {
  return computePath(engine, blockedCells(engine));
}

export function getState(engine: SnakeEngine): Float32Array {
  const head = engine.head;
  const [straight, right, left] = relativeDirections(engine.direction);
  const blocked = blockedCells(engine);

  const budget = Math.max(4, engine.snake.length);
  const rayCap = Math.max(engine.grid.cols, engine.grid.rows);
  const food = engine.food ?? head;
  const [pathDirection] = computePath(engine, blocked);

  const features = [
    engine.isCollision(stepCell(head, straight)),
    engine.isCollision(stepCell(head, right)),
    engine.isCollision(stepCell(head, left)),

    pathBlocked(engine, straight),
    pathBlocked(engine, right),
    pathBlocked(engine, left),

    engine.direction === Direction.LEFT,
    engine.direction === Direction.RIGHT,
    engine.direction === Direction.UP,
    engine.direction === Direction.DOWN,

    food.x < head.x,
    food.x > head.x,
    food.y < head.y,
    food.y > head.y,

    freeSpace(engine, straight, blocked, budget),
    freeSpace(engine, right, blocked, budget),
    freeSpace(engine, left, blocked, budget),

    tailReachable(engine, straight, blocked),

    distanceToObstruction(engine, straight, rayCap),
    distanceToObstruction(engine, right, rayCap),
    distanceToObstruction(engine, left, rayCap),

    pathDirection === straight,
    pathDirection === right,
    pathDirection === left,
    pathDirection !== null,
  ];

  return Float32Array.from(features, (value) => (typeof value === 'boolean' ? (value ? 1 : 0) : value));
}
