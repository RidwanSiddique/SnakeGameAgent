/**
 * Port of snake/core/engine.py.
 *
 * Same rules, same RNG consumption, same food-placement order. `web/tests`
 * replays trajectories recorded by Python and asserts every frame matches; if
 * this file drifts, that suite fails rather than the site quietly behaving
 * differently from the environment the agent was trained in.
 */
import { Rng } from './rng.ts';
import { type Level, resolveObstacles, spawnCells } from './levels.ts';
import {
  Action,
  type GridSize,
  type Point,
  type StepResult,
  Direction,
  cellKey,
  samePoint,
  step as stepCell,
  turn,
} from './types.ts';

export const STALL_STEPS_PER_SEGMENT = 100;

export class SnakeEngine {
  readonly level: Level;
  readonly grid: GridSize;

  snake!: Point[];
  obstacles!: Set<number>;
  direction!: Direction;
  food!: Point | null;
  score!: number;
  steps!: number;

  private body!: Set<number>;
  private rng!: Rng;
  private seedValue: number;
  private fixedObstacles: Set<number> | null;
  private stepsSinceFood!: number;

  constructor(level: Level, seed = 0) {
    this.level = level;
    this.grid = level.grid;
    this.seedValue = seed;

    // Fixed geometry cannot change between episodes, and resolving it also
    // validates it, so do that once rather than on every reset. Safe for
    // determinism because fixed levels draw nothing from the RNG.
    this.fixedObstacles =
      level.obstacles.kind === 'procedural' ? null : resolveObstacles(level, new Rng(0));

    this.reset(seed);
  }

  get head(): Point {
    return this.snake[0];
  }

  get seed(): number {
    return this.seedValue;
  }

  reset(seed?: number): void {
    if (seed !== undefined) this.seedValue = seed;
    this.rng = new Rng(this.seedValue);

    this.obstacles = this.fixedObstacles ?? resolveObstacles(this.level, this.rng);

    this.snake = spawnCells(this.grid).map((cell) => ({ ...cell }));
    this.body = new Set(this.snake.map((cell) => cellKey(cell, this.grid.cols)));
    this.direction = Direction.RIGHT;
    this.score = 0;
    this.steps = 0;
    this.stepsSinceFood = 0;
    this.food = null;
    this.placeFood();
  }

  isCollision(cell?: Point, ignoreTail = false): boolean {
    const target = cell ?? this.head;

    if (target.x < 0 || target.x >= this.grid.cols || target.y < 0 || target.y >= this.grid.rows) {
      return true;
    }
    const key = cellKey(target, this.grid.cols);
    if (this.obstacles.has(key)) return true;
    if (!this.body.has(key)) return false;

    if (samePoint(target, this.snake[0])) return false;
    return !(ignoreTail && samePoint(target, this.snake[this.snake.length - 1]));
  }

  step(action: Action): StepResult {
    this.steps += 1;
    this.stepsSinceFood += 1;

    this.direction = turn(this.direction, action);
    const newHead = stepCell(this.head, this.direction);

    if (this.isCollision(newHead, !samePoint(newHead, this.food))) {
      return { ate: false, died: true, score: this.score, steps: this.steps };
    }
    if (this.stepsSinceFood > STALL_STEPS_PER_SEGMENT * this.snake.length) {
      return { ate: false, died: true, score: this.score, steps: this.steps };
    }

    this.snake.unshift(newHead);
    this.body.add(cellKey(newHead, this.grid.cols));

    const ate = samePoint(newHead, this.food);
    if (ate) {
      this.score += 1;
      this.stepsSinceFood = 0;
      this.placeFood();
    } else {
      const tail = this.snake.pop()!;
      this.body.delete(cellKey(tail, this.grid.cols));
    }

    return { ate, died: false, score: this.score, steps: this.steps };
  }

  /**
   * Enumerate free cells row-major and make a single draw.
   *
   * Rejection sampling would be shorter, but the number of RNG draws would then
   * depend on how full the board is and Python would have to match that count
   * exactly. One draw over an ordered list keeps both sides trivially in sync —
   * the row-major order (y outer, x inner) is part of the contract.
   */
  private placeFood(): void {
    const free: Point[] = [];
    for (let y = 0; y < this.grid.rows; y += 1) {
      for (let x = 0; x < this.grid.cols; x += 1) {
        const key = y * this.grid.cols + x;
        if (!this.body.has(key) && !this.obstacles.has(key)) free.push({ x, y });
      }
    }

    if (free.length === 0) {
      this.food = null;
      return;
    }
    this.food = free[this.rng.randint(0, free.length - 1)];
  }

  snapshot() {
    return {
      head: [this.head.x, this.head.y],
      snake: this.snake.map((cell) => [cell.x, cell.y]),
      food: this.food ? [this.food.x, this.food.y] : null,
      direction: this.direction,
      score: this.score,
      steps: this.steps,
    };
  }
}
