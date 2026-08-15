/**
 * Port of snake/core/levels.py.
 *
 * Reads the same shared/levels.json the trainer does, so level geometry has one
 * definition rather than two. The procedural generator consumes the RNG in
 * exactly the order Python does — block count, then per block width, height, x,
 * y — because a race on level 4 must produce the same board on both sides.
 */
import { floodFill } from './grid.ts';
import { type GridSize, type Point, cellKey } from './types.ts';

export const SPAWN_CLEARANCE_AHEAD = 5;
export const SPAWN_CLEARANCE_BEHIND = 3;

export class LevelError extends Error {}

export interface ObstacleSpec {
  kind: 'fixed' | 'procedural';
  blocks?: { x: number; y: number; w: number; h: number }[];
  cells?: [number, number][];
  block_count?: [number, number];
  block_size?: [number, number];
  coverage_limit?: number;
}

export interface Level {
  id: number;
  name: string;
  difficulty: number;
  grid: GridSize;
  obstacles: ObstacleSpec;
}

export interface LevelFile {
  schema_version: number;
  grid: GridSize;
  levels: Omit<Level, 'grid'>[];
}

export function loadLevels(data: LevelFile): Map<number, Level> {
  if (data.schema_version !== 1) {
    throw new LevelError(`unsupported schema_version: ${data.schema_version}`);
  }
  const levels = new Map<number, Level>();
  for (const entry of data.levels) {
    if (levels.has(entry.id)) throw new LevelError(`duplicate level id: ${entry.id}`);
    levels.set(entry.id, { ...entry, grid: data.grid });
  }
  return levels;
}

export function spawnCells(grid: GridSize): Point[] {
  const head = { x: Math.floor(grid.cols / 2), y: Math.floor(grid.rows / 2) };
  return [head, { x: head.x - 1, y: head.y }, { x: head.x - 2, y: head.y }];
}

export function spawnCorridor(grid: GridSize): Set<number> {
  const head = spawnCells(grid)[0];
  const corridor = new Set<number>();
  for (let offset = -SPAWN_CLEARANCE_BEHIND; offset <= SPAWN_CLEARANCE_AHEAD; offset += 1) {
    const cell = { x: head.x + offset, y: head.y };
    if (cell.x >= 0 && cell.x < grid.cols && cell.y >= 0 && cell.y < grid.rows) {
      corridor.add(cellKey(cell, grid.cols));
    }
  }
  return corridor;
}

function expandFixed(spec: ObstacleSpec, grid: GridSize): Set<number> {
  const cells = new Set<number>();
  for (const block of spec.blocks ?? []) {
    for (let dx = 0; dx < block.w; dx += 1) {
      for (let dy = 0; dy < block.h; dy += 1) {
        const cell = { x: block.x + dx, y: block.y + dy };
        if (cell.x < 0 || cell.x >= grid.cols || cell.y < 0 || cell.y >= grid.rows) {
          throw new LevelError(`obstacles outside the ${grid.cols}x${grid.rows} grid`);
        }
        cells.add(cellKey(cell, grid.cols));
      }
    }
  }
  for (const [x, y] of spec.cells ?? []) {
    if (x < 0 || x >= grid.cols || y < 0 || y >= grid.rows) {
      throw new LevelError(`obstacles outside the ${grid.cols}x${grid.rows} grid`);
    }
    cells.add(cellKey({ x, y }, grid.cols));
  }
  return cells;
}

function isFullyReachable(obstacles: Set<number>, grid: GridSize): boolean {
  const head = spawnCells(grid)[0];
  const reached = floodFill(head, obstacles, grid.cols, grid.rows);
  return reached.size === grid.cols * grid.rows - obstacles.size;
}

function generateProcedural(spec: ObstacleSpec, grid: GridSize, rng: { randint(a: number, b: number): number }): Set<number> {
  const [minBlocks, maxBlocks] = spec.block_count ?? [6, 12];
  const [minSize, maxSize] = spec.block_size ?? [1, 3];
  const coverageLimit = spec.coverage_limit ?? 0.15;

  const protectedCells = spawnCorridor(grid);
  const maxCells = Math.floor(grid.cols * grid.rows * coverageLimit);

  for (let attempt = 0; attempt < 30; attempt += 1) {
    const cells = new Set<number>();
    const blockCount = rng.randint(minBlocks, maxBlocks);
    for (let b = 0; b < blockCount; b += 1) {
      const w = rng.randint(minSize, maxSize);
      const h = rng.randint(minSize, maxSize);
      const x = rng.randint(0, Math.max(0, grid.cols - w));
      const y = rng.randint(0, Math.max(0, grid.rows - h));

      const candidate = new Set<number>();
      for (let dx = 0; dx < w; dx += 1) {
        for (let dy = 0; dy < h; dy += 1) {
          candidate.add(cellKey({ x: x + dx, y: y + dy }, grid.cols));
        }
      }

      let touchesSpawn = false;
      for (const key of candidate) if (protectedCells.has(key)) { touchesSpawn = true; break; }
      if (touchesSpawn) continue;

      const merged = new Set(cells);
      for (const key of candidate) merged.add(key);
      if (merged.size > maxCells) continue;

      for (const key of candidate) cells.add(key);
    }

    if (isFullyReachable(cells, grid)) return cells;
  }
  return new Set();
}

export function resolveObstacles(level: Level, rng: { randint(a: number, b: number): number }): Set<number> {
  if (level.obstacles.kind === 'procedural') {
    return generateProcedural(level.obstacles, level.grid, rng);
  }

  const cells = expandFixed(level.obstacles, level.grid);
  const corridor = spawnCorridor(level.grid);
  for (const key of cells) {
    if (corridor.has(key)) {
      throw new LevelError(`level ${level.id} (${level.name}) puts obstacles in the spawn corridor`);
    }
  }
  if (!isFullyReachable(cells, level.grid)) {
    throw new LevelError(`level ${level.id} (${level.name}) walls off part of the board`);
  }
  return cells;
}

export function levelFromCells(cells: Point[], grid: GridSize, id = 0, name = 'Custom'): Level {
  return {
    id,
    name,
    difficulty: 0,
    grid,
    obstacles: { kind: 'fixed', blocks: [], cells: cells.map((c) => [c.x, c.y] as [number, number]) },
  };
}
