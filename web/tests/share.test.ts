/** Board codes must survive a round trip exactly — a shared link is the level. */
import test from 'node:test';
import assert from 'node:assert/strict';

import { decodeBoard, encodeBoard } from '../lib/engine/share.ts';
import type { Point } from '../lib/engine/types.ts';

const GRID = { cols: 32, rows: 24 };
const key = (cell: Point) => `${cell.x},${cell.y}`;

test('round trips an empty board', () => {
  assert.deepEqual(decodeBoard(encodeBoard([], GRID), GRID), []);
});

test('round trips a scattered board', () => {
  const cells = [
    { x: 0, y: 0 }, { x: 31, y: 23 }, { x: 15, y: 12 }, { x: 1, y: 22 }, { x: 30, y: 2 },
  ];
  const got = decodeBoard(encodeBoard(cells, GRID), GRID);
  assert.deepEqual(new Set(got.map(key)), new Set(cells.map(key)));
});

test('round trips a full board', () => {
  const cells: Point[] = [];
  for (let y = 0; y < GRID.rows; y += 1) for (let x = 0; x < GRID.cols; x += 1) cells.push({ x, y });
  assert.equal(decodeBoard(encodeBoard(cells, GRID), GRID).length, GRID.cols * GRID.rows);
});

test('code stays short enough to paste', () => {
  const cells: Point[] = [];
  for (let y = 0; y < GRID.rows; y += 1) for (let x = 0; x < GRID.cols; x += 1) cells.push({ x, y });
  assert.ok(encodeBoard(cells, GRID).length <= 128, 'board code should fit in 128 characters');
});

test('rejects a malformed code', () => {
  assert.throws(() => decodeBoard('!!!not valid!!!', GRID), /invalid characters/);
});

test('rejects a truncated code', () => {
  assert.throws(() => decodeBoard('AAAA', GRID), /too short/);
});
