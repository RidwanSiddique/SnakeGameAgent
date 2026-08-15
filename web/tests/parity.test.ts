/**
 * Golden-trajectory parity: the TypeScript engine must reproduce Python exactly.
 *
 * Python recorded these episodes (shared/golden/trajectories.json) by playing
 * fixed levels with fixed seeds and a scripted action sequence, capturing every
 * frame and every feature value. This suite replays the same inputs and compares
 * frame by frame.
 *
 * A failure here means the trained agent and the deployed agent are playing
 * different games — which is exactly the bug that would otherwise surface as
 * "the model plays worse on the website" with no obvious cause.
 *
 * Run: node --test web/tests/
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import test from 'node:test';
import assert from 'node:assert/strict';

import { SnakeEngine } from '../lib/engine/engine.ts';
import { loadLevels } from '../lib/engine/levels.ts';
import { FEATURE_COUNT, FEATURE_VERSION, getState } from '../lib/engine/state.ts';
import { Rng } from '../lib/engine/rng.ts';
import { Action } from '../lib/engine/types.ts';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(here, '..', '..');

const levelFile = JSON.parse(readFileSync(join(repoRoot, 'shared', 'levels.json'), 'utf8'));
const golden = JSON.parse(readFileSync(join(repoRoot, 'shared', 'golden', 'trajectories.json'), 'utf8'));

const levels = loadLevels(levelFile);

// Features are stored rounded to 6 decimal places, and float32 round-trips add a
// little more; this tolerance is far tighter than any real divergence would be.
const EPSILON = 1e-5;

test('golden set matches the current feature version', () => {
  assert.equal(golden.feature_version, FEATURE_VERSION);
  assert.equal(golden.feature_count, FEATURE_COUNT);
});

test('frozen RNG sequence reproduces', () => {
  const rng = new Rng(1);
  const got = Array.from({ length: 8 }, () => rng.nextU32());
  assert.deepEqual(got, [
    2233660604, 3039944688, 311919074, 3056116658, 607987423, 533246967, 2986260861, 1111009731,
  ]);
});

for (const episode of golden.episodes) {
  test(`level ${episode.level_id} (${episode.level_name}) seed ${episode.seed} replays identically`, () => {
    const level = levels.get(episode.level_id);
    assert.ok(level, `level ${episode.level_id} missing from levels.json`);

    const engine = new SnakeEngine(level!, episode.seed);

    // Obstacle layout, including procedurally generated boards, must match.
    const obstacles = [...engine.obstacles]
      .map((key) => [key % engine.grid.cols, Math.floor(key / engine.grid.cols)])
      .sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const expected = [...episode.obstacles].sort(
      (a: number[], b: number[]) => a[0] - b[0] || a[1] - b[1],
    );
    assert.deepEqual(obstacles, expected, 'obstacle layout diverged');

    const compareFrame = (frameIndex: number) => {
      const expectedFrame = episode.frames[frameIndex];
      assert.deepEqual(engine.snapshot().head, expectedFrame.head, `head at frame ${frameIndex}`);
      assert.deepEqual(engine.snapshot().snake, expectedFrame.snake, `snake at frame ${frameIndex}`);
      assert.deepEqual(engine.snapshot().food, expectedFrame.food, `food at frame ${frameIndex}`);
      assert.equal(engine.score, expectedFrame.score, `score at frame ${frameIndex}`);

      const state = getState(engine);
      assert.equal(state.length, FEATURE_COUNT);
      for (let i = 0; i < FEATURE_COUNT; i += 1) {
        assert.ok(
          Math.abs(state[i] - expectedFrame.state[i]) < EPSILON,
          `feature ${i} at frame ${frameIndex}: ts ${state[i]} vs py ${expectedFrame.state[i]}`,
        );
      }
    };

    compareFrame(0);

    episode.actions.forEach((actionValue: number, index: number) => {
      const result = engine.step(actionValue as Action);
      const expectedFrame = episode.frames[index + 1];
      assert.equal(result.died, expectedFrame.died, `died flag at frame ${index + 1}`);
      assert.equal(result.ate, expectedFrame.ate, `ate flag at frame ${index + 1}`);
      if (!result.died) compareFrame(index + 1);
    });
  });
}
