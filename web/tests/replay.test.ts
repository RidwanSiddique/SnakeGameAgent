/**
 * Replay verification is the only thing standing between the leaderboard and
 * anyone with a fetch call, so it is tested against forged claims as well as
 * honest ones.
 */
import test from 'node:test';
import assert from 'node:assert/strict';

import { SnakeEngine } from '../lib/engine/engine.ts';
import { LEVELS } from '../lib/engine/levelData.ts';
import { Rng } from '../lib/engine/rng.ts';
import { decodeActions, encodeActions, verifyRun } from '../lib/replay.ts';
import { Action } from '../lib/engine/types.ts';

/** Play a scripted run and return a claim describing it honestly. */
function honestRun(levelId: number, seed: number) {
  const engine = new SnakeEngine(LEVELS.get(levelId)!, seed);
  const rng = new Rng(seed ^ 0x5f5f);
  const actions: number[] = [];
  for (let i = 0; i < 400; i += 1) {
    const action = rng.randint(0, 2);
    actions.push(action);
    if (engine.step(action as Action).died) break;
  }
  return { levelId, seed, moves: encodeActions(actions), score: engine.score };
}

test('action encoding round trips', () => {
  const actions = Array.from({ length: 501 }, (_, i) => i % 3);
  assert.deepEqual(decodeActions(encodeActions(actions)), actions);
});

test('encoding is compact', () => {
  const actions = Array.from({ length: 4000 }, () => 1);
  assert.ok(encodeActions(actions).length < 1500, 'a long run should stay under ~1.5KB');
});

test('accepts an honest run', () => {
  for (const levelId of [1, 2, 3, 4]) {
    const claim = honestRun(levelId, 4242 + levelId);
    const verdict = verifyRun(claim);
    assert.ok(verdict.ok, `level ${levelId}: ${verdict.reason}`);
    assert.equal(verdict.score, claim.score);
  }
});

test('rejects an inflated score', () => {
  const claim = honestRun(1, 77);
  const verdict = verifyRun({ ...claim, score: claim.score + 50 });
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason!, /replay scored/);
});

test('rejects a claim with no moves but a high score', () => {
  const verdict = verifyRun({ levelId: 1, seed: 5, moves: encodeActions([]), score: 9999 });
  assert.equal(verdict.ok, false);
});

test('rejects a run replayed against a different seed', () => {
  // The run has to actually score for this to prove anything: a claim of zero
  // verifies under any seed, because zero is what the replay produces too. That
  // is fine — a forged score of zero gains nobody anything — but it makes a
  // zero-scoring run useless as a test of seed binding.
  let claim = honestRun(1, 1);
  for (let seed = 1; seed < 200 && claim.score === 0; seed += 1) {
    claim = honestRun(1, seed);
  }
  assert.ok(claim.score > 0, 'needed a scoring run to test seed binding');

  const verdict = verifyRun({ ...claim, seed: claim.seed + 1 });
  assert.equal(verdict.ok, false, 'the same moves on a different board should not reproduce the score');
});

test('rejects an unknown level', () => {
  const verdict = verifyRun({ levelId: 99, seed: 1, moves: encodeActions([0]), score: 0 });
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason!, /unknown level/);
});

test('rejects a malformed move record', () => {
  const verdict = verifyRun({ levelId: 1, seed: 1, moves: 'garbage', score: 0 });
  assert.equal(verdict.ok, false);
});

test('rejects an over-long move record rather than replaying it', () => {
  const moves = encodeActions(Array.from({ length: 25_000 }, () => 0));
  const verdict = verifyRun({ levelId: 1, seed: 1, moves, score: 0 });
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason!, /too long/);
});

test('verifies a run on a custom board', () => {
  const engine = new SnakeEngine(LEVELS.get(1)!, 900);
  const actions: number[] = [];
  for (let i = 0; i < 60; i += 1) {
    actions.push(0);
    if (engine.step(Action.STRAIGHT).died) break;
  }
  // An empty custom board behaves exactly like level 1.
  const verdict = verifyRun({
    board: 'A'.repeat(128),
    seed: 900,
    moves: encodeActions(actions),
    score: engine.score,
  });
  assert.ok(verdict.ok, verdict.reason);
});
