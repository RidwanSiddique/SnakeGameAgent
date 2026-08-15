/**
 * Server-side replay verification.
 *
 * The game runs in the browser, so a submitted score is just a number a client
 * asserted — anyone can POST 9999. Rather than trust it or bolt on a token, the
 * client submits the seed and the exact sequence of moves it made, and the
 * server replays that sequence through the same deterministic engine. A score is
 * accepted only if replaying it reproduces the claim.
 *
 * This is possible only because the engine is deterministic and shared. It also
 * protects the free-tier write budget: a forged run costs the attacker a full
 * valid playthrough.
 */
import { SnakeEngine } from './engine/engine.ts';
import { LEVELS } from './engine/levelData.ts';
import { GRID } from './engine/levelData.ts';
import { levelFromCells, type Level } from './engine/levels.ts';
import { decodeBoard } from './engine/share.ts';
import { Action } from './engine/types.ts';

const ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_';

/** Actions are 0-2, so four fit in a byte. A 200-point run is ~1.5KB encoded. */
export function encodeActions(actions: number[]): string {
  const bytes = new Uint8Array(Math.ceil(actions.length / 4));
  actions.forEach((action, index) => {
    bytes[index >> 2] |= (action & 3) << ((index & 3) * 2);
  });

  let out = '';
  for (let i = 0; i < bytes.length; i += 3) {
    const chunk = (bytes[i] << 16) | ((bytes[i + 1] ?? 0) << 8) | (bytes[i + 2] ?? 0);
    out += ALPHABET[(chunk >> 18) & 63] + ALPHABET[(chunk >> 12) & 63];
    if (i + 1 < bytes.length) out += ALPHABET[(chunk >> 6) & 63];
    if (i + 2 < bytes.length) out += ALPHABET[chunk & 63];
  }
  return `${actions.length}.${out}`;
}

export function decodeActions(encoded: string): number[] {
  const separator = encoded.indexOf('.');
  if (separator < 0) throw new Error('malformed move record');

  const count = Number(encoded.slice(0, separator));
  if (!Number.isInteger(count) || count < 0) throw new Error('malformed move record');

  const text = encoded.slice(separator + 1);
  const values = [...text].map((character) => ALPHABET.indexOf(character));
  if (values.some((value) => value < 0)) throw new Error('malformed move record');

  const bytes: number[] = [];
  for (let i = 0; i < values.length; i += 4) {
    const chunk =
      (values[i] << 18) | ((values[i + 1] ?? 0) << 12) | ((values[i + 2] ?? 0) << 6) | (values[i + 3] ?? 0);
    bytes.push((chunk >> 16) & 255);
    if (i + 2 < values.length) bytes.push((chunk >> 8) & 255);
    if (i + 3 < values.length) bytes.push(chunk & 255);
  }

  const actions: number[] = [];
  for (let index = 0; index < count; index += 1) {
    actions.push((bytes[index >> 2] >> ((index & 3) * 2)) & 3);
  }
  return actions;
}

export interface RunClaim {
  levelId?: number;
  board?: string; // share code, for a custom board
  seed: number;
  moves: string; // encodeActions output
  score: number;
}

export interface Verdict {
  ok: boolean;
  score: number;
  reason?: string;
}

// A generous ceiling on replay work, so one request cannot pin a serverless
// function. 20k moves is far beyond any plausible human run.
const MAX_MOVES = 20_000;

export function verifyRun(claim: RunClaim): Verdict {
  let level: Level | undefined;

  if (claim.board) {
    try {
      level = levelFromCells(decodeBoard(claim.board, GRID), GRID, 0, 'Custom');
    } catch (cause) {
      return { ok: false, score: 0, reason: (cause as Error).message };
    }
  } else if (claim.levelId !== undefined) {
    level = LEVELS.get(claim.levelId);
  }

  if (!level) return { ok: false, score: 0, reason: 'unknown level' };
  if (!Number.isInteger(claim.seed)) return { ok: false, score: 0, reason: 'invalid seed' };

  let actions: number[];
  try {
    actions = decodeActions(claim.moves);
  } catch (cause) {
    return { ok: false, score: 0, reason: (cause as Error).message };
  }

  if (actions.length > MAX_MOVES) {
    return { ok: false, score: 0, reason: 'move record too long' };
  }

  let engine: SnakeEngine;
  try {
    engine = new SnakeEngine(level, claim.seed);
  } catch (cause) {
    return { ok: false, score: 0, reason: (cause as Error).message };
  }

  for (const action of actions) {
    if (engine.step(action as Action).died) break;
  }

  if (engine.score !== claim.score) {
    return {
      ok: false,
      score: engine.score,
      reason: `replay scored ${engine.score}, not ${claim.score}`,
    };
  }
  return { ok: true, score: engine.score };
}
