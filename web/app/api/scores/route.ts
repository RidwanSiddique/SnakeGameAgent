import { NextResponse } from 'next/server';
import { DATABASE_ABSENT, cleanName, ensureSchema, getPool } from '../../../lib/db';
import { verifyRun } from '../../../lib/replay';

export const runtime = 'nodejs';

/** Top scores for a level. */
export async function GET(request: Request) {
  const pool = getPool();
  if (!pool) return NextResponse.json({ scores: [], note: DATABASE_ABSENT });

  const levelId = Number(new URL(request.url).searchParams.get('level') ?? 1);
  if (!Number.isInteger(levelId)) {
    return NextResponse.json({ error: 'level must be a whole number' }, { status: 400 });
  }

  await ensureSchema();
  const { rows } = await pool.query(
    `SELECT player, score, created_at
       FROM scores
      WHERE level_id = $1
      ORDER BY score DESC, created_at ASC
      LIMIT 25`,
    [levelId],
  );
  return NextResponse.json({ scores: rows });
}

/**
 * Submit a run.
 *
 * The score is never taken on trust: the client sends the seed and its move
 * sequence, and the run is replayed here before anything is written.
 */
export async function POST(request: Request) {
  let body: Record<string, unknown>;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: 'expected a JSON body' }, { status: 400 });
  }

  const claim = {
    levelId: typeof body.levelId === 'number' ? body.levelId : undefined,
    board: typeof body.board === 'string' ? body.board : undefined,
    seed: Number(body.seed),
    moves: String(body.moves ?? ''),
    score: Number(body.score),
  };

  // Verify before touching storage: whether a run is genuine does not depend on
  // whether a database happens to be attached, and an invalid submission should
  // get the same answer either way.
  const verdict = verifyRun(claim);
  if (!verdict.ok) {
    return NextResponse.json(
      { error: 'That run does not check out.', detail: verdict.reason },
      { status: 422 },
    );
  }

  const pool = getPool();
  if (!pool) return NextResponse.json({ error: DATABASE_ABSENT }, { status: 503 });

  await ensureSchema();
  await pool.query(
    `INSERT INTO scores (level_id, board, player, score, seed) VALUES ($1, $2, $3, $4, $5)`,
    [claim.levelId ?? null, claim.board ?? null, cleanName(body.player), verdict.score, claim.seed],
  );

  return NextResponse.json({ ok: true, score: verdict.score }, { status: 201 });
}
