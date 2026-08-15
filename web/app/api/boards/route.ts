import { NextResponse } from 'next/server';
import { DATABASE_ABSENT, cleanName, ensureSchema, getPool } from '../../../lib/db';
import { GRID } from '../../../lib/engine/levelData';
import { levelFromCells, resolveObstacles } from '../../../lib/engine/levels';
import { Rng } from '../../../lib/engine/rng';
import { decodeBoard } from '../../../lib/engine/share';

export const runtime = 'nodejs';

/** Published boards, newest first. */
export async function GET() {
  const pool = getPool();
  if (!pool) return NextResponse.json({ boards: [], note: DATABASE_ABSENT });

  await ensureSchema();
  const { rows } = await pool.query(
    `SELECT code, name, author, agent_score, walls, created_at
       FROM boards
      ORDER BY created_at DESC
      LIMIT 60`,
  );
  return NextResponse.json({ boards: rows });
}

/** Publish a board to the gallery. */
export async function POST(request: Request) {
  let body: Record<string, unknown>;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: 'expected a JSON body' }, { status: 400 });
  }

  const code = String(body.code ?? '');

  // Validate with the same rules the trainer uses, so the gallery cannot hold a
  // board that is unplayable or that walls off part of the grid.
  let walls = 0;
  try {
    const cells = decodeBoard(code, GRID);
    walls = cells.length;
    resolveObstacles(levelFromCells(cells, GRID), new Rng(0));
  } catch (cause) {
    return NextResponse.json(
      { error: 'That board is not playable.', detail: (cause as Error).message },
      { status: 422 },
    );
  }

  const pool = getPool();
  if (!pool) return NextResponse.json({ error: DATABASE_ABSENT }, { status: 503 });

  const agentScore =
    typeof body.agentScore === 'number' && Number.isFinite(body.agentScore)
      ? Math.max(0, Math.trunc(body.agentScore))
      : null;

  await ensureSchema();
  await pool.query(
    `INSERT INTO boards (code, name, author, agent_score, walls)
     VALUES ($1, $2, $3, $4, $5)
     ON CONFLICT (code) DO UPDATE
       SET agent_score = LEAST(boards.agent_score, EXCLUDED.agent_score)`,
    [code, cleanName(body.name, 'Untitled board'), cleanName(body.author), agentScore, walls],
  );

  return NextResponse.json({ ok: true }, { status: 201 });
}
