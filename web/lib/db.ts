import { Pool } from 'pg';

/**
 * Postgres access, with the database treated as optional.
 *
 * Vercel's Neon integration supplies POSTGRES_URL (or DATABASE_URL). When
 * neither is set — a fresh clone, a local run, a preview before the database is
 * attached — the site must still work: race and the designer are entirely
 * client-side and need nothing from here. So `getPool` returns null rather than
 * throwing, and callers degrade to an explanatory empty state instead of a 500.
 */

const CONNECTION_STRING = process.env.POSTGRES_URL ?? process.env.DATABASE_URL ?? '';

// Reused across hot reloads and warm lambda invocations; a new pool per request
// would exhaust Neon's free-tier connection budget quickly.
declare global {
  // eslint-disable-next-line no-var
  var __snakePool: Pool | null | undefined;
  // eslint-disable-next-line no-var
  var __snakeSchemaReady: Promise<void> | undefined;
}

export function getPool(): Pool | null {
  if (!CONNECTION_STRING) return null;
  if (global.__snakePool === undefined) {
    global.__snakePool = new Pool({
      connectionString: CONNECTION_STRING,
      max: 3,
      ssl: CONNECTION_STRING.includes('localhost') ? undefined : { rejectUnauthorized: false },
    });
  }
  return global.__snakePool;
}

export const DATABASE_ABSENT =
  'No database is attached, so published boards and scores are unavailable. ' +
  'Racing and the designer work without one.';

const SCHEMA = `
  CREATE TABLE IF NOT EXISTS scores (
    id          BIGSERIAL PRIMARY KEY,
    level_id    INTEGER,
    board       TEXT,
    player      TEXT NOT NULL,
    score       INTEGER NOT NULL CHECK (score >= 0),
    seed        BIGINT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
  );
  CREATE INDEX IF NOT EXISTS scores_level_rank ON scores (level_id, score DESC);

  CREATE TABLE IF NOT EXISTS boards (
    id           BIGSERIAL PRIMARY KEY,
    code         TEXT UNIQUE NOT NULL,
    name         TEXT NOT NULL,
    author       TEXT NOT NULL,
    agent_score  INTEGER,
    walls        INTEGER NOT NULL DEFAULT 0,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
  );
  CREATE INDEX IF NOT EXISTS boards_recent ON boards (created_at DESC);
`;

/** Create tables on first use. Idempotent, and awaited once per process. */
export function ensureSchema(): Promise<void> {
  const pool = getPool();
  if (!pool) return Promise.resolve();
  if (!global.__snakeSchemaReady) {
    global.__snakeSchemaReady = pool.query(SCHEMA).then(() => undefined);
  }
  return global.__snakeSchemaReady;
}

/** Trim and bound a user-supplied display name. */
export function cleanName(value: unknown, fallback = 'anonymous'): string {
  if (typeof value !== 'string') return fallback;
  const cleaned = value.replace(/\s+/g, ' ').trim().slice(0, 24);
  return cleaned || fallback;
}
