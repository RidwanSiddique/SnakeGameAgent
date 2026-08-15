import Link from 'next/link';
import { Masthead } from '../Masthead';
import { DATABASE_ABSENT, ensureSchema, getPool } from '../../lib/db';

export const dynamic = 'force-dynamic';
export const metadata = { title: 'Gallery · Snake Agent' };

interface BoardRow {
  code: string;
  name: string;
  author: string;
  agent_score: number | null;
  walls: number;
}

async function boards(): Promise<{ rows: BoardRow[]; note?: string }> {
  const pool = getPool();
  if (!pool) return { rows: [], note: DATABASE_ABSENT };
  await ensureSchema();
  const { rows } = await pool.query<BoardRow>(
    `SELECT code, name, author, agent_score, walls
       FROM boards ORDER BY created_at DESC LIMIT 60`,
  );
  return { rows };
}

export default async function GalleryPage() {
  const { rows, note } = await boards();

  return (
    <main className="shell">
      <Masthead />
      <p className="eyebrow">Boards people built</p>
      <h1>Levels it has to solve cold.</h1>
      <p className="lede">
        None of these were in training. Each card shows the best the agent managed — a low number
        means somebody found a shape it does not understand.
      </p>

      {note && <p className="notice">{note}</p>}

      {rows.length === 0 && !note && (
        <p className="notice">
          The gallery is empty. <Link href="/design">Build the first board.</Link>
        </p>
      )}

      <ul className="gallery">
        {rows.map((board) => (
          <li key={board.code} className="panel gallery-card">
            <h2>{board.name}</h2>
            <p className="muted byline">by {board.author}</p>
            <dl className="readout">
              <div>
                <dt>Agent scored</dt>
                <dd className="tag-agent">{board.agent_score ?? '—'}</dd>
              </div>
              <div>
                <dt>Walls</dt>
                <dd>{board.walls}</dd>
              </div>
            </dl>
            <Link className="button" href={`/design?b=${board.code}`}>
              Open this board
            </Link>
          </li>
        ))}
      </ul>
    </main>
  );
}
