import { Masthead } from '../Masthead';
import { LEVEL_LIST } from '../../lib/engine/levelData';
import { DATABASE_ABSENT, ensureSchema, getPool } from '../../lib/db';

export const dynamic = 'force-dynamic';
export const metadata = { title: 'Leaderboard · Snake Agent' };

interface Row {
  level_id: number;
  player: string;
  score: number;
}

async function topScores(): Promise<{ rows: Row[]; note?: string }> {
  const pool = getPool();
  if (!pool) return { rows: [], note: DATABASE_ABSENT };
  await ensureSchema();
  const { rows } = await pool.query<Row>(
    `SELECT DISTINCT ON (level_id, player) level_id, player, score
       FROM scores
      ORDER BY level_id, player, score DESC`,
  );
  return { rows };
}

export default async function LeaderboardPage() {
  const { rows, note } = await topScores();

  return (
    <main className="shell">
      <Masthead />
      <p className="eyebrow">Verified runs only</p>
      <h1>Who has beaten it.</h1>
      <p className="lede">
        Every score here was replayed on the server before it was accepted. Submitting a run means
        submitting the moves that produced it, so a number on its own buys nothing.
      </p>

      {note && <p className="notice">{note}</p>}

      <div className="leaderboards">
        {LEVEL_LIST.map((level) => {
          const forLevel = rows
            .filter((row) => row.level_id === level.id)
            .sort((a, b) => b.score - a.score)
            .slice(0, 10);

          return (
            <section key={level.id} className="panel board-scores">
              <h2>{level.name}</h2>
              {forLevel.length === 0 ? (
                <p className="muted empty">
                  Nobody has posted a verified run here yet. Race it and be first.
                </p>
              ) : (
                <table>
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Player</th>
                      <th style={{ textAlign: 'right' }}>Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {forLevel.map((row, index) => (
                      <tr key={`${row.player}-${index}`}>
                        <td className="muted">{index + 1}</td>
                        <td className="tag-human">{row.player}</td>
                        <td style={{ textAlign: 'right' }}>{row.score}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </section>
          );
        })}
      </div>
    </main>
  );
}
