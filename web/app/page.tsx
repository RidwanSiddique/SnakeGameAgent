import Link from 'next/link';
import { Masthead } from './Masthead';
import { AgentDemo } from '../components/AgentDemo';
import { LEVEL_LIST } from '../lib/engine/levelData';

export default function Home() {
  return (
    <main className="shell">
      <Masthead />

      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Deep Q-learning · 25 inputs · 39 KB</p>
          <h1>
            It learned to play
            <br />
            by dying first.
          </h1>
          <p className="lede">
            This snake is driven by a neural network trained from nothing but a reward for eating
            and a penalty for dying. It runs entirely in your browser — no server decides its moves.
            Race it, or build a level it cannot finish.
          </p>
          <div className="hero-actions">
            <Link href="/race" className="button primary-link">
              Race the agent
            </Link>
            <Link href="/design" className="button">
              Design a level
            </Link>
          </div>
        </div>

        <div className="hero-board">
          <AgentDemo />
        </div>
      </section>

      <section className="levels">
        <h2>The boards</h2>
        <p className="lede">
          One network plays all four. Nothing in what it sees identifies the level, so it had to
          learn obstacles as a general skill rather than memorise a map — which is the only reason
          it stands a chance on a board you draw yourself.
        </p>
        <ol className="level-list">
          {LEVEL_LIST.map((level) => (
            <li key={level.id} className="panel level-card">
              <p className="stat-label">Level {level.id}</p>
              <h3>{level.name}</h3>
              <p className="muted level-note">
                {level.obstacles.kind === 'procedural'
                  ? 'Redrawn from the seed every game, so it is never the same board twice.'
                  : 'A fixed layout, identical on every attempt.'}
              </p>
            </li>
          ))}
        </ol>
      </section>
    </main>
  );
}
