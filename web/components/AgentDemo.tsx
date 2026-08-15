'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { SnakeEngine } from '../lib/engine/engine.ts';
import { LEVEL_LIST } from '../lib/engine/levelData.ts';
import { Agent, loadAgent } from '../lib/agent/infer.ts';
import { useGameLoop } from '../lib/ui/useGameLoop.ts';
import { Board } from './Board.tsx';

/** The agent playing, unattended. The page opens on the thing it is about. */
export function AgentDemo({ cell = 18, speed = 14 }: { cell?: number; speed?: number }) {
  const [agent, setAgent] = useState<Agent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [version, setVersion] = useState(0);
  const [best, setBest] = useState(0);
  const [levelIndex, setLevelIndex] = useState(0);

  const engine = useMemo(
    () => new SnakeEngine(LEVEL_LIST[levelIndex], Math.floor(Math.random() * 1e9)),
    [levelIndex],
  );
  const seed = useRef(1);

  useEffect(() => {
    loadAgent()
      .then(setAgent)
      .catch((cause: Error) => setError(cause.message));
  }, []);

  const tick = useCallback(() => {
    if (!agent) return;
    if (engine.step(agent.act(engine)).died) {
      setBest((previous) => Math.max(previous, engine.score));
      seed.current += 1;
      engine.reset(seed.current);
      // Rotate levels on death so the demo shows the whole game, not one board.
      setLevelIndex((index) => (index + 1) % LEVEL_LIST.length);
    }
    setVersion((v) => v + 1);
  }, [agent, engine]);

  useGameLoop(tick, speed, agent !== null);

  if (error) {
    return (
      <div className="panel" style={{ padding: '1.5rem' }}>
        <p className="eyebrow">Agent unavailable</p>
        <p className="muted" style={{ margin: 0 }}>
          {error}. Run <code>python3 -m snake.export.to_web</code> to publish trained weights.
        </p>
      </div>
    );
  }

  return (
    <div>
      <Board engine={engine} version={version} accent="var(--agent)" cell={cell} />
      <dl className="readout" style={{ marginTop: '1rem' }}>
        <div>
          <dt>Level</dt>
          <dd>{LEVEL_LIST[levelIndex]?.name ?? '—'}</dd>
        </div>
        <div>
          <dt>Score</dt>
          <dd className="tag-agent">{engine.score}</dd>
        </div>
        <div>
          <dt>Best this session</dt>
          <dd>{best}</dd>
        </div>
        <div>
          <dt>Trained on</dt>
          <dd>{agent?.provenance ? `${agent.provenance.games.toLocaleString()} games` : '…'}</dd>
        </div>
      </dl>
    </div>
  );
}
