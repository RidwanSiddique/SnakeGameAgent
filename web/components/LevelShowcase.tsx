'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { SnakeEngine } from '../lib/engine/engine.ts';
import { LEVEL_LIST } from '../lib/engine/levelData.ts';
import type { Level } from '../lib/engine/levels.ts';
import { Agent, loadAgent } from '../lib/agent/infer.ts';
import { useGameLoop } from '../lib/ui/useGameLoop.ts';
import { usePrefersReducedMotion } from '../lib/ui/usePrefersReducedMotion.ts';
import { Board } from './Board.tsx';

const CELL = 7;
const SPEED = 11; // steps per second per board

const NOTES: Record<number, string> = {
  1: 'Nothing in the way. The baseline the agent scores highest on.',
  2: 'Fixed blocks it has to steer around without boxing itself in.',
  3: 'Staggered walls. Reaching the food means committing to a route.',
  4: 'Redrawn from the seed every game, so it is never the same board twice.',
};

/** One card: a level, with the agent playing it. */
function LevelCard({ level, agent, animate }: { level: Level; agent: Agent | null; animate: boolean }) {
  const [version, setVersion] = useState(0);
  const [best, setBest] = useState(0);
  const seed = useRef(Math.floor(Math.random() * 1e9));

  // Level 4 is procedural, so a fresh seed each round also redraws its layout —
  // which is the point of the level and worth showing.
  const engine = useMemo(() => new SnakeEngine(level, seed.current), [level]);

  const tick = useCallback(() => {
    if (!agent) return;
    if (engine.step(agent.act(engine)).died) {
      setBest((previous) => Math.max(previous, engine.score));
      seed.current += 1;
      engine.reset(seed.current);
    }
    setVersion((value) => value + 1);
  }, [agent, engine]);

  useGameLoop(tick, SPEED, animate && agent !== null);

  return (
    <li className="panel level-card">
      <div className="level-card-board">
        <Board engine={engine} version={version} accent="var(--agent)" cell={CELL} />
      </div>

      <div className="level-card-head">
        <p className="stat-label">Level {level.id}</p>
        <p className="level-card-score tag-agent">{engine.score}</p>
      </div>
      <h3>{level.name}</h3>
      <p className="muted level-note">{NOTES[level.id] ?? ''}</p>
      <p className="level-card-best">
        {animate ? `best this session ${best}` : 'paused — reduced motion'}
      </p>
    </li>
  );
}

export function LevelShowcase() {
  const [agent, setAgent] = useState<Agent | null>(null);
  const reducedMotion = usePrefersReducedMotion();

  useEffect(() => {
    loadAgent()
      .then(setAgent)
      .catch(() => setAgent(null));
  }, []);

  return (
    <ol className="level-list">
      {LEVEL_LIST.map((level) => (
        <LevelCard key={level.id} level={level} agent={agent} animate={!reducedMotion} />
      ))}
    </ol>
  );
}
