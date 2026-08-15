'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { SnakeEngine } from '../lib/engine/engine.ts';
import { LEVEL_LIST } from '../lib/engine/levelData.ts';
import { Agent, loadAgent } from '../lib/agent/infer.ts';
import { useGameLoop } from '../lib/ui/useGameLoop.ts';
import { Board } from './Board.tsx';
import { Action, CLOCKWISE, Direction } from '../lib/engine/types.ts';
import { encodeActions } from '../lib/replay.ts';

const SPEED = 9; // steps per second, identical for both racers

type Phase = 'ready' | 'counting' | 'running' | 'over';

/** Turn an absolute key press into a relative action, or null if impossible. */
function actionFor(current: Direction, desired: Direction): Action | null {
  if (desired === current) return Action.STRAIGHT;
  const index = CLOCKWISE.indexOf(current);
  if (CLOCKWISE[(index + 1) % 4] === desired) return Action.RIGHT;
  if (CLOCKWISE[(index + 3) % 4] === desired) return Action.LEFT;
  return null; // a reversal: the snake cannot turn back through its own neck
}

const KEYS: Record<string, Direction> = {
  ArrowUp: Direction.UP,
  ArrowDown: Direction.DOWN,
  ArrowLeft: Direction.LEFT,
  ArrowRight: Direction.RIGHT,
  w: Direction.UP,
  s: Direction.DOWN,
  a: Direction.LEFT,
  d: Direction.RIGHT,
};

export function Race() {
  const [levelId, setLevelId] = useState(1);
  const [seed, setSeed] = useState(() => Math.floor(Math.random() * 1e9));
  const [phase, setPhase] = useState<Phase>('ready');
  const [countdown, setCountdown] = useState(3);
  const [version, setVersion] = useState(0);
  const [agent, setAgent] = useState<Agent | null>(null);
  const [result, setResult] = useState<{ human: number; agent: number; loser: string } | null>(null);

  const level = LEVEL_LIST.find((entry) => entry.id === levelId) ?? LEVEL_LIST[0];

  // Both engines take the same level and the same seed, so the food sequence is
  // identical for both racers. Without a shared deterministic RNG this would be
  // two different games shown side by side.
  const human = useMemo(() => new SnakeEngine(level, seed), [level, seed]);
  const machine = useMemo(() => new SnakeEngine(level, seed), [level, seed]);

  const pending = useRef<Direction | null>(null);
  // Every move the human makes, so the server can replay the run rather than
  // take the score on trust.
  const moves = useRef<number[]>([]);
  const [submission, setSubmission] = useState<'idle' | 'sending' | 'done' | string>('idle');
  const [player, setPlayer] = useState('');

  useEffect(() => {
    loadAgent().then(setAgent).catch(() => setAgent(null));
  }, []);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const direction = KEYS[event.key];
      if (!direction) return;
      event.preventDefault();
      pending.current = direction;
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  useEffect(() => {
    if (phase !== 'counting') return;
    if (countdown <= 0) {
      setPhase('running');
      return;
    }
    const timer = setTimeout(() => setCountdown((value) => value - 1), 700);
    return () => clearTimeout(timer);
  }, [phase, countdown]);

  const tick = useCallback(() => {
    const desired = pending.current;
    const humanAction = desired ? actionFor(human.direction, desired) ?? Action.STRAIGHT : Action.STRAIGHT;
    pending.current = null;

    moves.current.push(humanAction);
    const humanResult = human.step(humanAction);
    const agentResult = agent ? machine.step(agent.act(machine)) : { died: false };

    setVersion((value) => value + 1);

    if (humanResult.died || agentResult.died) {
      setPhase('over');
      setResult({
        human: human.score,
        agent: machine.score,
        loser: humanResult.died ? 'you' : 'agent',
      });
    }
  }, [agent, human, machine]);

  useGameLoop(tick, SPEED, phase === 'running');

  const start = () => {
    moves.current = [];
    setSubmission('idle');
    setResult(null);
    setCountdown(3);
    setPhase('counting');
  };

  const rematch = (newSeed = Math.floor(Math.random() * 1e9)) => {
    moves.current = [];
    setSubmission('idle');
    setSeed(newSeed);
    setResult(null);
    setPhase('ready');
  };

  const submit = async () => {
    setSubmission('sending');
    try {
      const response = await fetch('/api/scores', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          levelId,
          seed,
          moves: encodeActions(moves.current),
          score: result?.human ?? 0,
          player,
        }),
      });
      const body = await response.json();
      setSubmission(response.ok ? 'done' : (body.detail ?? body.error ?? 'Could not save that run.'));
    } catch {
      setSubmission('Could not reach the leaderboard.');
    }
  };

  const verdict = result
    ? result.human === result.agent
      ? 'Dead heat.'
      : result.human > result.agent
        ? 'You win.'
        : 'The agent wins.'
    : null;

  return (
    <div className="race">
      <div className="race-controls panel">
        <div className="field">
          <label className="stat-label" htmlFor="level">
            Board
          </label>
          <select
            id="level"
            value={levelId}
            disabled={phase === 'running' || phase === 'counting'}
            onChange={(event) => {
              setLevelId(Number(event.target.value));
              rematch(seed);
            }}
          >
            {LEVEL_LIST.map((entry) => (
              <option key={entry.id} value={entry.id}>
                {entry.name}
              </option>
            ))}
          </select>
        </div>

        <div className="field">
          <p className="stat-label">Shared seed</p>
          <p className="seed">{seed}</p>
        </div>

        {phase === 'ready' && (
          <button className="primary" onClick={start} disabled={!agent}>
            {agent ? 'Start race' : 'Loading agent…'}
          </button>
        )}
        {phase === 'over' && (
          <button className="primary" onClick={() => rematch()}>
            Race again
          </button>
        )}
        {(phase === 'running' || phase === 'counting') && (
          <p className="muted hint">Arrow keys or WASD</p>
        )}
      </div>

      {result && (
        <div className="verdict panel" role="status">
          <h2>{verdict}</h2>
          <p className="muted">
            {result.loser === 'you' ? 'You' : 'The agent'} crashed first — final score{' '}
            <span className="tag-human">{result.human}</span> to{' '}
            <span className="tag-agent">{result.agent}</span>. Same board, same food order.
          </p>

          {result.human > 0 && submission !== 'done' && (
            <form
              className="submit-row"
              onSubmit={(event) => {
                event.preventDefault();
                void submit();
              }}
            >
              <input
                value={player}
                onChange={(event) => setPlayer(event.target.value)}
                placeholder="Your name"
                maxLength={24}
                aria-label="Your name for the leaderboard"
              />
              <button type="submit" disabled={submission === 'sending'}>
                {submission === 'sending' ? 'Checking run…' : 'Post to leaderboard'}
              </button>
            </form>
          )}
          {submission === 'done' && (
            <p className="muted submit-note">Run verified and posted.</p>
          )}
          {typeof submission === 'string' && !['idle', 'sending', 'done'].includes(submission) && (
            <p className="submit-note submit-error">{submission}</p>
          )}
        </div>
      )}

      <div className="race-boards">
        <figure className="racer">
          <figcaption>
            <span className="stat-label">You</span>
            <span className="score tag-human">{human.score}</span>
          </figcaption>
          <div className="board-frame">
            <Board engine={human} version={version} accent="var(--human)" cell={15} />
            {phase === 'counting' && <div className="countdown">{countdown || 'Go'}</div>}
          </div>
        </figure>

        <figure className="racer">
          <figcaption>
            <span className="stat-label">Agent</span>
            <span className="score tag-agent">{machine.score}</span>
          </figcaption>
          <div className="board-frame">
            <Board engine={machine} version={version} accent="var(--agent)" cell={15} />
          </div>
        </figure>
      </div>
    </div>
  );
}
