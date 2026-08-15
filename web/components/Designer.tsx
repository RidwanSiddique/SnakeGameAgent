'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { SnakeEngine } from '../lib/engine/engine.ts';
import { GRID } from '../lib/engine/levelData.ts';
import { LevelError, levelFromCells, resolveObstacles, spawnCells, spawnCorridor } from '../lib/engine/levels.ts';
import { Rng } from '../lib/engine/rng.ts';
import { decodeBoard, encodeBoard } from '../lib/engine/share.ts';
import { Agent, loadAgent } from '../lib/agent/infer.ts';
import { useGameLoop } from '../lib/ui/useGameLoop.ts';
import { Board } from './Board.tsx';
import { cellKey, type Point } from '../lib/engine/types.ts';

const CELL = 20;

type Mode = 'edit' | 'run';

export function Designer() {
  const [cells, setCells] = useState<Set<number>>(new Set());
  const [mode, setMode] = useState<Mode>('edit');
  const [problem, setProblem] = useState<string | null>(null);
  const [agent, setAgent] = useState<Agent | null>(null);
  const [version, setVersion] = useState(0);
  const [outcome, setOutcome] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [bestAgentScore, setBestAgentScore] = useState<number | null>(null);
  const [publishState, setPublishState] = useState<'idle' | 'sending' | 'done' | string>('idle');
  const [boardName, setBoardName] = useState('');

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const painting = useRef<'add' | 'remove' | null>(null);

  const corridor = useMemo(() => spawnCorridor(GRID), []);
  const spawn = useMemo(() => new Set(spawnCells(GRID).map((c) => cellKey(c, GRID.cols))), []);

  useEffect(() => {
    loadAgent().then(setAgent).catch(() => setAgent(null));
  }, []);

  // Load a shared board from the URL. Read from location directly rather than
  // useSearchParams, which would force this whole page behind a Suspense
  // boundary during prerender for no benefit.
  useEffect(() => {
    const code = new URLSearchParams(window.location.search).get('b');
    if (!code) return;
    try {
      const decoded = decodeBoard(code, GRID);
      setCells(new Set(decoded.map((cell) => cellKey(cell, GRID.cols))));
    } catch (cause) {
      setProblem((cause as Error).message);
    }
  }, []);

  const width = GRID.cols * CELL;
  const height = GRID.rows * CELL;

  const paint = useCallback(
    (event: React.PointerEvent<HTMLCanvasElement>) => {
      if (mode !== 'edit' || !painting.current) return;
      const canvas = canvasRef.current;
      if (!canvas) return;

      const bounds = canvas.getBoundingClientRect();
      const x = Math.floor(((event.clientX - bounds.left) / bounds.width) * GRID.cols);
      const y = Math.floor(((event.clientY - bounds.top) / bounds.height) * GRID.rows);
      if (x < 0 || x >= GRID.cols || y < 0 || y >= GRID.rows) return;

      const key = cellKey({ x, y }, GRID.cols);
      if (corridor.has(key)) {
        setProblem('The snake starts on those cells, so they have to stay clear.');
        return;
      }

      setCells((previous) => {
        const next = new Set(previous);
        if (painting.current === 'add') next.add(key);
        else next.delete(key);
        return next;
      });
      setProblem(null);
      setOutcome(null);
    },
    [corridor, mode],
  );

  // Draw the editing surface.
  useEffect(() => {
    if (mode !== 'edit') return;
    const canvas = canvasRef.current;
    const context = canvas?.getContext('2d');
    if (!canvas || !context) return;

    const ratio = window.devicePixelRatio || 1;
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);

    context.fillStyle = '#0d1017';
    context.fillRect(0, 0, width, height);

    context.strokeStyle = '#161d29';
    context.beginPath();
    for (let x = 0; x <= GRID.cols; x += 1) {
      context.moveTo(x * CELL + 0.5, 0);
      context.lineTo(x * CELL + 0.5, height);
    }
    for (let y = 0; y <= GRID.rows; y += 1) {
      context.moveTo(0, y * CELL + 0.5);
      context.lineTo(width, y * CELL + 0.5);
    }
    context.stroke();

    // The protected start corridor, shown so the rule is visible rather than
    // only enforced when someone tries to draw there.
    context.fillStyle = 'rgba(245, 165, 36, 0.09)';
    for (const key of corridor) {
      context.fillRect((key % GRID.cols) * CELL, Math.floor(key / GRID.cols) * CELL, CELL, CELL);
    }
    context.fillStyle = 'rgba(245, 165, 36, 0.45)';
    for (const key of spawn) {
      context.fillRect((key % GRID.cols) * CELL + 5, Math.floor(key / GRID.cols) * CELL + 5, CELL - 10, CELL - 10);
    }

    context.fillStyle = '#414859';
    for (const key of cells) {
      context.fillRect((key % GRID.cols) * CELL + 1, Math.floor(key / GRID.cols) * CELL + 1, CELL - 2, CELL - 2);
    }
  }, [cells, corridor, spawn, mode, width, height]);

  const customLevel = useMemo(
    () =>
      levelFromCells(
        [...cells].map((key) => ({ x: key % GRID.cols, y: Math.floor(key / GRID.cols) })),
        GRID,
        0,
        'Your board',
      ),
    [cells],
  );

  const engine = useMemo(() => {
    if (mode !== 'run') return null;
    try {
      return new SnakeEngine(customLevel, Math.floor(Math.random() * 1e9));
    } catch {
      return null;
    }
  }, [customLevel, mode]);

  const tick = useCallback(() => {
    if (!engine || !agent) return;
    if (engine.step(agent.act(engine)).died) {
      setOutcome(`The agent scored ${engine.score} before it crashed.`);
      setBestAgentScore((previous) =>
        previous === null ? engine.score : Math.min(previous, engine.score),
      );
      setMode('edit');
      return;
    }
    setVersion((value) => value + 1);
  }, [engine, agent]);

  useGameLoop(tick, 13, mode === 'run' && engine !== null && agent !== null);

  const run = () => {
    // The same validation the trainer applies, so a board that runs here is a
    // board the agent could have been trained on.
    try {
      resolveObstacles(customLevel, new Rng(0));
    } catch (cause) {
      setProblem(
        cause instanceof LevelError && cause.message.includes('walls off')
          ? 'Part of the board is sealed off. Every open cell has to be reachable, or the food could appear somewhere the snake can never go.'
          : (cause as Error).message,
      );
      return;
    }
    setProblem(null);
    setOutcome(null);
    setMode('run');
  };

  const share = async () => {
    const url = `${window.location.origin}/design?b=${encodeBoard(
      [...cells].map((key) => ({ x: key % GRID.cols, y: Math.floor(key / GRID.cols) }) as Point),
      GRID,
    )}`;
    await navigator.clipboard.writeText(url);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const publish = async () => {
    setPublishState('sending');
    try {
      const code = encodeBoard(
        [...cells].map((key) => ({ x: key % GRID.cols, y: Math.floor(key / GRID.cols) }) as Point),
        GRID,
      );
      const response = await fetch('/api/boards', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ code, name: boardName, author: 'anonymous', agentScore: bestAgentScore }),
      });
      const body = await response.json();
      setPublishState(response.ok ? 'done' : (body.detail ?? body.error ?? 'Could not publish that board.'));
    } catch {
      setPublishState('Could not reach the gallery.');
    }
  };

  return (
    <div className="designer">
      <div className="race-controls panel">
        <button onClick={run} className="primary" disabled={mode === 'run' || !agent}>
          {agent ? 'Run the agent' : 'Loading agent…'}
        </button>
        <button onClick={() => setMode('edit')} disabled={mode === 'edit'}>
          Stop
        </button>
        <button onClick={() => { setCells(new Set()); setOutcome(null); setProblem(null); }}>
          Clear
        </button>
        <button onClick={share} disabled={cells.size === 0}>
          {copied ? 'Link copied' : 'Copy share link'}
        </button>
        <div className="field">
          <p className="stat-label">Walls placed</p>
          <p className="seed">{cells.size}</p>
        </div>
      </div>

      {problem && (
        <p className="notice notice-problem" role="alert">
          {problem}
        </p>
      )}
      {outcome && (
        <p className="notice notice-outcome" role="status">
          {outcome}
        </p>
      )}

      {bestAgentScore !== null && publishState !== 'done' && (
        <form
          className="submit-row"
          onSubmit={(event) => {
            event.preventDefault();
            void publish();
          }}
        >
          <input
            value={boardName}
            onChange={(event) => setBoardName(event.target.value)}
            placeholder="Name this board"
            maxLength={40}
            aria-label="Name this board"
          />
          <button type="submit" disabled={publishState === 'sending'}>
            {publishState === 'sending' ? 'Publishing…' : 'Publish to gallery'}
          </button>
        </form>
      )}
      {publishState === 'done' && <p className="muted submit-note">Published to the gallery.</p>}
      {typeof publishState === 'string' && !['idle', 'sending', 'done'].includes(publishState) && (
        <p className="submit-note submit-error">{publishState}</p>
      )}

      <div className="board-frame designer-frame">
        {mode === 'edit' ? (
          <canvas
            ref={canvasRef}
            style={{ width, height, display: 'block', touchAction: 'none', cursor: 'crosshair' }}
            onPointerDown={(event) => {
              const canvas = canvasRef.current!;
              const bounds = canvas.getBoundingClientRect();
              const x = Math.floor(((event.clientX - bounds.left) / bounds.width) * GRID.cols);
              const y = Math.floor(((event.clientY - bounds.top) / bounds.height) * GRID.rows);
              painting.current = cells.has(cellKey({ x, y }, GRID.cols)) ? 'remove' : 'add';
              canvas.setPointerCapture(event.pointerId);
              paint(event);
            }}
            onPointerMove={paint}
            onPointerUp={() => { painting.current = null; }}
            onPointerCancel={() => { painting.current = null; }}
            aria-label="Level designer grid. Drag to place walls."
          />
        ) : engine ? (
          <Board engine={engine} version={version} accent="var(--agent)" cell={CELL} />
        ) : null}
      </div>
    </div>
  );
}
