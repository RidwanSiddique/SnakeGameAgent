'use client';

import { useEffect, useRef } from 'react';
import type { SnakeEngine } from '../lib/engine/engine.ts';

export interface BoardProps {
  engine: SnakeEngine;
  /** Bumped by the parent on every tick to trigger a repaint. */
  version: number;
  /** Snake colour. Amber marks a human, blue marks the agent, everywhere. */
  accent: string;
  cell?: number;
  dimmed?: boolean;
}

const COLOURS = {
  ground: '#0d1017',
  lattice: '#161d29',
  wall: '#414859',
  wallEdge: '#525a6d',
  food: '#ec4858',
};

export function Board({ engine, version, accent, cell = 18, dimmed = false }: BoardProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const width = engine.grid.cols * cell;
  const height = engine.grid.rows * cell;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext('2d');
    if (!context) return;

    // Render at device resolution; a 1px lattice line looks muddy otherwise.
    const ratio = window.devicePixelRatio || 1;
    if (canvas.width !== width * ratio) {
      canvas.width = width * ratio;
      canvas.height = height * ratio;
    }
    context.setTransform(ratio, 0, 0, ratio, 0, 0);

    context.fillStyle = COLOURS.ground;
    context.fillRect(0, 0, width, height);

    context.strokeStyle = COLOURS.lattice;
    context.lineWidth = 1;
    context.beginPath();
    for (let x = 0; x <= engine.grid.cols; x += 1) {
      context.moveTo(x * cell + 0.5, 0);
      context.lineTo(x * cell + 0.5, height);
    }
    for (let y = 0; y <= engine.grid.rows; y += 1) {
      context.moveTo(0, y * cell + 0.5);
      context.lineTo(width, y * cell + 0.5);
    }
    context.stroke();

    context.globalAlpha = dimmed ? 0.45 : 1;

    for (const key of engine.obstacles) {
      const x = (key % engine.grid.cols) * cell;
      const y = Math.floor(key / engine.grid.cols) * cell;
      context.fillStyle = COLOURS.wall;
      context.fillRect(x, y, cell, cell);
      context.fillStyle = COLOURS.wallEdge;
      context.fillRect(x, y, cell, 2);
    }

    if (engine.food) {
      const x = engine.food.x * cell;
      const y = engine.food.y * cell;
      context.fillStyle = COLOURS.food;
      context.beginPath();
      context.arc(x + cell / 2, y + cell / 2, cell * 0.3, 0, Math.PI * 2);
      context.fill();
    }

    engine.snake.forEach((segment, index) => {
      // Fade toward the tail so direction of travel is readable at a glance.
      const falloff = 1 - Math.min(index / Math.max(engine.snake.length, 12), 0.55);
      context.globalAlpha = (dimmed ? 0.45 : 1) * falloff;
      context.fillStyle = accent;
      const inset = index === 0 ? 1 : 2;
      context.fillRect(
        segment.x * cell + inset,
        segment.y * cell + inset,
        cell - inset * 2,
        cell - inset * 2,
      );
    });

    context.globalAlpha = 1;
  }, [engine, version, accent, cell, width, height, dimmed]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width, height, display: 'block', borderRadius: 2 }}
      role="img"
      aria-label={`Game board, score ${engine.score}`}
    />
  );
}
