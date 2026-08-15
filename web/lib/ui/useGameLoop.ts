'use client';

import { useEffect, useRef } from 'react';

/**
 * Drives game ticks at a fixed rate on top of requestAnimationFrame.
 *
 * An accumulator separates game speed from display refresh, so a 120Hz screen
 * does not play at double speed and a slow frame does not skip a move. The
 * accumulator is clamped so returning to a backgrounded tab replays a moment of
 * catch-up rather than several thousand queued steps.
 */
export function useGameLoop(onTick: () => void, stepsPerSecond: number, running: boolean) {
  const callback = useRef(onTick);
  callback.current = onTick;

  useEffect(() => {
    if (!running || stepsPerSecond <= 0) return;

    const interval = 1000 / stepsPerSecond;
    const maxCatchUp = interval * 5;
    let accumulated = 0;
    let previous = performance.now();
    let frame = 0;

    const advance = (now: number) => {
      frame = requestAnimationFrame(advance);
      accumulated = Math.min(accumulated + (now - previous), maxCatchUp);
      previous = now;

      while (accumulated >= interval) {
        accumulated -= interval;
        callback.current();
      }
    };

    frame = requestAnimationFrame(advance);
    return () => cancelAnimationFrame(frame);
  }, [running, stepsPerSecond]);
}
