# Multi-Level Snake Agent + Web Platform — Design

**Date:** 2026-08-14
**Status:** Approved (Section 1); sub-projects 2–4 to be detailed before their own implementation

## Goal

Turn the existing single-board DQN snake project into a multi-level game with an
agent that generalizes across levels, and publish it as a Next.js site on
Vercel's free tier where visitors race the agent and design levels for it.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Level model | Discrete selectable levels | Matches existing `level=` parameter; simplest to reason about |
| Site modes | Race the agent; Level designer vs agent | Chosen by user |
| Agent perception | 14 features → ~20, adding flood-fill free-space | Standard fix for self-trapping; required for hand-drawn boards |
| Backend | Postgres (Neon free tier) + leaderboard + level gallery | Genuinely full-stack, still free at expected traffic |
| Inference | Client-side TypeScript | Model is 14→256→3 (~20KB); no server cost, no latency |

Explicitly declined: explainability overlay, human-takeover mode, full user accounts,
CNN/grid vision.

## Problems With The Current Code

These are the defects the redesign must fix, not incidental cleanups.

1. **The engine cannot run headless.** `game.py` calls `pygame.display.set_mode()` in
   `__init__` and `_update_ui()` + `clock.tick(40)` on every `play_step`. Training is
   capped at 40 steps/sec and requires a display.
2. **Training level 2 destroys level 1's model.** `Linear_QNet.save()` always writes
   `model/model.pth`; `agent.py` calls it on every record. Checkpoints carry no
   identity — the current `model.pth` cannot be attributed to a level.
3. **Level 2 is one hardcoded layout.** `_generate_obstacles_level2` is 30 literal
   `Point(...)` calls, deliberately fixed. An agent trained on it memorizes geometry
   rather than learning obstacle avoidance, which fails the level-designer mode.
4. **Reward shaping leaks across episodes.** `agent.last_distance` is set in `train()`
   and never reset on `game.reset()`, so the first step of each episode is shaped
   against the last step of the previous one.
5. **State vector cannot perceive enclosed space.** Three danger bits and a 3-step
   lookahead give no signal about pockets, which is the dominant failure mode on
   obstacle-dense boards.

## The Parity Contract

The game must exist in Python (training) and TypeScript (browser) and behave
identically. Divergence would mean the trained agent is not the deployed agent.
Three mechanisms enforce this.

**Levels as data.** `shared/levels.json` holds geometry, dimensions, speed, and spawn
rules. Both languages load it. A user-designed board uses the same schema as a
built-in level, so the designer mode requires no separate format.

**A specified PRNG.** Python's `random` and JavaScript's `Math.random` cannot agree,
so a shared xorshift128 is implemented in both languages and used for every food
placement and procedural layout. Same seed ⇒ same sequence ⇒ a genuinely fair race.

**Golden trajectory tests.** Python records episodes (fixed seed, fixed action
sequence) to `shared/golden/`. A TypeScript test replays them and asserts equality
of head position, food position, score, collision flag, and every state feature,
frame by frame. Drift fails CI.

## Repository Structure

```
snake/
  core/          engine.py  levels.py  rng.py  state.py  types.py   (no pygame)
  render/        pygame_view.py                                      (optional)
  train/         agent.py  model.py  curriculum.py  evaluate.py
  export/        to_web.py                          (weights → JSON)
shared/
  levels.json    golden/
web/                                                 (Next.js, sub-project 4)
  lib/engine/    lib/agent/    app/
docs/superpowers/specs/
```

`game.py`, `agent.py`, `model.py`, `helper.py` move into this layout and split along
the logic/rendering seam.

## Sub-Project 1 — Headless Core + Level System

The only sub-project specified in full here. Later ones get their own design pass.

**`snake/core/rng.py`** — xorshift128 with explicit 32-bit wrapping, seeded
constructor, `next_u32()` and `randint(lo, hi)`. Ported verbatim to TypeScript later.
Must not use Python's `random` anywhere in core.

**`snake/core/types.py`** — `Point`, `Direction`, `Action`, `StepResult`. Preserves the
existing `Point`/`Direction` public names so training code reads familiarly.

**`snake/core/levels.py`** — loads and validates `shared/levels.json`; resolves a level
into concrete obstacle cells. Supports two obstacle sources: `fixed` (explicit cell
list, used by built-in and user-designed levels) and `procedural` (seeded generator
producing random blocks/corridors, used for generalization training).

**`snake/core/engine.py`** — `SnakeEngine`: pure logic, zero pygame, zero I/O, no
frame limiting. `reset(seed)`, `step(action) -> StepResult`, `is_collision(point)`.
Deterministic given a seed. This is what training drives at full speed.

**`snake/core/state.py`** — the ~20-feature encoder. Existing 14 features preserved in
their current order (so prior work remains comparable), then appended:
free-space fraction reachable straight/right/left via flood fill, whether the tail is
reachable from the head, and normalized distance-to-obstacle rays. Appending rather
than reordering keeps the diff legible and the feature indices stable.

**`snake/render/pygame_view.py`** — takes an engine and draws it. The only module that
imports pygame. Training never touches it.

**Testing.** Pytest, TDD. Determinism (same seed ⇒ identical episode), collision
correctness at each boundary, food never spawning on snake or obstacles, flood-fill
correctness on hand-built boards with known answers, and level-schema validation.

## Sub-Projects 2–4 (sketch only)

2. **Training + evaluation.** Curriculum sampling a mix of levels and procedural
   boards per episode; per-level checkpoints with metadata (feature version, level
   set, score history) to fix defect #2; an evaluation harness reporting mean score
   per level over N seeded episodes so "is it better" stops being a vibe.
3. **TypeScript engine + inference.** Port of `core` to `web/lib/engine`; weights
   exported as JSON and applied with two matmuls and a ReLU; golden-trajectory parity
   tests in CI.
4. **Next.js app.** Race mode, designer mode, level gallery, leaderboard. Server-side
   replay verification of submitted scores — the client sends seed + input sequence,
   the server replays it with the same engine and accepts the score only if it
   reproduces. This defeats forged scores and protects free-tier write limits.

## Constraints

- Must fit Vercel Hobby (free): static-first, client-side inference, Neon free-tier
  Postgres, no always-on process. Hobby is non-commercial use only.
- Model stays small enough to ship to the browser (~20KB target, hard ceiling 100KB).
- Python 3.14 with `pygame-ce` (upstream pygame has no 3.14 wheel).

## Risks

- **Engine divergence** between Python and TypeScript. Mitigated by golden tests; it
  is the single most likely source of hard-to-diagnose bugs.
- **Generalization to hand-drawn boards.** Procedural training and free-space features
  raise the floor but cannot guarantee an arbitrary adversarial board is solvable. The
  designer UI should validate reachability before offering a board to the agent.
- **Visual thinness.** With the explainability overlay declined, the UI carries less
  intrinsic content; race and designer modes must be well-executed to compensate.
