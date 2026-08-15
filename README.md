# Snake Agent

A deep Q-learning agent that plays Snake across four levels, and a Next.js site
where you can race it or design a level it has never seen.

The agent is a 25 → 256 → 3 network trained from nothing but a reward for eating
and a penalty for dying. It weighs 39 KB and runs **in the browser** — no server
decides its moves, which is what lets the whole site fit Vercel's free tier.

```text
                shared/levels.json  ·  shared/golden/
                   one definition, two implementations
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
  PYTHON (training)                            TYPESCRIPT (browser)
  snake/core/    engine, levels, state         web/lib/engine/
  snake/train/   DQN, curriculum, eval         web/lib/agent/   inference
        └──────────── weights.json (39 KB) ────────────▶
```

## How well it plays

Measured over 60 greedy episodes per level on fixed seeds:

| Level | Mean | Median | Best |
| --- | ---: | ---: | ---: |
| 1 · Open Field | 120.2 | 117.5 | 177 |
| 2 · Scattered Blocks | 107.6 | 111.0 | 170 |
| 3 · Corridors | 49.3 | 47.5 | 103 |
| 4 · Shifting Ground *(procedural)* | 96.1 | 102.5 | 162 |
| **Overall** | **93.3** | | |

One network plays all four. Nothing in its input identifies which level it is
on, so it had to learn obstacles as a general skill rather than memorise maps —
which is the only reason it can attempt a board you draw yourself.

## Quick start

Requires Python 3.14 and Node 20+.

```bash
python3 -m venv .venv
.venv/bin/python3 -m pip install -r requirements.txt
```

> The project depends on **pygame-ce**, not `pygame`. Upstream pygame publishes
> no wheel for Python 3.14 and its source build fails on macOS. pygame-ce is a
> drop-in fork — `import pygame` is unchanged.

Watch the trained agent in a desktop window:

```bash
python3 -m snake.play --level 2      # agent plays
python3 -m snake.play --human        # you play, arrow keys or WASD
```

Run the site:

```bash
cd web && npm install && npm run dev
```

## Training

```bash
python3 -m snake.train                       # 2000 episodes, full curriculum
python3 -m snake.train --episodes 6000       # longer
python3 -m snake.train --levels 1 2          # restrict to some levels
```

Two checkpoints are written to `model/`:

- **`agent_best.pth`** — the best-evaluating weights seen during the run. **Use this.**
- `agent.pth` — whatever the final episode produced.

They are not the same, and the gap is large. In the run that produced the
shipped agent, the best weights scored **93.31** and the final weights **84.92**
over 60 episodes per level. An earlier run was worse still: it peaked at 83.25
and finished at 34.01, because Q-learning can collapse late in training. Saving
only on a schedule ships the collapse.

Publish trained weights to the site:

```bash
python3 -m snake.export.to_web --name agent_best.pth
```

## How it works

### The engine is pure logic

`snake/core/engine.py` has no pygame, no clock, no I/O. The original
implementation called `pygame.display.set_mode()` in its constructor and
`clock.tick(40)` on every step, which capped training at **40 steps/second** and
required a display. Separating logic from rendering is what makes both fast
training and a browser port possible.

Rendering lives in `snake/render/pygame_view.py`, the only module that imports
pygame, and it observes an engine rather than driving one.

### Levels are data

`shared/levels.json` holds geometry in **grid cells**, not pixels, so it is not
tied to any canvas size. Obstacles come from two sources:

- `fixed` — explicit rectangles and cells (levels 1–3, and every board drawn in
  the web designer)
- `procedural` — regenerated from the episode seed (level 4), so the layout
  differs every game

The loader rejects boards that block the spawn corridor or wall off part of the
grid. A sealed region means food can appear where the snake can never reach it,
which would train the agent to fail an impossible task.

### What the agent sees — 25 features

Indices 0–13 are the original formulation; 14–24 were added to fix specific,
measured failures.

| Index | Feature | Why |
| --- | --- | --- |
| 0–2 | Immediate danger straight / right / left | |
| 3–5 | Path blocked within 3 cells | |
| 6–9 | Current heading (one-hot) | |
| 10–13 | Food bearing (one-hot-ish) | |
| 14–16 | **Free space** via flood fill, capped at body length | Detects developing traps |
| 17 | **Tail reachable** after moving straight | Strongest anti-self-trapping signal |
| 18–20 | Normalised distance to obstruction | |
| 21–23 | **Shortest-path first step** (BFS) | Bearing points *through* walls |
| 24 | **Food reachable at all** | |

Features 21–24 exist because of a measurement, not a hunch. With bearing alone,
the agent on Corridors ate **nothing** in 30 of 30 episodes — every run ended at
the stall limit with score 0. A bearing says "food is to the right"; in a maze
that points directly at a wall, and the agent pressed into it until the stall
timer killed it.

`FEATURE_VERSION` is recorded in every checkpoint. Loading a checkpoint whose
version disagrees with the encoder raises, rather than silently misreading every
input.

### Reward shaping

Shaping lives in the trainer, never the engine — that is what keeps the engine
portable, and it fixed a bug where `last_distance` was never reset between
episodes, so the first step of every game was shaped against the last step of
the previous one.

Distance-to-food uses **path distance, not Manhattan**. Manhattan distance
ignores walls, so on obstacle-dense boards the correct move — going *around* an
obstacle — was penalised on every step of the detour. The shaping was actively
teaching the agent not to route.

### The parity contract

The game exists twice: Python for training, TypeScript for the browser. If they
disagree even slightly, the agent you trained is not the agent on your site, and
the discrepancy is very hard to diagnose. Three mechanisms prevent that.

**A specified PRNG.** Python's `random` and JavaScript's `Math.random` cannot be
made to agree, so neither is used. Both sides implement the same xorshift128 with
splitmix32 seeding and rejection sampling, pinned by a frozen golden sequence.

**Golden trajectories.** Python records episodes — fixed seed, fixed actions — to
`shared/golden/`. TypeScript replays them and asserts every frame matches: head,
body, food, score, collision, and all 25 feature values.

**Inference fixtures.** The trajectory tests are deliberately weight-independent,
so they cannot catch a bug in weight export, base64 decoding, or matrix layout.
`shared/golden/qvalues.json` holds 100 recorded states with PyTorch's Q-values;
the browser must reproduce them and agree on argmax.

Re-record both after changing rules, geometry, features, or weights:

```bash
python3 -m snake.export.golden     # trajectories
python3 -m snake.export.qcheck     # Q-values (after retraining)
```

### Weights in the browser

The network is two matrix multiplies and a ReLU — about 40 lines of TypeScript,
no runtime dependency, no WASM, no server call. Weights export as base64 float32
rather than nested JSON numbers: JSON spends ~20 characters per float and
produced a 157 KB payload, over budget. Base64 float32 is exact at the precision
the network uses and cuts it to **39 KB**.

### Scores are verified, not trusted

Because the game runs client-side, a submitted score is just a number a browser
asserted. Rather than trust it, the client submits its **seed and move
sequence**, and the server replays that sequence through the same deterministic
engine, accepting the score only if the replay reproduces it.

```text
POST /api/scores  {"levelId":1,"seed":5,"moves":"0.","score":9999}
→ 422  {"error":"That run does not check out.",
        "detail":"replay scored 0, not 9999"}
```

Forging a score costs a genuine playthrough, which also protects the free-tier
write budget. This is only possible because the engine is deterministic and
shared.

## The site

| Route | What it does |
| --- | --- |
| `/` | The agent playing unattended, rotating levels on death |
| `/race` | You vs the agent — one seed, so both get the same food order |
| `/design` | Draw a board and set the agent loose on it |
| `/gallery` | Published boards, each showing the agent's best |
| `/leaderboard` | Verified runs only |

**Race** runs two engines from a single seed. Without the shared deterministic
RNG this would be two different games shown side by side, not a race.

**The designer** validates with the same rules the trainer uses, so a board that
runs is a board the agent could have trained on. Boards encode into the URL as a
768-bit bitmap (128 characters), so sharing one privately needs no database row.

**Design note:** colour carries information rather than mood — amber is always
the human, blue always the agent, on every board and table. Those match the
pygame renderer, so the desktop trainer and the site read as one product.

The database is **optional**. Race and the designer are entirely client-side;
without `POSTGRES_URL` the gallery and leaderboard explain themselves instead of
returning errors.

## Repository layout

```text
snake/
  core/          engine.py levels.py state.py rng.py grid.py types.py   (no pygame)
  render/        pygame_view.py                       (the only pygame import)
  train/         agent.py model.py curriculum.py evaluate.py loop.py
  export/        to_web.py golden.py qcheck.py
  play.py        desktop runner — watch the agent or play yourself
shared/
  levels.json    level geometry, read by both languages
  golden/        trajectories + Q-value fixtures
web/
  lib/engine/    TypeScript port of snake/core
  lib/agent/     browser inference
  lib/replay.ts  server-side run verification
  app/           routes and API
tests/           Python suite
web/tests/       parity, inference, replay, share codes
docs/            DEPLOY.md and the design spec
```

## Testing

```bash
python3 -m pytest tests/     # 52 tests — engine, levels, features
cd web && npm test           # 34 tests — parity, inference, replay, encoding
```

The parity suite matters most. If it fails, the browser is playing a subtly
different game from the one the agent trained in.

## Deploying

See [docs/DEPLOY.md](docs/DEPLOY.md). In short: import to Vercel with root
directory `web`, and optionally attach Neon Postgres for the leaderboard and
gallery. Vercel's Hobby plan is non-commercial use only.

## Performance

| | Steps/sec |
| --- | ---: |
| Original engine (display-bound, `clock.tick(40)`) | 40 |
| Headless engine, no feature encoding | ~70k–390k |
| Headless engine + full 25-feature encoding | ~3.4k |

Encoding costs roughly 10× what it did before the BFS path features were added —
a breadth-first search per frame is not free. It buys a level that went from
mean 2.2 to 49.3, so the trade is worth making, but it is the obvious place to
optimise if training time becomes the constraint.

## Known limits

- **Corridors (level 3) remains the weakest board** at 49.3 against 96–120
  elsewhere. Narrow routing with a long snake is genuinely hard for this
  architecture.
- **The shipped checkpoint is from episode 750 of 4000.** The remaining episodes
  never beat it. Evaluation is stable (78–89, no collapse), so this is a plateau
  rather than degradation — more episodes are not the lever. A larger hidden
  layer or full grid vision would be.
- **An adversarial hand-drawn board can still defeat the agent.** Procedural
  training and free-space features raise the floor; they cannot guarantee every
  legal board is solvable by this policy.

## Built on

Python 3.14 · PyTorch 2.13 · pygame-ce 2.5.8 · Next.js 15 · React 19 ·
Postgres (Neon)

Originally a single-level pygame Snake with a 14-feature DQN; the design
document for the rewrite is in
[docs/superpowers/specs/](docs/superpowers/specs/).
