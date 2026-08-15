# Deploying to Vercel

The site is designed to fit the Hobby (free) plan. The agent runs in the
visitor's browser, so there is no inference cost and no always-on process — the
only server work is verifying submitted runs and reading two small tables.

> Vercel's Hobby plan is for non-commercial use.

## 1. Publish the trained weights

The site loads `web/public/agent/weights.json`. Regenerate it whenever you
retrain:

```bash
python3 -m snake.export.to_web --name agent_best.pth
```

Use `agent_best.pth`, not `agent.pth`. Q-learning can degrade late in a run, so
the trainer keeps the best-evaluating weights separately; `agent.pth` is simply
whatever the last episode produced and is often much worse.

The export prints the payload size and warns above 100 KB. It is ~39 KB, which
is smaller than most hero images.

## 2. Import the repository

Point Vercel at the repo and set **Root Directory** to `web`.

The app imports `shared/levels.json` from outside `web/`, which is deliberate:
level geometry has one definition shared with the Python trainer rather than a
copy that can drift. `next.config.mjs` sets `outputFileTracingRoot` to the repo
root so that file is included in the deployment bundle. If you move the app,
that setting has to move with it.

## 3. Attach Postgres (optional)

Race and the designer work with no database at all. The leaderboard and gallery
need one.

In the Vercel dashboard: **Storage → Create Database → Neon Postgres** (free
tier), then connect it to the project. That sets `POSTGRES_URL` automatically;
`DATABASE_URL` is also accepted.

Tables are created on first use — there is no migration step to run. Without a
database, the pages render an explanation rather than an error, so a deploy
before you attach one is not broken.

## 4. Deploy

Vercel runs `npm run build`. Nothing else is required.

## Checks worth running before you ship

```bash
python3 -m pytest tests/     # engine, levels, features
cd web && npm test           # parity, inference, replay verification, share codes
```

The parity suite is the one that matters most: it replays trajectories recorded
by Python through the TypeScript engine and asserts every frame and every
feature value matches. If it fails, the browser is playing a subtly different
game from the one the agent trained in, and the agent will underperform on the
site for reasons that are otherwise very hard to diagnose.

Re-record the fixtures after any change to game rules, level geometry, or the
feature encoder:

```bash
python3 -m snake.export.golden     # engine trajectories
python3 -m snake.export.qcheck     # Q-value fixtures (after retraining)
```

## Free-tier notes

- **Score submissions are replayed server-side.** A forged score costs the
  attacker a genuine playthrough, which also keeps junk out of the write budget.
- **Board sharing needs no database row.** A level is a 768-bit bitmap encoded
  into the URL, so sharing privately costs nothing. The gallery stores only
  boards people explicitly publish.
- **Connection pooling** is capped at 3 and reused across warm invocations;
  Neon's free tier does not tolerate a pool per request.
