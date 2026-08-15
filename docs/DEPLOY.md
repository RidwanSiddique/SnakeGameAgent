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

> **The one setting that will break your build.** Directly beneath the Root
> Directory field, Vercel shows a checkbox:
>
> **"Include files outside of the Root Directory in the Build Step"** — this
> must be **ON**.
>
> The app imports `shared/levels.json` from outside `web/`, which is deliberate:
> level geometry has one definition shared with the Python trainer rather than a
> copy that can drift. With that checkbox off, Vercel clones only `web/` and the
> build fails with:
>
> ```text
> Module not found: Can't resolve '../../../shared/levels.json'
> Build failed because of webpack errors
> ```
>
> This is reproducible locally: copy `web/` somewhere on its own and run
> `npx next build`.

Two separate mechanisms are involved and both are needed:

- the checkbox above puts `shared/` on disk at **build** time, for module
  resolution
- `outputFileTracingRoot` in `next.config.mjs` includes `shared/levels.json` in
  the serverless bundle at **run** time, for the API routes and the dynamic
  gallery and leaderboard pages

If you ever move the app out of `web/`, both have to move with it.

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

## Custom domain via Namecheap + Cloudflare

All three services stay on free tiers: Cloudflare Free, Vercel Hobby, and the
domain you already own.

A caveat worth understanding first: Vercel already serves from a global CDN and
issues certificates itself. Cloudflare in front of it is a *second* CDN, and
proxying through it is the usual cause of TLS redirect loops with Vercel. Use
Cloudflare for DNS, keep the records unproxied, and everything is simple.

### 1. Add the domain to Cloudflare

Sign in to Cloudflare → **Add a site** → enter the domain → choose the **Free**
plan. Cloudflare scans existing records and gives you two nameservers, for
example `dana.ns.cloudflare.com` and `rick.ns.cloudflare.com`. They are unique
to your account — use the pair Cloudflare shows you.

### 2. Point Namecheap at Cloudflare

Namecheap → **Domain List** → **Manage** next to the domain → **Nameservers** →
change **Namecheap BasicDNS** to **Custom DNS**, then enter both Cloudflare
nameservers and save (the green tick).

Propagation usually takes minutes but is allowed up to 24 hours. Cloudflare
emails you when the zone is active. Until then, nothing else here will work.

### 3. Add the domain in Vercel

Vercel project → **Settings** → **Domains** → enter the domain → **Add**.

Add the apex (`example.com`) and let Vercel create the `www` redirect, or the
other way round — Vercel handles redirecting one to the other. Vercel then shows
the exact DNS records it wants.

### 4. Create those records in Cloudflare — unproxied

Cloudflare → your domain → **DNS** → **Records**. Add what Vercel asked for.
Typically:

| Type | Name | Value | Proxy |
| --- | --- | --- | --- |
| A | `@` | `76.76.21.21` | **DNS only** (grey cloud) |
| CNAME | `www` | `cname.vercel-dns.com` | **DNS only** (grey cloud) |

**Use the values Vercel displays, not the ones above.** They are the current
defaults, but Vercel changes them and the dashboard is authoritative.

**The proxy status is the part people get wrong.** Click the orange cloud so it
turns grey. Proxied records break Vercel's domain verification and its
certificate renewals, because Cloudflare terminates TLS before Vercel sees the
request.

If you have any **CAA** records, they must permit Let's Encrypt (`letsencrypt.org`)
or Vercel cannot issue a certificate. No CAA records at all is also fine.

### 5. Wait for verification

Vercel's Domains page moves to **Valid Configuration** and issues a certificate,
usually within minutes. The site is then live on your domain over HTTPS.

### 6. Only if you want Cloudflare's proxy features

Skip this unless you specifically want Cloudflare's WAF, caching rules, or
analytics. Once Vercel shows a valid certificate:

1. Cloudflare → **SSL/TLS** → set encryption mode to **Full (strict)**.
   **Flexible** causes an infinite redirect loop with Vercel — it is the single
   most common failure here.
2. Turn the cloud orange on the records above.

Watch for certificate renewal failures afterwards. If Vercel later reports an
invalid configuration, turn the cloud grey again and let it renew.

### Troubleshooting

| Symptom | Cause |
| --- | --- |
| Vercel stuck on "Invalid Configuration" | Records proxied (orange cloud), or nameservers not yet propagated |
| `ERR_TOO_MANY_REDIRECTS` | Cloudflare SSL mode is Flexible; set Full (strict) |
| Certificate never issues | A CAA record blocking Let's Encrypt |
| Domain resolves to a parking page | Namecheap still on BasicDNS, not Custom DNS |

Check what the world actually sees:

```bash
dig +short example.com
dig +short www.example.com
dig +short NS example.com
```
