/**
 * Verifies the browser's forward pass reproduces PyTorch's.
 *
 * The engine parity suite is weight-independent by design, so it cannot catch a
 * bug in weight export, base64 decoding, or matrix layout — each of which would
 * leave the site running an agent that picks different moves from the one that
 * was trained. This suite feeds recorded states through the exported weights and
 * compares Q-values against PyTorch's.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import test from 'node:test';
import assert from 'node:assert/strict';

import { Agent, type Weights } from '../lib/agent/infer.ts';
import { FEATURE_COUNT, FEATURE_VERSION } from '../lib/engine/state.ts';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(here, '..', '..');

const weights = JSON.parse(
  readFileSync(join(repoRoot, 'web', 'public', 'agent', 'weights.json'), 'utf8'),
) as Weights;
const fixture = JSON.parse(readFileSync(join(repoRoot, 'shared', 'golden', 'qvalues.json'), 'utf8'));

// Q-values are recorded to 5 decimals; float32 accumulation in a different order
// contributes a little more. Any real layout bug is orders of magnitude larger.
const EPSILON = 2e-3;

test('exported weights match the encoder contract', () => {
  assert.equal(weights.feature_version, FEATURE_VERSION);
  assert.equal(weights.input_size, FEATURE_COUNT);
  assert.equal(fixture.feature_version, FEATURE_VERSION);
});

test('weights payload stays inside the browser budget', () => {
  const bytes = readFileSync(join(repoRoot, 'web', 'public', 'agent', 'weights.json')).length;
  assert.ok(bytes < 100 * 1024, `weights are ${(bytes / 1024).toFixed(1)}KB, over the 100KB budget`);
});

test('forward pass reproduces PyTorch Q-values', () => {
  const agent = new Agent(weights);
  let maxDelta = 0;
  for (const [index, sample] of fixture.samples.entries()) {
    const got = agent.qValuesFromState(Float32Array.from(sample.state));
    assert.equal(got.length, sample.q.length);
    for (let i = 0; i < got.length; i += 1) {
      const delta = Math.abs(got[i] - sample.q[i]);
      maxDelta = Math.max(maxDelta, delta);
      assert.ok(
        delta < EPSILON,
        `sample ${index} action ${i}: ts ${got[i]} vs py ${sample.q[i]} (delta ${delta})`,
      );
    }
  }
  assert.ok(maxDelta < EPSILON);
});

test('chosen action matches PyTorch argmax on every sample', () => {
  const agent = new Agent(weights);
  for (const [index, sample] of fixture.samples.entries()) {
    const got = agent.qValuesFromState(Float32Array.from(sample.state));
    const argmax = (values: ArrayLike<number>) => {
      let best = 0;
      for (let i = 1; i < values.length; i += 1) if (values[i] > values[best]) best = i;
      return best;
    };
    assert.equal(argmax(got), argmax(sample.q), `action disagreement at sample ${index}`);
  }
});

test('agent weights are fetched and decoded once, not per board', () => {
  // The home page mounts five boards. Each would otherwise fetch and decode its
  // own copy of the network, so loadAgent memoises per URL.
  const originalFetch = globalThis.fetch;
  let fetches = 0;

  globalThis.fetch = (async () => {
    fetches += 1;
    return { ok: true, json: async () => weights } as unknown as Response;
  }) as typeof fetch;

  return (async () => {
    const { loadAgent } = await import('../lib/agent/infer.ts');
    const url = '/agent/weights.json?memo-test';
    const [a, b, c] = await Promise.all([loadAgent(url), loadAgent(url), loadAgent(url)]);
    const d = await loadAgent(url);

    globalThis.fetch = originalFetch;

    assert.equal(fetches, 1, 'weights should be fetched once for repeated callers');
    assert.equal(a, b);
    assert.equal(b, c);
    assert.equal(c, d, 'a later mount should reuse the resolved agent');
  })();
});

test('a failed weights load is not cached', () => {
  const originalFetch = globalThis.fetch;
  let attempts = 0;

  globalThis.fetch = (async () => {
    attempts += 1;
    if (attempts === 1) return { ok: false, status: 503 } as unknown as Response;
    return { ok: true, json: async () => weights } as unknown as Response;
  }) as typeof fetch;

  return (async () => {
    const { loadAgent } = await import('../lib/agent/infer.ts');
    const url = '/agent/weights.json?retry-test';

    await assert.rejects(() => loadAgent(url), /could not load agent weights/);
    const recovered = await loadAgent(url);

    globalThis.fetch = originalFetch;
    assert.ok(recovered, 'a retry after a failure should succeed rather than replay the error');
    assert.equal(attempts, 2);
  })();
});
