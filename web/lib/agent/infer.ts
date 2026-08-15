/**
 * The trained agent, running in the browser.
 *
 * The network is 25 -> 256 -> 3, which is two matrix multiplies and a ReLU —
 * about 20KB of weights. There is no runtime dependency, no WASM, no server
 * call: a forward pass costs microseconds, so the agent can think on every
 * animation frame without a hosting bill.
 */
import type { SnakeEngine } from '../engine/engine.ts';
import { getState } from '../engine/state.ts';
import { Action } from '../engine/types.ts';

export interface RawLayer {
  shape: [number, number]; // [out, in]
  weight: string; // base64 float32, row-major
  bias: string;
  activation: 'relu' | 'linear';
}

export interface Weights {
  format: 'float32-base64';
  input_size: number;
  hidden_size: number;
  output_size: number;
  feature_version: number;
  layers: RawLayer[];
  trained?: { games: number; levels: number[]; best_score: number; mean_score: number };
}

interface Layer {
  out: number;
  in: number;
  weight: Float32Array;
  bias: Float32Array;
  relu: boolean;
}

function decodeFloat32(base64: string): Float32Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

function prepare(layer: RawLayer): Layer {
  const [out, inSize] = layer.shape;
  return {
    out,
    in: inSize,
    weight: decodeFloat32(layer.weight),
    bias: decodeFloat32(layer.bias),
    relu: layer.activation === 'relu',
  };
}

/** Flat row-major matrix-vector product; the weight matrix is [out][in]. */
function forwardLayer(input: Float32Array, layer: Layer): Float32Array {
  const result = new Float32Array(layer.out);
  for (let i = 0; i < layer.out; i += 1) {
    const rowStart = i * layer.in;
    let sum = layer.bias[i];
    for (let j = 0; j < layer.in; j += 1) sum += layer.weight[rowStart + j] * input[j];
    result[i] = layer.relu && sum < 0 ? 0 : sum;
  }
  return result;
}

export class Agent {
  // Declared explicitly rather than as constructor parameter properties: those
  // emit runtime assignments, so they are not erasable and Node's
  // type-stripping loader rejects them.
  private readonly layers: Layer[];
  private readonly weights: Weights;

  constructor(weights: Weights) {
    this.weights = weights;
    this.layers = weights.layers.map(prepare);
  }

  get provenance() {
    return this.weights.trained;
  }

  /** Q-values for [straight, right, left] from an already-encoded state. */
  qValuesFromState(state: Float32Array): Float32Array {
    if (state.length !== this.weights.input_size) {
      throw new Error(
        `feature mismatch: encoder produced ${state.length}, weights expect ` +
          `${this.weights.input_size}. The weights were trained on a different ` +
          `feature version and would misread every input.`,
      );
    }
    let activations = state;
    for (const layer of this.layers) activations = forwardLayer(activations, layer);
    return activations;
  }

  /** Q-values for [straight, right, left] at the engine's current position. */
  qValues(engine: SnakeEngine): Float32Array {
    return this.qValuesFromState(getState(engine));
  }

  act(engine: SnakeEngine): Action {
    const q = this.qValues(engine);
    let best = 0;
    for (let i = 1; i < q.length; i += 1) if (q[i] > q[best]) best = i;
    return best as Action;
  }
}

// Memoised per URL. The home page runs five boards at once, and each would
// otherwise fetch, decode and hold its own copy of the weights. The network is
// stateless, so one instance serves every board.
const pending = new Map<string, Promise<Agent>>();

export function loadAgent(url = '/agent/weights.json'): Promise<Agent> {
  const existing = pending.get(url);
  if (existing) return existing;

  const request = fetch(url)
    .then(async (response) => {
      if (!response.ok) throw new Error(`could not load agent weights (${response.status})`);
      return new Agent((await response.json()) as Weights);
    })
    .catch((cause) => {
      pending.delete(url); // let a later mount retry rather than cache the failure
      throw cause;
    });

  pending.set(url, request);
  return request;
}
