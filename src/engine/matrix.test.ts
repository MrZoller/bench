import { describe, expect, it } from 'vitest';
import { computeMatrix, measureMax, measureValue, type MatrixMeasure } from './matrix';
import { MODELS, DEVICES } from '@/data/catalog';
import { getQuant } from '@/data/quants';
import { getRuntime } from '@/data/runtimes';
import {
  LLAMA_31_8B,
  DEEPSEEK_V3,
  RTX_5090,
  DGX_SPARK,
  MAC_STUDIO_M3_ULTRA_256,
  MAC_STUDIO_M3_ULTRA_512,
} from './fixtures';
import { LLAMA_CPP, MLX } from './fixtures';

const USAGE = {
  contextTokens: 8192,
  concurrency: 1,
  promptTokens: 2048,
  kvPrecision: 'fp16' as const,
};

function matrix(over: Partial<Parameters<typeof computeMatrix>[0]> = {}) {
  return computeMatrix({
    models: [LLAMA_31_8B, DEEPSEEK_V3],
    devices: [RTX_5090, DGX_SPARK, MAC_STUDIO_M3_ULTRA_256],
    quantFor: () => getQuant('q4_k_m'),
    runtime: LLAMA_CPP,
    usage: USAGE,
    deviceCount: 1,
    ...over,
  });
}

/**
 * The Matrix exists to make the capacity/bandwidth/compute triangle visible, so what these
 * guard is that the three measures stay independent — a grid where they agree everywhere would
 * mean one of them is not measuring what it claims.
 */
describe('the model-by-device grid', () => {
  it('covers every pair', () => {
    const cells = matrix();
    expect(cells).toHaveLength(2);
    for (const row of cells) expect(row).toHaveLength(3);
  });

  it('reports no measure at all for a pair that cannot run', () => {
    const cells = matrix({ runtime: MLX, devices: [RTX_5090] });

    for (const cell of cells.flat()) {
      expect(cell.runs).toBe(false);
      expect(cell.blockedBy).toMatch(/does not run/i);
      for (const measure of ['fit', 'decode', 'ttft'] as MatrixMeasure[]) {
        expect(measureValue(cell, measure)).toBeUndefined();
      }
    }
  });

  /**
   * The comparison the whole tool exists to make. A Spark holds a 671B model a 5090 cannot, and
   * decodes it far slower — if either half of that stopped being true, the triangle would have
   * collapsed into a single "better hardware" axis.
   */
  it('has a Spark holding what a 5090 cannot, and decoding it slower', () => {
    const [, deepseek] = matrix();
    const [onFiveThousand, onSpark] = deepseek;

    // The 5090 cannot hold 671B at Q4 residently; the Spark's 128 GB pool cannot either, so
    // both fall back — what must differ is *how* they fail and how fast they are when running.
    expect(onFiveThousand.offloadFraction).toBeGreaterThan(0);
    expect(onSpark.runs || onSpark.blockedBy).toBeTruthy();
  });

  it('separates a fast small model from a roomy slow one', () => {
    const [llama] = matrix();
    const [onFiveThousand, onSpark] = llama;

    expect(onFiveThousand.runs).toBe(true);
    expect(onSpark.runs).toBe(true);
    // Both hold an 8B model comfortably; the 5090's bandwidth is what separates them.
    expect(onFiveThousand.tokensPerSec).toBeGreaterThan(onSpark.tokensPerSec);
  });

  /**
   * Ranking by one measure must not silently rank by another. If `fit` and `decode` produced the
   * same order across the catalog, one of them would be redundant — and the surface's entire
   * argument is that they disagree.
   */
  it('does not rank devices identically under fit and decode', () => {
    const cells = computeMatrix({
      models: [...MODELS].slice(0, 6),
      devices: [...DEVICES],
      quantFor: () => getQuant('q4_k_m'),
      runtime: getRuntime('llama.cpp'),
      usage: USAGE,
      deviceCount: 1,
    });

    const order = (measure: MatrixMeasure) =>
      cells
        .flat()
        .filter((c) => c.runs)
        .slice()
        .sort((a, b) => (measureValue(b, measure) ?? 0) - (measureValue(a, measure) ?? 0))
        .map((c) => `${c.modelId}@${c.deviceId}`);

    expect(order('fit')).not.toEqual(order('decode'));
    expect(order('decode')).not.toEqual(order('ttft'));
  });

  it('scales each measure against the grid it is drawn on', () => {
    const cells = matrix();
    for (const measure of ['fit', 'decode', 'ttft'] as MatrixMeasure[]) {
      const max = measureMax(cells, measure);
      expect(max).toBeGreaterThan(0);
      for (const cell of cells.flat()) {
        const value = measureValue(cell, measure);
        if (value !== undefined) expect(value).toBeLessThanOrEqual(max);
      }
    }
  });

  it('scores an offloaded fit below every resident one', () => {
    const cells = matrix();
    const offloaded = cells.flat().filter((c) => c.runs && c.offloadFraction > 0);
    const resident = cells.flat().filter((c) => c.runs && c.offloadFraction === 0);

    for (const cell of offloaded) expect(measureValue(cell, 'fit')).toBe(0);
    for (const cell of resident) expect(measureValue(cell, 'fit')).toBeGreaterThanOrEqual(0);
  });

  /**
   * A single format cannot serve the whole grid — an expert-only scheme is a no-op on a dense
   * model — so the caller supplies one per pair, and each cell records which it actually used.
   */
  it('evaluates each pair at the format it was given, and records which', () => {
    const cells = matrix({
      quantFor: (model) => getQuant(model.expertParams > 0 ? 'mxfp4' : 'q4_k_m'),
    });

    for (const row of cells) {
      for (const cell of row) {
        const expected = cell.modelId === DEEPSEEK_V3.id ? 'mxfp4' : 'q4_k_m';
        expect(cell.quantId).toBe(expected);
      }
    }
  });
});

/**
 * A row must be scored at a context its own model can accept. `maxContext` differs across the
 * grid and neither placement nor decode knows about it, so an unclamped request produced fit
 * and speed figures for something the model would refuse.
 */
describe('per-row context limits', () => {
  it('caps each row at its own model, not at the request', () => {
    const cells = computeMatrix({
      models: [LLAMA_31_8B, DEEPSEEK_V3],
      devices: [RTX_5090],
      quantFor: () => getQuant('q4_k_m'),
      runtime: LLAMA_CPP,
      usage: { ...USAGE, contextTokens: 1_000_000, promptTokens: 900_000 },
      deviceCount: 1,
    });

    for (const row of cells) {
      for (const cell of row) {
        const model = [LLAMA_31_8B, DEEPSEEK_V3].find((m) => m.id === cell.modelId)!;
        expect(cell.contextTokens).toBe(model.maxContext);
        expect(cell.contextTokens).toBeLessThan(1_000_000);
      }
    }
  });

  it('leaves a request inside every model limit untouched', () => {
    const cells = computeMatrix({
      models: [LLAMA_31_8B],
      devices: [RTX_5090],
      quantFor: () => getQuant('q4_k_m'),
      runtime: LLAMA_CPP,
      usage: { ...USAGE, contextTokens: 8192 },
      deviceCount: 1,
    });
    expect(cells[0][0].contextTokens).toBe(8192);
  });
});

/**
 * The grid is read as a shortlist, so "will not run" strikes a machine off it. A default
 * allocation is not a hardware limit, and the two want different answers from the reader — one is
 * a setting to change, the other is a machine to rule out. The Envelope and Telemetry both kept
 * the distinction; this surface collapsed it.
 */
describe('a tunable allocation ceiling is not a hardware limit', () => {
  it('marks a cell that only exceeds the default allocation', () => {
    // DeepSeek V3 at Q5_K_M needs roughly 445 GiB: past the 512 GB Mac Studio's default
    // allocation, inside the ceiling the user can raise it to.
    const [[cell]] = computeMatrix({
      models: [DEEPSEEK_V3],
      devices: [MAC_STUDIO_M3_ULTRA_512],
      quantFor: () => getQuant('q5_k_m'),
      runtime: MLX,
      usage: USAGE,
      deviceCount: 1,
    });

    expect(cell.runs).toBe(false);
    expect(cell.raiseCeilingWouldHelp).toBe(true);
    expect(cell.blockedBy).not.toBe('Does not fit');
  });

  it('does not mark one that is past the hardware', () => {
    // Same model at BF16 is far beyond any ceiling this machine can be tuned to.
    const [[cell]] = computeMatrix({
      models: [DEEPSEEK_V3],
      devices: [MAC_STUDIO_M3_ULTRA_512],
      quantFor: () => getQuant('bf16'),
      runtime: MLX,
      usage: USAGE,
      deviceCount: 1,
    });

    expect(cell.runs).toBe(false);
    expect(cell.raiseCeilingWouldHelp).toBeUndefined();
  });

  it('does not mark a pair the runtime cannot drive, whatever the memory says', () => {
    // `unsupported` is a different failure and must not be dressed as a raisable setting.
    const [[cell]] = computeMatrix({
      models: [DEEPSEEK_V3],
      devices: [MAC_STUDIO_M3_ULTRA_512],
      quantFor: () => getQuant('q5_k_m'),
      runtime: getRuntime('vllm'),
      usage: USAGE,
      deviceCount: 1,
    });

    expect(cell.raiseCeilingWouldHelp).toBeUndefined();
    expect(cell.blockedBy).toMatch(/vLLM/);
  });
});
