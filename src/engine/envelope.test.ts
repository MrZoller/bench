import { describe, expect, it } from 'vitest';
import { computeEnvelope, comfortableFrontier, type EnvelopeRequest } from './envelope';
import {
  DGX_SPARK,
  EPYC_9654,
  GPT_OSS_20B,
  DEEPSEEK_V3,
  LLAMA_31_8B,
  LLAMA_CPP,
  MLX,
  RTX_5090,
} from './fixtures';
import { getQuant } from '@/data/quants';

/**
 * The envelope answers "how much room is left", so what these guard is the *shape* of the
 * region rather than any individual cell: it has to shrink monotonically as usage grows, and it
 * must never report headroom where the configuration cannot run at all.
 */

const CONTEXTS = [2048, 8192, 32768, 131072] as const;
const CONCURRENCIES = [1, 4, 16, 64] as const;

function envelope(over: Partial<EnvelopeRequest> = {}) {
  return computeEnvelope({
    model: LLAMA_31_8B,
    quant: getQuant('q4_k_m'),
    runtime: LLAMA_CPP,
    rig: { device: RTX_5090, count: 1 },
    usage: { contextTokens: 8192, concurrency: 1, promptTokens: 2048, kvPrecision: 'fp16' },
    contexts: CONTEXTS,
    concurrencies: CONCURRENCIES,
    usableTokensPerSec: 15,
    tightUtilization: 0.9,
    ...over,
  });
}

const RANK = { comfortable: 3, tight: 2, offloaded: 1, over: 0 } as const;

describe('the feasibility region', () => {
  it('covers every combination asked for', () => {
    const grid = envelope();
    expect(grid.cells).toHaveLength(CONCURRENCIES.length);
    for (const row of grid.cells) expect(row).toHaveLength(CONTEXTS.length);
  });

  /**
   * The load-bearing property. KV scales with context times concurrency, so pushing either axis
   * can only make things worse — a region that improved as usage grew would mean the placement
   * or the decode model had a sign error somewhere.
   */
  it('never improves as context or concurrency grows', () => {
    const grid = envelope();

    for (const row of grid.cells) {
      for (let i = 1; i < row.length; i++) {
        expect(RANK[row[i].state]).toBeLessThanOrEqual(RANK[row[i - 1].state]);
      }
    }
    for (let c = 0; c < grid.contexts.length; c++) {
      for (let r = 1; r < grid.cells.length; r++) {
        expect(RANK[grid.cells[r][c].state]).toBeLessThanOrEqual(RANK[grid.cells[r - 1][c].state]);
      }
    }
  });

  it('reports per-user throughput falling as users are added', () => {
    const grid = envelope();
    const atLowConcurrency = grid.cells[0][0].tokensPerSec;
    const atHighConcurrency = grid.cells[grid.cells.length - 1][0].tokensPerSec;

    expect(atLowConcurrency).toBeGreaterThan(atHighConcurrency);
  });

  /**
   * A runtime that cannot drive the hardware has no envelope at all. Shading a comfortable
   * region for it would be the same overclaim the Bench's verdict tiles already refuse.
   */
  it('is entirely closed when the runtime cannot drive the device', () => {
    const grid = envelope({ runtime: MLX, rig: { device: RTX_5090, count: 1 } });

    for (const row of grid.cells) {
      for (const cell of row) expect(cell.state).toBe('over');
    }
    expect(comfortableFrontier(grid).every((f) => f === undefined)).toBe(true);
  });

  it('calls out offload separately from merely being tight', () => {
    // 671B at Q4 on a 32 GB card runs only by spilling most of its weights.
    const grid = envelope({
      model: DEEPSEEK_V3,
      quant: getQuant('q4_k_m'),
      rig: { device: RTX_5090, count: 1 },
    });

    // Asserting the state itself, not "offloaded or over" — that disjunction passed with the
    // offload branch deleted entirely, since the top-right cells are `over` regardless.
    const offloaded = grid.cells.flat().filter((c) => c.state === 'offloaded');
    expect(offloaded.length).toBeGreaterThan(0);
    expect(grid.cells.flat().some((c) => c.state === 'comfortable')).toBe(false);
  });

  /**
   * Tight means two unrelated things — nearly full, or too slow — and a reader looking at one
   * amber square cannot tell which. Every tight cell has to say.
   */
  it('says why each tight cell is tight', () => {
    for (const grid of [
      envelope(),
      envelope({
        model: GPT_OSS_20B,
        quant: getQuant('mxfp4'),
        rig: { device: DGX_SPARK, count: 1 },
      }),
    ]) {
      for (const cell of grid.cells.flat()) {
        if (cell.state === 'tight') expect(cell.tightBecause).toBeDefined();
        else expect(cell.tightBecause).toBeUndefined();
      }
    }
  });

  /**
   * The classification has to be made on the figure the UI prints. Judged on the raw estimate, a
   * cell of 14.7 tok/s renders "Tight · 15 tok/s" against a threshold of 15.
   */
  it('classifies on the displayed rate when one is supplied', () => {
    const displayedRate = (n: number) => Math.round(n);
    const grid = envelope({ usableTokensPerSec: 15, displayedRate });

    for (const cell of grid.cells.flat()) {
      if (cell.state !== 'tight' || cell.tightBecause !== 'speed') continue;
      expect(displayedRate(cell.tokensPerSec)).toBeLessThan(15);
    }
  });

  /**
   * The frontier is what a reader takes from the picture: not which cells are green, but how
   * far each row can be pushed before it stops being pleasant.
   */
  it('reports a frontier that recedes as concurrency rises', () => {
    const grid = envelope({
      model: GPT_OSS_20B,
      quant: getQuant('mxfp4'),
      rig: { device: DGX_SPARK, count: 1 },
    });
    const frontier = comfortableFrontier(grid);

    const defined = frontier.filter((f): f is number => f !== undefined);
    for (let i = 1; i < defined.length; i++) {
      expect(defined[i]).toBeLessThanOrEqual(defined[i - 1]);
    }
  });

  it('leaves a slow machine with no comfortable region rather than a small one', () => {
    // ~6 tok/s on an EPYC host is below any interactive bar, at every context.
    const grid = envelope({
      model: DEEPSEEK_V3,
      quant: getQuant('q8_0'),
      rig: { device: EPYC_9654, count: 1 },
      usableTokensPerSec: 15,
    });

    expect(grid.cells.flat().some((c) => c.state === 'comfortable')).toBe(false);
  });
});
