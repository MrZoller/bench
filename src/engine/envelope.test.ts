import { describe, expect, it } from 'vitest';
import {
  computeEnvelope,
  comfortableFrontier,
  type CellState,
  type EnvelopeRequest,
} from './envelope';
import {
  DGX_SPARK,
  EPYC_9654,
  GPT_OSS_20B,
  DEEPSEEK_V3,
  LLAMA_31_8B,
  LLAMA_CPP,
  MAC_STUDIO_M3_ULTRA_512,
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
    usableTtftSeconds: 10,
    ...over,
  });
}

/**
 * Worse is lower. `unsupported` sits alongside `over` because both mean the configuration does
 * not run — they differ in what to do about it, not in how bad they are.
 */
const RANK: Record<CellState, number> = {
  comfortable: 3,
  tight: 2,
  offloaded: 1,
  over: 0,
  unsupported: 0,
};

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
   *
   * `unsupported` rather than `over`, because the two carry opposite advice: `over` means find
   * more memory, `unsupported` means pick another runtime. Collapsing them told an MLX-on-5090
   * user their hardware was too small, which is both wrong and unactionable — no amount of VRAM
   * makes MLX drive an NVIDIA card.
   */
  it('is entirely closed, and says why, when the runtime cannot drive the device', () => {
    const grid = envelope({ runtime: MLX, rig: { device: RTX_5090, count: 1 } });

    for (const row of grid.cells) {
      for (const cell of row) expect(cell.state).toBe('unsupported');
    }
    expect(comfortableFrontier(grid).every((f) => f === undefined)).toBe(true);
  });

  /**
   * Latency is half of what "usable" means, and only decode was being tested — so a resident
   * configuration with a long prompt was painted comfortable while the tile beside it read
   * "Slow start" in red about the same scenario.
   */
  /**
   * The prompt is part of the context, so a cell cannot be timed for a prompt it could not hold.
   * `coerce` enforces this for the selected scenario; the grid has to enforce it per column, or
   * every column is timed for the longest one. Carrying the slider's prompt through painted all
   * seven columns amber at an identical 41s — a latency impossible in six of them.
   */
  it('times each column for a prompt that column could actually hold', () => {
    const grid = envelope({
      usage: { contextTokens: 131072, concurrency: 1, promptTokens: 131072, kvPrecision: 'fp16' },
    });

    const row = grid.cells[0];
    const runnable = row.filter((c) => c.state !== 'over' && c.state !== 'unsupported');
    expect(runnable.length).toBeGreaterThan(1);

    // Strictly increasing: a bigger window admits a bigger prompt, which takes longer to read.
    for (let i = 1; i < runnable.length; i++) {
      expect(runnable[i].ttftSeconds).toBeGreaterThan(runnable[i - 1].ttftSeconds);
    }
  });

  it('refuses to call a cell comfortable when the first token is minutes away', () => {
    const grid = envelope({
      // A 128K prompt on a device with modest compute: fits, decodes acceptably, takes an age
      // to get going.
      usage: { contextTokens: 131072, concurrency: 1, promptTokens: 131072, kvPrecision: 'fp16' },
      usableTtftSeconds: 0.001,
    });

    for (const row of grid.cells) {
      for (const cell of row) {
        if (cell.state === 'comfortable') {
          throw new Error(`comfortable at ${cell.ttftSeconds}s to first token`);
        }
      }
    }

    // And the reason is carried, so the table can say which of the three it was.
    const tight = grid.cells.flat().filter((c) => c.state === 'tight');
    expect(tight.length).toBeGreaterThan(0);
    expect(tight.some((c) => c.tightBecause === 'latency')).toBe(true);
  });

  /**
   * A raiseable ceiling is not a hardware limit, and the Telemetry tile already says so. The grid
   * painting the same cells "past what this hardware can hold" contradicted it, and hid the one
   * change that would fix it.
   */
  it('separates a raiseable ceiling from the hardware itself', () => {
    // 512 GiB of physical memory, 384 GiB handed out by default. DeepSeek V3 at Q5 needs about
    // 444 GiB — inside the machine, outside the default. One `sysctl` away from running.
    const grid = envelope({
      model: DEEPSEEK_V3,
      quant: getQuant('q5_k_m'),
      rig: { device: MAC_STUDIO_M3_ULTRA_512, count: 1 },
      runtime: MLX,
    });

    const closed = grid.cells.flat().filter((c) => c.state === 'over');
    expect(closed.length).toBeGreaterThan(0);
    // Both kinds appear in one grid, which is the point: the small-context corner is a raiseable
    // ceiling away from running, and the far corner is past the machine however it is tuned.
    expect(closed.some((c) => c.overBecause === 'allocation')).toBe(true);
    expect(closed.some((c) => c.overBecause === 'capacity')).toBe(true);
    // And the distinction tracks the physical pool rather than being cosmetic: raising the
    // ceiling can only ever help the cells that fit inside the machine.
    const allocation = closed.filter((c) => c.overBecause === 'allocation');
    const capacity = closed.filter((c) => c.overBecause === 'capacity');
    expect(Math.max(...allocation.map((c) => c.utilization))).toBeLessThan(
      Math.min(...capacity.map((c) => c.utilization))
    );
  });

  it('still blames the hardware when the ceiling is not the thing in the way', () => {
    // A 32 GiB card with a fixed ceiling: raising a setting cannot help.
    const grid = envelope({
      model: DEEPSEEK_V3,
      quant: getQuant('q8_0'),
      rig: { device: RTX_5090, count: 1 },
    });

    const closed = grid.cells.flat().filter((c) => c.state === 'over');
    expect(closed.length).toBeGreaterThan(0);
    expect(closed.every((c) => c.overBecause === 'capacity')).toBe(true);
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
