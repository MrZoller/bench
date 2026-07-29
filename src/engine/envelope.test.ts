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
  GPT_OSS_120B,
  DEEPSEEK_V3,
  LLAMA_31_8B,
  LLAMA_CPP,
  MAC_STUDIO_M3_ULTRA_512,
  MLX,
  RTX_5090,
  STRIX_HALO_395,
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
    // And the distinction tracks a real boundary rather than being cosmetic: raising the ceiling
    // can only ever help the cells below it. That boundary is the platform maximum — 480 of the
    // machine's 512 GiB — not the physical pool, since the last 32 GiB are what macOS needs to
    // keep running and are never handed to the model.
    const allocation = closed.filter((c) => c.overBecause === 'allocation');
    const capacity = closed.filter((c) => c.overBecause === 'capacity');
    expect(Math.max(...allocation.map((c) => c.utilization))).toBeLessThan(
      Math.min(...capacity.map((c) => c.utilization))
    );
  });

  it('will not offer to raise a ceiling that is already at its maximum', () => {
    // The Ryzen has 128 GiB physically and Variable Graphics Memory exposes 96 — which is its
    // catalogued default too, so nothing can be raised. Comparing against physical capacity
    // instead of the platform maximum told the user to change a setting that will not move.
    const grid = envelope({
      model: GPT_OSS_120B,
      quant: getQuant('q8_0'),
      rig: { device: STRIX_HALO_395, count: 1 },
      runtime: LLAMA_CPP,
    });

    const closed = grid.cells.flat().filter((c) => c.state === 'over');
    expect(closed.length).toBeGreaterThan(0);
    expect(closed.every((c) => c.overBecause === 'capacity')).toBe(true);
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

/**
 * The finding behind #65, kept as a property because it is why the picture is no longer painted
 * from `state`.
 *
 * At the shipped default scenario — gpt-oss 120B at MXFP4 on one DGX Spark, 8K prompt — 45 of the
 * 56 cells come back `tight`, one comes back `comfortable`, and not a single one is
 * `tightBecause: 'capacity'`. The classification is not wrong; it is *flat*, and three buckets over
 * a grid whose cells differ by 20x in decode and 561x in first-token latency cannot draw a shape.
 * Nor can the capacity threshold rescue it: the tight band is the last tenth of the ceiling while
 * both axes double per step, so the grid steps from 82% straight to 112% in every column.
 *
 * What this asserts is the half that has to stay true for the fill to be worth changing — that the
 * readings carry the shape the states do not. If a future change makes the *verdicts* vary too, this
 * still passes; it is a claim about cost, not about how many buckets there are.
 */
describe('the readings under one verdict', () => {
  // The axes the panel builds at the default scenario: `CONTEXT_STOPS` up to the model's own
  // 131,072 ceiling, and every concurrency stop.
  const DEFAULT_CONTEXTS = [2048, 4096, 8192, 16384, 32768, 65536, 131072] as const;
  const DEFAULT_CONCURRENCIES = [1, 2, 4, 8, 16, 32, 64, 128] as const;

  const defaultScenario = () =>
    envelope({
      model: GPT_OSS_120B,
      quant: getQuant('mxfp4'),
      runtime: LLAMA_CPP,
      rig: { device: DGX_SPARK, count: 1 },
      usage: { contextTokens: 32768, concurrency: 1, promptTokens: 8192, kvPrecision: 'fp16' },
      contexts: DEFAULT_CONTEXTS,
      concurrencies: DEFAULT_CONCURRENCIES,
      // The responsive bar, which is what the panel passes — not this file's default of 10.
      usableTtftSeconds: 2,
    });

  it('spreads orders of magnitude of cost across cells that share one verdict', () => {
    const cells = defaultScenario().cells.flat();
    expect(cells).toHaveLength(DEFAULT_CONTEXTS.length * DEFAULT_CONCURRENCIES.length);

    // The largest group of same-verdict cells, which at this scenario is 45 of the 56.
    const byState = new Map<CellState, typeof cells>();
    for (const cell of cells) byState.set(cell.state, [...(byState.get(cell.state) ?? []), cell]);
    const largest = [...byState.values()].sort((a, b) => b.length - a.length)[0];
    expect(largest.length / cells.length, 'no verdict covers much of this grid').toBeGreaterThan(
      0.5
    );

    const rates = largest.map((c) => c.tokensPerSec);
    const waits = largest.map((c) => c.ttftSeconds);
    expect(Math.max(...rates) / Math.min(...rates)).toBeGreaterThan(10);
    expect(Math.max(...waits) / Math.min(...waits)).toBeGreaterThan(100);
  });

  it('leaves the capacity band unreached, because the axes double and the band does not', () => {
    /*
     * Not a defect in the threshold — 90% of the ceiling is the right place to call a *point*
     * scenario nearly full, and Telemetry uses it that way. It is a fact about this grid: every step
     * of either axis doubles the cache, so the utilization of adjacent cells jumps clean over a band
     * ten points wide, and the one cause that is about the two quantities these axes measure never
     * fires. That is what the fill had to stop depending on.
     */
    const cells = defaultScenario().cells.flat();
    expect(cells.some((c) => c.state === 'tight')).toBe(true);
    expect(cells.filter((c) => c.tightBecause === 'capacity')).toHaveLength(0);

    // And the utilizations really do straddle it rather than merely staying low: the fullest cell
    // that runs is under the band, and the emptiest that does not is over the ceiling entirely.
    const running = cells.filter((c) => c.state !== 'over' && c.state !== 'unsupported');
    const closed = cells.filter((c) => c.state === 'over');
    expect(Math.max(...running.map((c) => c.utilization))).toBeLessThan(0.9);
    expect(Math.min(...closed.map((c) => c.utilization))).toBeGreaterThan(1);
  });
});

/**
 * Every cell used to be timed for a single prompt, because `estimatePrefill` read `promptTokens`
 * and never `concurrency`. A row at 64 users therefore inherited the one-user time-to-first-token,
 * so it could stay Comfortable on a latency nobody in that row would actually see — the promptness
 * half of the comfort claim was being made from a measurement of a quieter machine.
 *
 * The engine now prices the concurrent prefill workload, and each cell passes its own concurrency
 * through, so these assert the property from the outside rather than trusting that it propagated.
 */
describe('a cell is timed for the traffic its own row carries', () => {
  const ttftAt = (concurrency: number) => {
    const grid = envelope({ concurrencies: [concurrency], contexts: [8192] });
    return grid.cells.flat()[0].ttftSeconds;
  };

  it('charges a busy row for the prompts it is serving', () => {
    // Compute-bound work does not amortize the way decode does, so sixteen prompts is sixteen
    // times the arithmetic — not the same wait reported sixteen times over.
    expect(ttftAt(16)).toBeGreaterThan(ttftAt(1));
    expect(ttftAt(16) / ttftAt(1)).toBeCloseTo(16, 0);
  });

  it('withdraws the comfort claim once the wait crosses the bar', () => {
    // A latency bar low enough that the one-user cell clears it and the sixteen-user cell cannot.
    const bar = ttftAt(1) * 4;
    const grid = envelope({
      contexts: [8192],
      concurrencies: [1, 16],
      usableTtftSeconds: bar,
      // Held out of the way so promptness is the only thing that can decide these two cells.
      usableTokensPerSec: 0,
      tightUtilization: 1,
    });

    const [one, many] = grid.cells.flat();
    expect(one.state).toBe('comfortable');
    expect(many.state).not.toBe('comfortable');
  });
});

/**
 * The prompt clamp and the prefix clamp are the same rule applied to the two halves of the
 * working set, and for as long as both parameters existed only one of them was clamped. Decode
 * ignores the prefix, so the tell is entirely in time-to-first-token: every new token is charged
 * for attending over the resident session, whether or not the column could hold it.
 */
describe('a cell is timed for a working set its own column could hold', () => {
  // No column here is wider than the prompt, so once the prompt is clamped to the column there is
  // nothing left for a prefix to occupy in any of them. A wider column would have room, and would
  // rightly be charged — which is the next test.
  const FILLED = [2048, 8192, 32768, 65536] as const;

  const at = (over: Partial<EnvelopeRequest['usage']>) =>
    envelope({
      contexts: FILLED,
      usage: {
        contextTokens: 65536,
        concurrency: 1,
        promptTokens: 65536,
        kvPrecision: 'fp16',
        ...over,
      },
    }).cells[0];

  it('ignores a cached prefix the column has no room for', () => {
    const without = at({ cachedPrefixTokens: 0 });
    const with64K = at({ cachedPrefixTokens: 65536 });

    // Not merely "close": the prefix has nowhere to go in any of these columns, so the two grids
    // are the same grid. Unclamped, the 2K column was timed against a session 32 times its size.
    for (let i = 0; i < without.length; i++) {
      expect(with64K[i].ttftSeconds).toBe(without[i].ttftSeconds);
      expect(with64K[i].state).toBe(without[i].state);
    }
  });

  it('still charges for a prefix that does fit beside the prompt', () => {
    // The clamp has to narrow the prefix, not discard it — a cell where the session genuinely is
    // resident really does pay to attend over it, and that is the whole point of the parameter.
    const prompt = 2048;
    const without = envelope({
      usage: { contextTokens: 65536, concurrency: 1, promptTokens: prompt, kvPrecision: 'fp16' },
    }).cells[0];
    const withPrefix = envelope({
      usage: {
        contextTokens: 65536,
        concurrency: 1,
        promptTokens: prompt,
        cachedPrefixTokens: 32768,
        kvPrecision: 'fp16',
      },
    }).cells[0];

    const wide = withPrefix.findIndex((c) => c.contextTokens >= prompt + 32768);
    expect(wide).toBeGreaterThanOrEqual(0);
    expect(withPrefix[wide].ttftSeconds).toBeGreaterThan(without[wide].ttftSeconds);
  });

  it('leaves a narrow column untouched by a prefix only a wide one can hold', () => {
    // The same request read across the row: the wide column pays, the narrow one cannot and does
    // not. A clamp on the prompt alone gets the first half of this right and the second wrong.
    // The prompt fills the narrow column exactly, so the room left there is nil and the room left
    // in the wide one is nearly all of it.
    const narrow = 2048;
    const grid = envelope({
      contexts: [narrow, 131072],
      usage: {
        contextTokens: 131072,
        concurrency: 1,
        promptTokens: narrow,
        cachedPrefixTokens: 65536,
        kvPrecision: 'fp16',
      },
    }).cells[0];
    const baseline = envelope({
      contexts: [narrow, 131072],
      usage: { contextTokens: 131072, concurrency: 1, promptTokens: narrow, kvPrecision: 'fp16' },
    }).cells[0];

    expect(grid[0].ttftSeconds).toBe(baseline[0].ttftSeconds);
    expect(grid[1].ttftSeconds).toBeGreaterThan(baseline[1].ttftSeconds);
  });
});
