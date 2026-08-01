import { describe, expect, it } from 'vitest';
import { recommend, type Candidate, type RecommendInputs } from './recommend';
import { planPlacement } from './placement';
import { WORKLOADS } from './verdict';
import { MODELS, getDevice } from '@/data/catalog';
import { QUANTS, getQuant } from '@/data/quants';
import { RUNTIMES } from '@/data/runtimes';
import { quantApplies } from '@/lib/quantChoice';
import type { DeviceSpec } from './types';
import type { Fitness } from './verdict';

/** Best to worst, so a tier comparison reads as one. */
const TIERS: readonly Fitness[] = ['fail', 'tight', 'good'];

/**
 * The recommender (#138).
 *
 * The issue names three pieces of real work and they are all *policy* — the ranking tie-break, the
 * quant auto-pick, and what to say when nothing clears the bar. So these tests are mostly about
 * whether a stated rule is the rule actually implemented, which is the failure mode a ranked list
 * has: an order that looks reasonable and is not the one printed beside it is an opinion wearing
 * the chassis of a measurement.
 *
 * Run against the **shipped catalog** rather than fixtures. A recommender is a claim about what is
 * in the catalog, and one checked only against three fixture models is checking the fixtures.
 */

const sweep = (device: DeviceSpec, over: Partial<RecommendInputs> = {}): RecommendInputs => ({
  device,
  deviceCount: 1,
  kvPrecision: 'fp16',
  concurrency: 1,
  workloadId: 'agent',
  models: MODELS,
  runtimes: RUNTIMES,
  quantsFor: (model, runtime) => QUANTS.filter((q) => quantApplies(q, model, device, runtime)),
  ...over,
});

const RTX_5090 = getDevice('rtx-5090');

describe('the sweep’s axes are the engine’s, not the Matrix’s', () => {
  it('crosses runtimes, so a model one runtime refuses can still be recommended', () => {
    const list = recommend(sweep(RTX_5090));

    // The Matrix renders every cell under one globally selected runtime. If this swept only one,
    // every row would name it — and a model categorically refused there would be absent entirely.
    const runtimes = new Set(list.ranked.map((c) => c.runtime.id));
    expect(runtimes.size).toBeGreaterThan(1);
  });

  it('takes device count as an input rather than assuming one', () => {
    // `Matrix.tsx` scores every cell at a hardcoded `deviceCount: 1`, which is the assumption this
    // exists to escape: a second card changes both fit and rank.
    const one = recommend(sweep(RTX_5090, { deviceCount: 1 }));
    const four = recommend(sweep(RTX_5090, { deviceCount: 4 }));

    expect(four.ranked.length).toBeGreaterThan(0);
    expect(
      four.ranked.map((c) => `${c.model.id}/${c.quant.id}`).join(),
      'the device count changed nothing, so it is not an axis'
    ).not.toBe(one.ranked.map((c) => `${c.model.id}/${c.quant.id}`).join());
  });

  it('takes KV precision as an input, since a narrower cache changes what fits', () => {
    const fp16 = recommend(sweep(RTX_5090, { kvPrecision: 'fp16' }));
    const q8 = recommend(sweep(RTX_5090, { kvPrecision: 'q8' }));

    expect(q8.ranked.length).toBeGreaterThan(0);
    expect(q8.ranked.map((c) => c.model.id).join()).not.toBe(
      fp16.ranked.map((c) => c.model.id).join()
    );
  });

  it('grades every archetype it is asked for, rather than one hardcoded scenario', () => {
    for (const workload of WORKLOADS) {
      const list = recommend(sweep(RTX_5090, { workloadId: workload.id }));
      expect(list.workload.id, workload.id).toBe(workload.id);
      expect(list.ranked.length, workload.id).toBeGreaterThan(0);
    }
  });

  it('refuses a workload it does not have', () => {
    expect(() => recommend(sweep(RTX_5090, { workloadId: 'telepathy' }))).toThrow(
      /Unknown workload/
    );
  });
});

describe('the ranking is the rule that is printed', () => {
  const ordered = (list: readonly Candidate[]) => list.map((c) => `${c.model.id} ${c.quant.id}`);

  it('puts every cleared verdict above every failed one', () => {
    const list = recommend(sweep(RTX_5090));
    const tiers = list.ranked.map((c) => c.fitness);
    const lastPass = tiers.lastIndexOf('good');
    const firstFail = tiers.indexOf('fail');

    expect(lastPass, 'nothing cleared, so the tier order is untested here').toBeGreaterThanOrEqual(
      0
    );
    if (firstFail >= 0) expect(firstFail).toBeGreaterThan(lastPass);
  });

  it('ranks within a tier by parameter count, which is the stated proxy', () => {
    const list = recommend(sweep(RTX_5090));
    const good = list.ranked.filter((c) => c.fitness === 'good');
    expect(good.length, 'fewer than two cleared, so ordering is untested').toBeGreaterThan(1);

    for (let i = 1; i < good.length; i++) {
      expect(
        good[i - 1].model.totalParams,
        `${good[i - 1].model.id} ranked above ${good[i].model.id}`
      ).toBeGreaterThanOrEqual(good[i].model.totalParams);
    }
  });

  it('breaks a parameter tie towards the less compressed format', () => {
    // The same model at two formats both clearing the bar: the wider one is less lossy, so it wins.
    // Without this clause the order would fall through to decode rate, which prefers the *narrower*
    // format — recommending Q3 over Q8 for being faster at a bar both already clear.
    const list = recommend(sweep(RTX_5090));
    const byModel = new Map<string, Candidate[]>();
    for (const c of list.ranked) {
      byModel.set(c.model.id, [...(byModel.get(c.model.id) ?? []), c]);
    }

    let checked = 0;
    for (const entries of byModel.values()) {
      const pairs = entries.filter((c) => c.fitness === entries[0].fitness);
      for (let i = 1; i < pairs.length; i++) {
        if (pairs[i - 1].quant.bpw === pairs[i].quant.bpw) continue;
        expect(pairs[i - 1].quant.bpw).toBeGreaterThan(pairs[i].quant.bpw);
        checked += 1;
      }
    }
    expect(checked, 'no model appeared twice in one tier, so this checked nothing').toBeGreaterThan(
      0
    );
  });

  it('is a total order, so the same catalog always produces the same shortlist', () => {
    // The model id is the last clause and is deliberately not in the printed rule — it is not a
    // judgement, it is what stops two configurations equal on every stated axis ranking by
    // whichever order the sweep happened to visit them in.
    expect(ordered(recommend(sweep(RTX_5090)).ranked)).toEqual(
      ordered(recommend(sweep(RTX_5090)).ranked)
    );
  });
});

describe('the quant auto-pick', () => {
  it('goes through the caller’s applicability rule, with the runtime', () => {
    // The Matrix's P1 in recommender form: a hardcoded fallback scored dense rows at a format vLLM
    // cannot read. Every format on the shortlist has to be one its own runtime loads.
    for (const c of recommend(sweep(RTX_5090)).ranked) {
      expect(c.runtime.weightFormats, `${c.runtime.id} cannot load ${c.quant.id}`).toContain(
        c.quant.id
      );
      expect(quantApplies(c.quant, c.model, RTX_5090, c.runtime)).toBe(true);
    }
  });

  /**
   * **The test that had to catch the real defect and could not.**
   *
   * Its first version filtered wider formats by `q.bpw > c.quant.bpw` and then asserted exactly
   * that — `expect(q.bpw).toBeGreaterThan(c.quant.bpw)`, a restatement of its own filter. It cannot
   * fail on any input, and it passed green over 347 shipping configurations where the pick was
   * genuinely wrong: `QUANTS` is grouped by checkpoint family rather than by width, so the
   * widest-first walk was not widest-first and stopped early on a narrower format.
   *
   * The falsifiable form has to *grade* the wider format, which means running the sweep again with
   * that one format and comparing tiers. Reverting the sort in `recommend` turns this red.
   */
  it('never picks a narrower format than one grading better', () => {
    /**
     * **Three machines, not one, and that is the difference between this test working and not.**
     * A first version swept only the 5090 and stayed green under a mutation that deletes the width
     * sort — the divergences the review found were on a Ryzen AI Max and the M4 Pro minis, where
     * `mxfp4` and `int8` sit above the wider GGUF formats in catalog order. A policy test run on
     * one row of the hardware catalog is a test of that row.
     */
    let compared = 0;
    for (const deviceId of ['rtx-5090', 'ryzen-ai-max-395', 'mac-mini-m4-pro-64']) {
      const device = getDevice(deviceId);
      const list = recommend(sweep(device, { workloadId: 'chat' }));
      expect(list.ranked.length, deviceId).toBeGreaterThan(0);

      for (const c of list.ranked) {
        const wider = QUANTS.filter(
          (q) => q.bpw > c.quant.bpw && quantApplies(q, c.model, device, c.runtime)
        );

        for (const q of wider) {
          // The same sweep, restricted to this one pairing at this one format, so the grade comes
          // from the same path rather than from a reimplementation of it.
          const alone = recommend(
            sweep(device, {
              workloadId: 'chat',
              models: [c.model],
              runtimes: [c.runtime],
              quantsFor: () => [q],
            })
          );
          if (alone.ranked.length === 0) continue; // does not load: why narrowing is a strategy
          compared += 1;

          /**
           * The claim, and its direction was wrong on the first attempt — which the fixed test
           * caught immediately, by failing on Mistral Small 4 119B where BF16 grades `fail` and the
           * pick is the narrower NVFP4 at `good`. That is correct: narrowing when the wider format
           * fails is the entire strategy. What must never happen is the reverse — a wider format
           * grading *better* than the one picked.
           */
          expect(
            TIERS.indexOf(alone.ranked[0].fitness),
            `${c.model.id} under ${c.runtime.id}: ${q.label} grades ` +
              `${alone.ranked[0].fitness} and the pick was the narrower ${c.quant.label} at ${c.fitness}`
          ).toBeLessThan(TIERS.indexOf(c.fitness));
        }
      }
    }

    expect(compared, 'no wider format loaded anywhere, so this compared nothing').toBeGreaterThan(
      0
    );
  });

  it('walks widest-first whatever order the catalog is in', () => {
    // The precondition the first draft delegated to the caller and the caller could not meet.
    // `quants.ts` is grouped by checkpoint family on purpose — q8_0 at 8.5 bpw sits below nvfp4 at
    // 4.5 — so an unsorted `QUANTS.filter` is not a width order, and the shipped call site passes
    // exactly that. The sweep sorts, so the two orders must give one answer.
    const forward = recommend(sweep(RTX_5090));
    const reversed = recommend(
      sweep(RTX_5090, {
        quantsFor: (model, runtime) =>
          [...QUANTS].reverse().filter((q) => quantApplies(q, model, RTX_5090, runtime)),
      })
    );

    expect(reversed.ranked.map((c) => `${c.model.id} ${c.quant.id}`)).toEqual(
      forward.ranked.map((c) => `${c.model.id} ${c.quant.id}`)
    );
  });

  it('never recommends a model the machine cannot load at all', () => {
    // Absence rather than a low ranking: a shortlist entry is a recommendation, and the Matrix is
    // where refusals are already legible.
    const tiny: DeviceSpec = { ...RTX_5090, allocatableBytes: 1024, capacityBytes: 2048 };
    const list = recommend(sweep(tiny));

    for (const c of list.ranked) {
      const placement = planPlacement(
        c.model,
        c.quant,
        {
          contextTokens: Math.min(c.model.maxContext, list.workload.typicalPromptTokens + 512),
          concurrency: 1,
          promptTokens: list.workload.typicalPromptTokens,
          kvPrecision: 'fp16',
        },
        { device: tiny, count: 1 },
        c.runtime
      );
      expect(placement.impossible, `${c.model.id} is on the list and cannot load`).toBe(false);
    }
  });
});

describe('"nothing" is a wrong answer when something runs', () => {
  /**
   * A machine where the bar is genuinely out of reach: an 8 GiB card asked for a coding agent,
   * which wants 25 tok/s at a 64K session. Plenty loads; nothing clears.
   */
  const small: DeviceSpec = {
    ...RTX_5090,
    id: 'small-fixture',
    name: 'Small fixture',
    capacityBytes: 8 * 1024 ** 3,
    allocatableBytes: 7 * 1024 ** 3,
    bandwidthBytesPerSec: 2e10,
  };

  it('offers the fastest thing that loads when nothing clears the bar', () => {
    const list = recommend(sweep(small, { workloadId: 'agent' }));

    expect(list.best, 'something cleared, so the fallback path is untested').toBeUndefined();
    expect(
      list.ranked.length,
      'nothing loads at all, so there is nothing to fall back to'
    ).toBeGreaterThan(0);
    expect(list.fallback).toBeDefined();

    // By decode rate, per FALLBACK_RULE — deliberately not `ranked[0]`, which is the *largest*
    // thing that loads. A 671B decoding at 0.3 tok/s is not more useful than an 8B at 40 that
    // merely missed a threshold, and the two picks answer different questions.
    const fastest = Math.max(...list.ranked.map((c) => c.tokensPerSec));
    expect(list.fallback!.tokensPerSec).toBe(fastest);
  });

  it('does not offer a fallback when something already clears', () => {
    const list = recommend(sweep(RTX_5090));
    expect(list.best).toBeDefined();
    expect(list.fallback, 'a fallback beside a real answer is two headlines').toBeUndefined();
  });

  it('returns an empty shortlist rather than inventing one when nothing loads', () => {
    const useless: DeviceSpec = { ...RTX_5090, allocatableBytes: 1, capacityBytes: 2 };
    const list = recommend(sweep(useless));

    expect(list.ranked).toEqual([]);
    expect(list.best).toBeUndefined();
    expect(list.fallback).toBeUndefined();
    expect(list.runnersUp).toEqual([]);
    // It still says what it looked at, which is what distinguishes "nothing runs" from "nothing ran".
    expect(list.pairsConsidered).toBeGreaterThan(0);
  });
});

describe('the runners-up are a choice rather than one model spelled three ways', () => {
  it('names a different model each', () => {
    const list = recommend(sweep(RTX_5090));
    const headline = list.best ?? list.fallback;

    expect(headline).toBeDefined();
    expect(list.runnersUp).toHaveLength(2);

    const ids = [headline!.model.id, ...list.runnersUp.map((c) => c.model.id)];
    expect(new Set(ids).size, 'the shortlist repeats a model').toBe(ids.length);
  });

  it('takes each model’s own best entry, not merely its first distinct one', () => {
    const list = recommend(sweep(RTX_5090));
    for (const runner of list.runnersUp) {
      const forModel = list.ranked.filter((c) => c.model.id === runner.model.id);
      expect(forModel[0]).toBe(runner);
    }
  });
});

describe('a spilled recommendation is marked as one', () => {
  it('carries the offload fraction, which is what the host-RAM qualifier keys on', () => {
    // `planPlacement` sizes a spill with no host-RAM input at all, so a shortlist row saying a
    // spilled configuration runs is promising something never checked. The surface owes it the
    // qualifier; this is the field it reads.
    const list = recommend(sweep(RTX_5090));
    const spilled = list.ranked.filter((c) => c.offloadFraction > 0);

    expect(spilled.length, 'nothing spilled, so the field is untested').toBeGreaterThan(0);
    for (const c of list.ranked) {
      expect(c.offloadFraction).toBeGreaterThanOrEqual(0);
      expect(c.offloadFraction).toBeLessThanOrEqual(1);
    }
  });
});

describe('the grade comes from the verdict layer, not from a second set of thresholds', () => {
  it('carries the verdict’s own sentence unrewritten', () => {
    const list = recommend(sweep(RTX_5090));
    for (const c of list.ranked.slice(0, 10)) {
      expect(c.reason.length, `${c.model.id} has no reason`).toBeGreaterThan(0);
      // Every verdict sentence names the bar it cleared or missed — that is the rule the whole
      // verdict layer is organised around, and a shortlist that summarised it would break it.
      expect(c.reason).toMatch(/\d/);
    }
  });

  it('agrees with the same configuration graded on its own', () => {
    // The point of going through `judgeWorkloads` rather than owning a copy of the thresholds: the
    // shortlist's grade and the Bench's grade for the same configuration have to be one answer.
    const list = recommend(sweep(RTX_5090));
    const top = list.best!;
    const again = recommend(
      sweep(RTX_5090, {
        models: [top.model],
        runtimes: [top.runtime],
        quantsFor: () => [getQuant(top.quant.id)],
      })
    );

    expect(again.ranked).toHaveLength(1);
    expect(again.ranked[0].fitness).toBe(top.fitness);
    expect(again.ranked[0].reason).toBe(top.reason);
  });
});
