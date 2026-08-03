import { describe, expect, it } from 'vitest';
import { recommend, type Candidate, type RecommendInputs } from './recommend';
import { planPlacement } from './placement';
import { estimateScenario } from './index';
import { WORKLOADS, WORKLOAD_BARS } from './verdict';
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

    /**
     * By decode rate, per `FALLBACK_RULE` — deliberately not `ranked[0]`, which is the *largest*
     * thing that loads. A 671B decoding at 0.3 tok/s is not more useful than an 8B at 40 that
     * merely missed a threshold, and the two picks answer different questions.
     *
     * **And it is drawn from every loadable configuration, not from `ranked`**, which the first
     * version asserted and which was the defect: `bestQuant` reduces each model × runtime pair to
     * its widest best-grading format, so the fastest member of `ranked` is the fastest survivor of
     * a pruning that ranked on *width*. The fastest thing that loads is routinely a narrower quant
     * of a pairing whose widest one won that reduction.
     */
    const fastestRanked = Math.max(...list.ranked.map((c) => c.tokensPerSec));
    expect(list.fallback!.tokensPerSec).toBeGreaterThanOrEqual(fastestRanked);

    // The evidence half, which is what makes the assertion above more than `>=` on itself: on this
    // machine the fallback really is a configuration the ranked list does not contain.
    expect(
      list.fallback!.tokensPerSec,
      'the fallback is the fastest of the pruned list, so the fix is untested here'
    ).toBeGreaterThan(fastestRanked);
  });

  /**
   * And it ranks by a rate that was actually measured somewhere (#172).
   *
   * `FALLBACK_RULE` promises the fastest configuration that loads, and the rate deciding that used
   * to come from the archetype's own turn — ~16.5K tokens for an agent — while the verdict layer had
   * evaluated the candidate at a 32K or 64K session. Decode slows as the cache grows, so the two are
   * different numbers on the same row, and the ordering was over one the reader is never shown and
   * no tier ever graded.
   */
  it('ranks by the rate at the tier the candidate was graded at, not at the archetype’s turn', () => {
    const list = recommend(sweep(small, { workloadId: 'agent' }));
    const fallback = list.fallback;
    expect(fallback).toBeDefined();

    const rateAt = (contextTokens: number, promptTokens: number) =>
      estimateScenario({
        model: fallback!.model,
        quant: fallback!.quant,
        runtime: fallback!.runtime,
        usage: { contextTokens, promptTokens, concurrency: 1, kvPrecision: 'fp16' },
        rig: { device: small, count: 1 },
      }).decode.perUserTokensPerSec;

    // The scenario the candidate carries is the one its rate was measured at — the agent's 64K
    // session here, since this machine can plan it.
    expect(fallback!.contextTokens).toBe(WORKLOAD_BARS.agent.good.session);
    expect(fallback!.tokensPerSec).toBe(rateAt(fallback!.contextTokens, fallback!.promptTokens));

    /**
     * The falsifiable half: the archetype's own turn is a *different* rate on this configuration —
     * 11.6 tok/s against 6.7 — so an ordering taken there is an ordering over a speed no tier
     * measured. Asserted as a direction rather than a band, because it is one: a smaller window
     * decodes at least as fast, which `verdict.test.ts` pins over the whole catalog.
     */
    const turn = Math.min(fallback!.model.maxContext, 16384 + 512);
    expect(
      rateAt(turn, 16384),
      'the turn and the session measure the same rate here, so this asserts nothing'
    ).toBeGreaterThan(fallback!.tokensPerSec);
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

/**
 * The sweep asks the verdict layer about tiers (#170).
 *
 * `grade()` used to plan exactly one placement per candidate, at the archetype's own scenario, and
 * drop the candidate when that placement was `impossible`. Several archetypes are *graded* at
 * working sizes that scenario never names — long-context's tight tier is a 64K prompt against its
 * 128K job — so the tier that would have accepted a machine never ran.
 *
 * The answer was not wrong, and that is worth stating precisely: `judgeWorkloads` refuses at the top
 * when the selected placement is impossible, so dropping the candidate and grading it `fail` say the
 * same thing. What neither says is the useful one. A reader asking what this machine can do for long
 * context was told nothing rather than "this one, at half the window".
 */
describe('a tier is reached at the scenario the tier is about', () => {
  /**
   * The placement the sweep plans for one of long-context's two tier prompts — the prompt plus room
   * to answer, capped by the model, which is the window `gradedScenarios` states for it.
   *
   * `512` restated rather than imported, like every other test that needs it: the allowance is
   * `verdict.ts`'s internal convention and this fails if the two ever disagree.
   */
  const placementAt = (c: Candidate, device: DeviceSpec, promptTokens: number) =>
    planPlacement(
      c.model,
      c.quant,
      {
        contextTokens: Math.min(c.model.maxContext, promptTokens + 512),
        concurrency: 1,
        promptTokens,
        kvPrecision: 'fp16',
      },
      { device, count: 1 },
      c.runtime
    );

  it('keeps a long-context candidate whose reduced tier admits the machine', () => {
    const list = recommend(sweep(RTX_5090, { workloadId: 'long-context' }));

    /**
     * The case the issue names, found rather than assumed: configurations sized *between* the two
     * tier prompts, where the 128K request cannot be placed at all and the 64K one can. On a 5090
     * the shipped catalog has a dozen — the large MoEs and the 70Bs, whose cache at 128K is over
     * the card's ceiling however much of the weights spill.
     */
    const halfWindow = list.ranked.filter(
      (c) =>
        placementAt(c, RTX_5090, 131072).impossible && !placementAt(c, RTX_5090, 65536).impossible
    );

    expect(
      halfWindow.length,
      'no configuration sits between the two tier prompts, so this asserts nothing'
    ).toBeGreaterThan(0);

    // Present rather than absent, and *graded* rather than merely present: the tight tier is what
    // accepts these machines, and it is the tier the reader came for.
    const tight = halfWindow.filter((c) => c.fitness === 'tight');
    expect(
      tight.length,
      'every candidate between the tiers failed, so no tier spoke'
    ).toBeGreaterThan(0);

    for (const c of tight) {
      // And the sentence describes the job the machine does, not the one it cannot: the reduced
      // tier is a smaller job, so the reason quotes the 64K it reads rather than the 128K it holds
      // nowhere. Timing the full request here is the defect `verdict.ts` fixed one layer down.
      expect(c.reason, `${c.model.id} ${c.quant.id}`).toMatch(/\b64K\b/);
    }
  });

  it('carries that scenario, for every archetype, and it is one the machine can place', () => {
    /**
     * The row is a deep link — clicking it loads the configuration into the Bench at the scenario
     * it was graded at. Rebuilding the archetype's own request there would send a candidate earned
     * at a tier's reduced window to a placement the Bench cannot make, and the verdict strip under
     * it would read `No` for the very workload the reader picked. That is #167's defect arriving
     * through the door #170 opened, so the scenario travels with the candidate and this is what
     * says it is a real one.
     */
    for (const workload of WORKLOADS) {
      const list = recommend(sweep(RTX_5090, { workloadId: workload.id }));
      expect(list.ranked.length, workload.id).toBeGreaterThan(0);

      for (const c of list.ranked) {
        const placement = planPlacement(
          c.model,
          c.quant,
          {
            contextTokens: c.contextTokens,
            concurrency: 1,
            promptTokens: c.promptTokens,
            kvPrecision: 'fp16',
          },
          { device: RTX_5090, count: 1 },
          c.runtime
        );

        const where = `${workload.id}: ${c.model.id} ${c.quant.id} ${c.runtime.id}`;
        expect(placement.impossible, where).toBe(false);
        expect(c.contextTokens, where).toBeLessThanOrEqual(c.model.maxContext);
        // The prompt is part of the window, never larger than it.
        expect(c.promptTokens, where).toBeLessThanOrEqual(c.contextTokens);
      }
    }
  });
});

/**
 * And every figure beside it describes *that* tier (#172).
 *
 * The same root as #170 one step further on: the sweep models one scenario per candidate and the
 * verdict layer models a tier structure, so wherever the two disagree a figure ends up attached to a
 * configuration the reader is not being recommended. The three the issue names are a `tight` row
 * carrying the `good` tier's spill caveat, a fallback ranked by a rate no tier measured — pinned
 * above, beside the rule it belongs to — and a footer naming the reader's user count over a serving
 * list graded at four users and two.
 */
describe('a figure describes the tier that earned it', () => {
  /** The placement at one stated window, which is what a spill caveat is read from. */
  const spillAt = (c: Candidate, device: DeviceSpec, contextTokens: number) =>
    planPlacement(
      c.model,
      c.quant,
      { contextTokens, concurrency: 1, promptTokens: c.promptTokens, kvPrecision: 'fp16' },
      { device, count: 1 },
      c.runtime
    );

  it('takes a tight agent’s spill from the 32K session it earned, not the 64K one', () => {
    let checked = 0;
    let widerDiffers = 0;
    let widerWouldCaveat = 0;

    /**
     * Three cards rather than one, and mid-range ones: the rows this is about are configurations
     * that hold a 32K session and not a 64K one, which on a 5090 is a handful and on a 5070 is most
     * of the interesting catalog.
     */
    for (const deviceId of ['rtx-5090', 'rtx-5080', 'rtx-5070']) {
      const device = getDevice(deviceId);
      const list = recommend(sweep(device, { workloadId: 'agent' }));
      const tight = list.ranked.filter(
        (c) => c.fitness === 'tight' && c.contextTokens === WORKLOAD_BARS.agent.tight.session
      );

      for (const c of tight) {
        checked += 1;
        const where = `${deviceId}: ${c.model.id} ${c.quant.id} ${c.runtime.id}`;
        // The claim: the caveat is read from the session this candidate was graded at.
        expect(c.offloadFraction, where).toBe(
          spillAt(c, device, WORKLOAD_BARS.agent.tight.session).offloadFraction
        );

        /**
         * And the reading it is not: the `good` tier's session capped by the model, which is what
         * the stopgap `Math.max` took. Capped, because that is what made it worse rather than
         * merely wider — on a 40,960-token model it read a window no tier states at all.
         */
        const wider = spillAt(
          c,
          device,
          Math.min(c.model.maxContext, WORKLOAD_BARS.agent.good.session)
        );
        if (wider.offloadFraction !== c.offloadFraction) widerDiffers += 1;
        if (c.offloadFraction === 0 && wider.offloadFraction > 0) widerWouldCaveat += 1;
      }
    }

    expect(
      checked,
      'nothing was tight at the reduced session, so this asserts nothing'
    ).toBeGreaterThan(0);
    expect(
      widerDiffers,
      'the two readings agree on every row, so the tier the figure comes from is untested'
    ).toBeGreaterThan(0);
    /**
     * The half that is the reader's problem rather than a number's. These rows keep every weight
     * resident at the session they were graded at and spill at one they were not, so the wider
     * reading put "Runs only by spilling weights to host RAM" — with its unchecked host-RAM
     * qualifier — on a recommendation that does no such thing.
     */
    expect(
      widerWouldCaveat,
      'no row gains or loses the caveat between the two readings, so the caveat is untested'
    ).toBeGreaterThan(0);
  });

  it('measures every candidate’s rate at the scenario it carries', () => {
    // The general form of the fallback's rule: a row's figures and its grade come from one
    // scenario, so a rate printed or ranked on is one that was really measured there.
    for (const workloadId of ['agent', 'long-context', 'serving']) {
      const list = recommend(sweep(RTX_5090, { workloadId }));
      expect(list.ranked.length, workloadId).toBeGreaterThan(0);

      for (const c of list.ranked) {
        const at = estimateScenario({
          model: c.model,
          quant: c.quant,
          runtime: c.runtime,
          usage: {
            contextTokens: c.contextTokens,
            promptTokens: c.promptTokens,
            concurrency: 1,
            kvPrecision: 'fp16',
          },
          rig: { device: RTX_5090, count: 1 },
        });
        const where = `${workloadId}: ${c.model.id} ${c.quant.id} ${c.runtime.id}`;
        expect(c.tokensPerSec, where).toBe(at.decode.perUserTokensPerSec);
        expect(c.ttftSeconds, where).toBe(at.prefill.ttftSeconds);
        expect(c.offloadFraction, where).toBe(at.placement.offloadFraction);
      }
    }
  });

  /**
   * The third finding, which is a caption rather than a number.
   *
   * Serving is the one archetype whose *subject* is user count, and its tiers declare four and two —
   * so a footer printing the reader's own setting under that list stated a count no grade in it
   * used. The panel reads this field; `Recommend.test.tsx` asserts the sentence.
   */
  it('states the user counts serving is graded at, rather than the reader’s', () => {
    const list = recommend(sweep(RTX_5090, { workloadId: 'serving', concurrency: 12 }));

    expect(list.declaredConcurrency).toEqual([
      WORKLOAD_BARS.serving.good.users,
      WORKLOAD_BARS.serving.tight.users,
    ]);
    // Not decoration: the sentence beside the row names the good tier's own count, at a reader
    // setting three times it.
    expect(list.best!.reason).toContain(`${WORKLOAD_BARS.serving.good.users} users`);
  });

  it('grades a serving row the same whatever the reader’s user count says', () => {
    /**
     * Which is what makes printing that count a misstatement rather than a redundancy. One pairing
     * at one format, so the comparison is of a grade rather than of two shortlists that may not
     * contain the same rows — the sweep still *plans* at the reader's count, so a high one changes
     * what loads at all.
     */
    const top = recommend(sweep(RTX_5090, { workloadId: 'serving' })).best!;
    const alone = (concurrency: number) =>
      recommend(
        sweep(RTX_5090, {
          workloadId: 'serving',
          concurrency,
          models: [top.model],
          runtimes: [top.runtime],
          quantsFor: () => [getQuant(top.quant.id)],
        })
      ).ranked;

    const one = alone(1);
    const twelve = alone(12);
    expect(one).toHaveLength(1);
    expect(twelve, 'the pairing stopped loading, so the grades are not comparable').toHaveLength(1);
    expect(twelve[0].fitness).toBe(one[0].fitness);
    expect(twelve[0].reason).toBe(one[0].reason);
  });

  it('declares nothing for the six archetypes that inherit the reader’s count', () => {
    // The footer's other branch, and the reason the field is empty rather than `[concurrency]`:
    // these six really are graded at whatever the reader set, and the caption says so.
    for (const workload of WORKLOADS.filter((w) => w.id !== 'serving')) {
      expect(
        recommend(sweep(RTX_5090, { workloadId: workload.id })).declaredConcurrency,
        workload.id
      ).toEqual([]);
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
