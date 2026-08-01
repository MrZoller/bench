import { useId, useMemo, useState } from 'react';
import {
  FALLBACK_RULE,
  QUANT_RULE,
  RANKING_RULE,
  recommend,
  type Candidate,
} from '@/engine/recommend';
import { WORKLOADS, type Fitness } from '@/engine/verdict';
import { MODELS, useConfig, type Config } from '@/store/config';
import { getDevice } from '@/data/catalog';
import { QUANTS } from '@/data/quants';
import { RUNTIMES, substitutionFor } from '@/data/runtimes';
import { quantApplies } from '@/lib/quantChoice';
import { HOST_RAM_UNCHECKED } from '@/lib/verdicts';
import { kvLabel } from '@/lib/stops';
import { params } from '@/lib/format';
import { Select } from './Controls';
import { DETAIL_ANCHOR_ID } from './Matrix';

/**
 * What this machine should run (#138).
 *
 * The question people arrive with — *"I want a local coding assistant, what is the best model I
 * can run?"* — asked of the catalog rather than of one configuration. The Matrix holds this answer
 * as 1,470 cells to interpret; this returns the decision, and every row deep-links back into the
 * Bench so "why that verdict" is one click and the explain layer does the explaining.
 *
 * **Three things about it are deliberate and would each be easy to get wrong.**
 *
 * *The rules are printed.* A ranked list is a recommendation, and a recommendation with an unstated
 * basis is an opinion wearing the chassis of a measurement. `RANKING_RULE`, `QUANT_RULE` and — when
 * it fires — `FALLBACK_RULE` are rendered under the shortlist, verbatim from the engine, because a
 * near-copy here is how the printed rule and the implemented one come apart.
 *
 * *The axes are named.* The sweep fixes device count and KV precision, and those are the reader's
 * current settings rather than universal truths — so the footer says which machine and which cache
 * the list describes. Without that, a shortlist read after moving a slider is a claim about a
 * configuration nobody is looking at.
 *
 * *A spilled row carries the host-RAM qualifier*, like every other surface that says a spilled
 * configuration runs. `planPlacement` sizes a spill with no host-RAM input at all, and this is the
 * surface most read as a shortlist — which is exactly the position the Matrix was in when #127 was
 * filed against it.
 */

const FITNESS: Record<Fitness, { icon: string; word: string; color: string }> = {
  good: { icon: '●', word: 'Yes', color: 'var(--color-good)' },
  tight: { icon: '◐', word: 'Tight', color: 'var(--color-warning)' },
  fail: { icon: '○', word: 'No', color: 'var(--color-critical)' },
};

/**
 * The archetype the shortlist optimises for, as local state rather than in the scenario.
 *
 * It is a question about this panel — "rank them for *what*" — and not a property of the
 * configuration the rest of the page describes, so putting it in `Config` would write it into every
 * shared link and make two scenarios differ that are the same scenario. The Matrix's measure toggle
 * is the same kind of control for the same reason.
 */
const DEFAULT_WORKLOAD = 'chat';

export function Recommend() {
  const headingId = useId();
  const config = useConfig();
  const replace = useConfig((s) => s.replace);
  const [workloadId, setWorkloadId] = useState(DEFAULT_WORKLOAD);

  const device = getDevice(config.deviceId);

  /**
   * Memoised on the four inputs the sweep actually reads, not on the whole config.
   *
   * The sweep is ~20ms over the shipped catalog — cheap for a selection change and not for a slider
   * frame — and context, prompt and concurrency are none of its business: every candidate is graded
   * at its archetype's own scenario, which is the point. Keying on `config` would re-run the whole
   * sweep on every drag of a slider that cannot change the answer.
   */
  const shortlist = useMemo(
    () =>
      recommend({
        device,
        deviceCount: config.deviceCount,
        kvPrecision: config.kvPrecision,
        concurrency: config.concurrency,
        workloadId,
        models: MODELS,
        runtimes: RUNTIMES,
        quantsFor: (model, runtime) =>
          QUANTS.filter((q) => quantApplies(q, model, device, runtime)),
      }),
    [device, config.deviceCount, config.kvPrecision, config.concurrency, workloadId]
  );

  const headline = shortlist.best ?? shortlist.fallback;

  /**
   * Load a row into the Bench, then scroll to where its figures appear.
   *
   * `replace` rather than three `set` calls: `coerce` runs per call, so setting the model before
   * the runtime can bounce the quant off `quantApplies` in between and land on the fallback format
   * — the shortlist would name one thing and the Bench would load another. One transition, one
   * coercion.
   */
  const load = (candidate: Candidate) => {
    replace({
      ...config,
      modelId: candidate.model.id,
      quantId: candidate.quant.id,
      runtimeId: candidate.runtime.id,
    } as Partial<Config>);
    // Optional on the method as well as the element, and the motion preference read here rather
    // than left to CSS — both for the reasons the Matrix's own cell handler records at length.
    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches;
    document.getElementById(DETAIL_ANCHOR_ID)?.scrollIntoView?.({
      behavior: reduce ? 'auto' : 'smooth',
      block: 'start',
    });
  };

  return (
    <section aria-labelledby={headingId} className="panel p-[min(1.25rem,5vw)]">
      <header className="mb-4 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 id={headingId} className="text-sm font-medium text-[var(--color-text)]">
            What this machine should run
          </h2>
          <p className="mt-1 text-xs text-[var(--color-text-muted)]">
            Every model in the catalog, at every format and runtime {device.name} can use — ranked.
          </p>
        </div>
        <Select
          label="Ranked for"
          value={workloadId}
          onChange={setWorkloadId}
          options={WORKLOADS.map((w) => ({ value: w.id, label: w.label, note: w.description }))}
        />
      </header>

      {headline === undefined ? (
        /* Not an empty panel: "nothing in the catalog loads here" is a real answer and a rare one,
           and it is different from "the sweep found nothing", which would be a bug. The count says
           which. */
        <p className="text-xs text-[var(--color-text-muted)]">
          Nothing in the catalog loads on {device.name} at this device count and cache precision.
          {shortlist.pairsConsidered} model and runtime pairings were considered.
        </p>
      ) : (
        <ol className="flex list-none flex-col gap-3">
          <Row candidate={headline} onLoad={load} headline kvPrecision={config.kvPrecision} />
          {shortlist.runnersUp.map((c) => (
            <Row
              key={`${c.model.id}:${c.quant.id}:${c.runtime.id}`}
              candidate={c}
              onLoad={load}
              kvPrecision={config.kvPrecision}
            />
          ))}
        </ol>
      )}

      {/* The rules, verbatim from the engine. Not behind a disclosure and not paraphrased: an
          unstated tie-break is the difference between a measurement and an opinion, and this panel
          is the one that most looks like the former. */}
      <div className="mt-4 flex flex-col gap-1 border-t border-[var(--color-border)] pt-3 text-[0.625rem] leading-relaxed text-[var(--color-text-muted)]">
        <p>{RANKING_RULE}</p>
        <p>{QUANT_RULE}</p>
        {shortlist.fallback !== undefined && <p>{FALLBACK_RULE}</p>}
        {/* Which machine, at what width — because the two axes the sweep fixes are the reader's
            current settings and not universal facts, and a shortlist read after a change is
            otherwise a claim about a configuration nobody is looking at. */}
        <p>
          Swept {shortlist.pairsConsidered} model and runtime pairings on{' '}
          {config.deviceCount > 1 ? `${config.deviceCount}x ` : ''}
          {device.name} with a {kvLabel(RUNTIMES[0], config.kvPrecision)} cache at{' '}
          {config.concurrency === 1 ? 'one user' : `${config.concurrency} users`}, each graded at
          the prompt its own workload sends. Change the hardware, the cache or the user count above
          and the list moves.
        </p>
      </div>
    </section>
  );
}

function Row({
  candidate,
  onLoad,
  headline = false,
  kvPrecision,
}: {
  candidate: Candidate;
  onLoad: (candidate: Candidate) => void;
  headline?: boolean;
  kvPrecision: Config['kvPrecision'];
}) {
  const { model, quant, runtime, fitness, reason } = candidate;
  const mark = FITNESS[fitness];
  const substitution = substitutionFor(runtime, quant.id);

  return (
    <li>
      <button
        type="button"
        onClick={() => onLoad(candidate)}
        /* The whole row is the target, which is what makes the deep link discoverable — a small
           "load" link at the end of a row of prose is a target most people never find. `text-left`
           because a button centres its content by default and this one is a paragraph. */
        className={`flex w-full flex-col gap-1 rounded-md border p-3 text-left hover:border-[var(--color-accent-dim)] ${
          headline ? 'border-[var(--color-accent-dim)]' : 'border-[var(--color-border)]'
        }`}
      >
        <span className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
          <span aria-hidden="true" style={{ color: mark.color }}>
            {mark.icon}
          </span>
          {/* The word, not only the colour — the same rule the workload strip follows, and the
              reason it exists: a grade carried by hue alone is not a grade for everyone. */}
          <span className="sr-only">{mark.word}: </span>
          <span className="text-sm font-medium text-[var(--color-text)]">{model.name}</span>
          <span className="text-xs text-[var(--color-text-muted)]">
            {params(model.totalParams)} · {quant.label} · {runtime.label}
          </span>
        </span>

        {/**
         * The verdict's own sentence, and **nothing beside it**.
         *
         * This row first printed the rate and the time-to-first-token after the reason, and the
         * result was "39 tok/s, 0.6s to first token on a short message. 39 tok/s, 557 ms to first
         * token." — one figure twice, at two roundings, from two scenarios. The verdict layer
         * measures at the archetype's own prompt and this panel's `Candidate` carries figures from
         * its own sweep, so the two are not merely redundant: they can disagree, which is the
         * failure this codebase keeps writing down about a limit stated twice.
         *
         * The reason already names the bar and the figure that decides it — that is the rule the
         * whole verdict layer is organised around — so it is the one sentence here.
         */}
        <span className="text-xs leading-relaxed text-[var(--color-text-muted)]">{reason}</span>

        {/* Same qualifier, same condition, same constant as every other surface that says a spilled
            configuration runs — and this is the one most read as a shortlist, which is the position
            the Matrix was in when #127 was filed against it. */}
        {candidate.offloadFraction > 0 && (
          <span className="text-xs leading-relaxed text-[var(--color-warning)]">
            Runs only by spilling weights to host RAM. {HOST_RAM_UNCHECKED}
          </span>
        )}

        {/* And the stand-in marker, for the same reason it appears on every other figure derived
            from a format the runtime cannot load (#18). A recommendation is the worst place for an
            unmarked approximation, because it is the one a reader acts on. */}
        {substitution !== undefined && (
          <span className="text-xs leading-relaxed text-[var(--color-warning)]">
            ◐ {substitution}
          </span>
        )}

        <span className="text-[0.625rem] text-[var(--color-text-faint)]">
          Load this into the bench above — {kvLabel(runtime, kvPrecision)} cache, its own numbers.
        </span>
      </button>
    </li>
  );
}
