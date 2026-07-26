import { useId, useMemo, useState } from 'react';
import type { Evaluation } from '@/engine';
import { judgeWorkloads, type Fitness } from '@/engine/verdict';
import { estimateConfig, type Config } from '@/store/config';

/**
 * What the setup can actually be used for.
 *
 * The number above this strip is the measurement; this is the answer. "3.2 tok/s" tells you
 * nothing unless you already know what a good number looks like — "fine for overnight batch,
 * unusable for a coding agent" is what someone deciding how to spend money needs.
 *
 * Each row carries an icon, a word and a written reason, so the grading never depends on colour.
 * Ordered by how tight the latency budget is — not by overall difficulty, and the two come apart
 * whenever capacity rather than speed is the binding constraint. Every row is graded at the
 * prompt that row really sends, so a row asking for less room can pass where one above it fails;
 * that is a real difference between the workloads, not an inconsistency, and the written reason
 * on each row is what explains it.
 */

const FITNESS: Record<Fitness, { icon: string; word: string; color: string }> = {
  good: { icon: '●', word: 'Yes', color: 'var(--color-good)' },
  tight: { icon: '◐', word: 'Tight', color: 'var(--color-warning)' },
  fail: { icon: '○', word: 'No', color: 'var(--color-critical)' },
};

export function Workloads({ evaluation, config }: { evaluation: Evaluation; config: Config }) {
  const headingId = useId();
  const [expanded, setExpanded] = useState(false);

  /**
   * Memoised on the scenario, which is the only thing the grades depend on.
   *
   * Without it, toggling the descriptions below re-graded all seven archetypes — a button that
   * changes nothing but which strings are rendered was paying for the whole verdict layer. The
   * dependencies are `evaluation` and `config`, both of which the Bench already holds stable
   * between renders that do not change the scenario.
   */
  const verdicts = useMemo(
    () =>
      judgeWorkloads({
        selectedPlacement: evaluation.placement,
        usage: {
          contextTokens: config.contextTokens,
          concurrency: config.concurrency,
          promptTokens: config.promptTokens,
          kvPrecision: config.kvPrecision,
        },
        maxContextTokens: evaluation.maxContextTokens,
        runnableContextTokens: evaluation.runnableContextTokens,
        /**
         * Graded at each archetype's own scenario, so this strip does not move when the sliders do
         * — a completion popup sends what it sends regardless of the current setting.
         *
         * Both context and prompt are raised, and decode is re-measured along with prefill. Raising
         * only the prompt left placement planned for the smaller slider context; re-running only
         * prefill left decode describing the smaller cache, so an agent could be graded on a rate
         * measured at 512 tokens while its own turn is 16K.
         *
         * `estimateConfig`, not `evaluateConfig`: the latter also computes `maxContextTokens` and
         * `runnableContextTokens`, and each of those is a binary search over the model's whole
         * context range. This callback discarded both, every time, for every archetype — roughly
         * forty `planPlacement` calls to use one, and on a layer-split rig each of those sorts the
         * model's layers.
         */
        evaluateAt: (promptTokens, contextTokens, cachedPrefixTokens) =>
          estimateConfig({ ...config, promptTokens, contextTokens }, cachedPrefixTokens),
      }),
    [evaluation, config]
  );

  const usable = verdicts.filter((v) => v.fitness !== 'fail').length;

  /**
   * When nothing can run, every row carries the same sentence — so it is said once, above the
   * list, and the rows keep only their status. Seven identical explanations read as seven
   * separate problems.
   */
  const sharedReason =
    verdicts.every((v) => v.fitness === 'fail') && new Set(verdicts.map((v) => v.reason)).size === 1
      ? verdicts[0].reason
      : undefined;

  return (
    <section aria-labelledby={headingId} className="panel p-5">
      <header className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <h2 id={headingId} className="text-sm font-semibold tracking-wide">
          What you could do with it
        </h2>
        <p className="text-sm whitespace-nowrap text-[var(--color-text-muted)]">
          <span className="tabular text-[var(--color-text)]">{usable}</span> of {verdicts.length}{' '}
          workloads
        </p>
      </header>

      {sharedReason && (
        <p className="mt-3 text-sm text-[var(--color-critical)]">
          <span aria-hidden="true">▲ </span>
          {sharedReason}
        </p>
      )}

      <ul className="mt-4 flex flex-col gap-2">
        {verdicts.map(({ workload, fitness, reason }) => {
          const style = FITNESS[fitness];
          return (
            <li
              key={workload.id}
              className="grid grid-cols-[auto_1fr] items-baseline gap-x-3 gap-y-0.5 sm:grid-cols-[9rem_auto_1fr]"
            >
              {/* Icon and word together, so the grading survives without colour. */}
              <span
                className="order-2 flex items-center gap-1.5 text-xs whitespace-nowrap sm:order-none"
                style={{ color: style.color }}
              >
                <span aria-hidden="true">{style.icon}</span>
                <span>{style.word}</span>
              </span>

              <span className="order-1 text-sm text-[var(--color-text)] sm:order-none">
                {workload.label}
              </span>

              <span className="order-3 col-span-2 text-xs leading-relaxed text-[var(--color-text-muted)] sm:order-none sm:col-span-1">
                {sharedReason
                  ? expanded
                    ? workload.description
                    : ''
                  : expanded
                    ? `${workload.description} ${reason}`
                    : reason}
              </span>
            </li>
          );
        })}
      </ul>

      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
        className="mt-4 text-xs text-[var(--color-accent)] underline underline-offset-2"
      >
        {expanded ? 'Hide' : 'Show'} what each workload means
      </button>
    </section>
  );
}
