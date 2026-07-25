import { useId, useState } from 'react';
import type { Evaluation } from '@/engine';
import { judgeWorkloads, type Fitness } from '@/engine/verdict';
import { evaluateConfig, type Config } from '@/store/config';

/**
 * What the setup can actually be used for.
 *
 * The number above this strip is the measurement; this is the answer. "3.2 tok/s" tells you
 * nothing unless you already know what a good number looks like — "fine for overnight batch,
 * unusable for a coding agent" is what someone deciding how to spend money needs.
 *
 * Each row carries an icon, a word and a written reason, so the grading never depends on colour.
 * Ordered by how tight the latency budget is, which makes the list read as a ladder: whatever
 * passes near the top implies everything below it.
 */

const FITNESS: Record<Fitness, { icon: string; word: string; color: string }> = {
  good: { icon: '●', word: 'Yes', color: 'var(--color-good)' },
  tight: { icon: '◐', word: 'Tight', color: 'var(--color-warning)' },
  fail: { icon: '○', word: 'No', color: 'var(--color-critical)' },
};

export function Workloads({ evaluation, config }: { evaluation: Evaluation; config: Config }) {
  const headingId = useId();
  const [expanded, setExpanded] = useState(false);

  const verdicts = judgeWorkloads({
    placement: evaluation.placement,
    decode: evaluation.decode,
    usage: {
      contextTokens: config.contextTokens,
      concurrency: config.concurrency,
      promptTokens: config.promptTokens,
      kvPrecision: config.kvPrecision,
    },
    maxContextTokens: evaluation.maxContextTokens,
    // Graded at each archetype's own prompt, so this strip does not move when the prompt
    // slider does — a completion popup sends what it sends regardless of the current setting.
    prefillAt: (promptTokens) => evaluateConfig({ ...config, promptTokens }).prefill,
  });

  const usable = verdicts.filter((v) => v.fitness !== 'fail').length;

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
                {expanded ? `${workload.description} ${reason}` : reason}
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
