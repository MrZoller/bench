import type { Evaluation } from '@/engine';
import type { StatusTone } from '@/design/tokens';
import { gibLabel, rate, seconds, tokens } from '@/lib/format';

/**
 * Three readouts, deliberately not one.
 *
 * Capacity, decode speed and time-to-first-token are independent axes, and the machines people
 * are actually choosing between sit at different corners: a Spark holds a model a 5090 cannot
 * and then decodes it three times slower; a Mac is the reverse of a Spark on prefill. Collapsing
 * these into a single "score" is precisely the move that makes existing calculators give bad
 * advice, so the layout refuses to do it.
 *
 * Each tile carries an icon and a word alongside its colour — a verdict must never be conveyed
 * by hue alone.
 */

const TONE_STYLE: Record<StatusTone, { color: string; icon: string; word: string }> = {
  good: { color: 'var(--color-good)', icon: '●', word: 'Comfortable' },
  warning: { color: 'var(--color-warning)', icon: '◐', word: 'Tight' },
  serious: { color: 'var(--color-serious)', icon: '◑', word: 'Marginal' },
  critical: { color: 'var(--color-critical)', icon: '▲', word: 'Will not run' },
};

interface Reading {
  key: string;
  label: string;
  value: string;
  unit: string;
  tone: StatusTone;
  /** Overrides the tone's default word when a more specific one is truer. */
  verdict?: string;
  detail: string;
}

function capacityReading(evaluation: Evaluation): Reading {
  const { placement } = evaluation;

  if (placement.unsupported) {
    return {
      key: 'capacity',
      label: 'Capacity',
      value: '—',
      unit: '',
      tone: 'critical',
      verdict: 'Unsupported',
      detail: placement.unsupported,
    };
  }

  const headroom = placement.headroomBytes;
  if (placement.impossible) {
    return {
      key: 'capacity',
      label: 'Capacity',
      value: gibLabel(-headroom),
      unit: 'over',
      tone: 'critical',
      detail:
        'Past the allocatable ceiling with nowhere to spill. Shared-memory machines have no faster tier to fall back from.',
    };
  }
  if (placement.offloadFraction > 0) {
    return {
      key: 'capacity',
      label: 'Capacity',
      value: gibLabel(-headroom),
      unit: 'offloaded',
      tone: 'serious',
      verdict: 'Spilling to RAM',
      detail:
        'Loads only if the host has RAM for the spilled part, which is not checked here. What does spill crosses the bus every token — usually the whole explanation for "why is it so slow".',
    };
  }

  const utilization = placement.utilization;
  return {
    key: 'capacity',
    label: 'Capacity',
    value: gibLabel(headroom),
    unit: 'free',
    tone: utilization > 0.9 ? 'warning' : 'good',
    verdict: utilization > 0.9 ? 'Tight' : 'Fits',
    detail:
      utilization > 0.9
        ? 'Fits, with little room to raise context or add a user.'
        : `Room to grow — ${tokens(evaluation.maxContextTokens)} context would still fit at this concurrency.`,
  };
}

function decodeReading(evaluation: Evaluation): Reading {
  const perUser = evaluation.decode.perUserTokensPerSec;

  // Thresholds are reading speed, not benchmarks: below ~10 tok/s a chat feels like waiting,
  // and above ~30 it outruns most people.
  const tone: StatusTone = perUser >= 30 ? 'good' : perUser >= 15 ? 'warning' : 'serious';
  return {
    key: 'decode',
    label: 'Decode',
    value: rate(perUser),
    unit: 'tok/s per user',
    tone,
    verdict: perUser >= 30 ? 'Fast' : perUser >= 15 ? 'Usable' : 'Slow',
    detail: evaluation.decode.kvBound
      ? 'KV traffic now outweighs weight traffic — at this context the cache, not the model, sets the speed.'
      : 'Bound by weight bandwidth. Lower quantization or faster memory is what moves this.',
  };
}

function prefillReading(evaluation: Evaluation): Reading {
  const { ttftSeconds, prefillTokensPerSec, attentionBound } = evaluation.prefill;
  const tone: StatusTone = ttftSeconds <= 2 ? 'good' : ttftSeconds <= 10 ? 'warning' : 'critical';

  return {
    key: 'prefill',
    label: 'Time to first token',
    value: seconds(ttftSeconds),
    unit: '',
    tone,
    verdict: ttftSeconds <= 2 ? 'Responsive' : ttftSeconds <= 10 ? 'Noticeable' : 'Slow start',
    detail: attentionBound
      ? `${rate(prefillTokensPerSec)} tok/s prompt processing. Quadratic attention now dominates the pass, so this degrades faster than linearly as the prompt grows.`
      : `${rate(prefillTokensPerSec)} tok/s prompt processing, bound by compute on the linear layers.`,
  };
}

export function Telemetry({ evaluation }: { evaluation: Evaluation }) {
  /**
   * A runtime that cannot drive this hardware has no throughput, so none is shown.
   *
   * The engine still returns arithmetic for the combination — it has no opinion about whether
   * the software exists — but rendering "28 tok/s, Usable" beside "vLLM does not run here" is a
   * plausible number for a thing that cannot happen, which is the exact failure the rest of this
   * project is built to avoid. The suppression lives here rather than in the engine, which stays
   * pure and unopinionated.
   */
  const { unsupported, impossible } = evaluation.placement;

  /**
   * Two distinct ways a configuration cannot run, and both must silence the speed tiles.
   *
   * `impossible` is the subtler one: past the ceiling with nowhere to spill, which is every
   * over-budget unified-memory and CPU-RAM config. There `offloadFraction` is 0, so decode
   * computes as though every weight were resident at full bandwidth — and paints a green
   * "Fast" beside a red "Will not run". The optimism is the danger, not the noise.
   */
  const blocked = unsupported ?? (impossible ? 'Past the ceiling with nowhere to spill.' : null);

  const readings: Reading[] = blocked
    ? [
        capacityReading(evaluation),
        ...(['Decode', 'Time to first token'] as const).map((label, i) => ({
          key: `blocked-${i}`,
          label,
          value: '—',
          unit: '',
          tone: 'critical' as const,
          verdict: unsupported ? 'Unsupported' : 'Will not run',
          detail: unsupported
            ? 'No estimate — this runtime cannot drive this hardware.'
            : 'No estimate — the model does not fit, so there is no speed to report.',
        })),
      ]
    : [capacityReading(evaluation), decodeReading(evaluation), prefillReading(evaluation)];

  return (
    <section aria-label="Verdicts" className="grid gap-3 sm:grid-cols-3">
      {readings.map((reading) => {
        const tone = TONE_STYLE[reading.tone];
        return (
          <article key={reading.key} className="panel flex flex-col gap-1 p-4">
            <h3 className="text-xs font-medium tracking-wide text-[var(--color-text-faint)] uppercase">
              {reading.label}
            </h3>

            <p className="tabular text-2xl leading-tight text-[var(--color-text)]">
              {reading.value}
              {reading.unit && (
                <span className="ml-1 text-sm text-[var(--color-text-faint)]">{reading.unit}</span>
              )}
            </p>

            {/* Icon + word + colour. Never colour alone. */}
            <p className="flex items-center gap-1.5 text-sm" style={{ color: tone.color }}>
              <span aria-hidden="true">{tone.icon}</span>
              <span>{reading.verdict ?? tone.word}</span>
            </p>

            <p className="mt-1 text-xs leading-relaxed text-[var(--color-text-muted)]">
              {reading.detail}
            </p>
          </article>
        );
      })}
    </section>
  );
}
