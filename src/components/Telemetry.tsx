import { useId } from 'react';
import type { Evaluation } from '@/engine';
import type { StatusTone } from '@/design/tokens';
import { CAPACITY_TIGHT, HOST_RAM_UNCHECKED, classifyDecode, classifyTtft } from '@/lib/verdicts';
import { gibLabel, percent, rate, tokens } from '@/lib/format';

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

function capacityReading(
  evaluation: Evaluation,
  canOffload: boolean,
  tunableCeiling: boolean
): Reading {
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
      /**
       * A discrete GPU reaches `impossible` by a different route than a Mac does: not because
       * there is nowhere to spill, but because KV and activations alone overflow the card, and
       * those cannot be offloaded at all. Explaining that as "shared memory has no faster tier"
       * misdiagnoses a high-context 5090 as a Mac.
       *
       * "a card in this rig", not "the card": under a layer split the cards hold different amounts,
       * and the one whose cache overflows need not be the one whose overage is printed above. No
       * figure is quoted here so this could not contradict itself the way BudgetBar's could, but a
       * definite singular attaches the claim to whichever card the reader has in mind — which is
       * the one the bar a few pixels up just drew.
       */
      detail: canOffload
        ? 'The cache and workspace alone overflow a card in this rig, and those cannot be offloaded. Lower the context, the concurrency, or the KV precision.'
        : tunableCeiling
          ? // The ceiling is a default, not a hardware limit: macOS caps wired GPU memory near
            // 75% and AMD exposes a Variable Graphics Memory setting. Reporting a flat "will
            // not run" hides the one thing that would actually fix it.
            'Past the default allocation ceiling — but this machine lets you raise it, and the catalog figure is the untuned default rather than a hardware limit.'
          : 'Past the allocatable ceiling with nowhere to spill — a shared-memory machine has no faster tier to fall back from.',
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
      // The qualifier is shared with the Envelope's legend — see `HOST_RAM_UNCHECKED`. This tile is
      // where someone goes to ask why the thing is slow, so it earns the longer tail; the legend
      // swatch takes a shorter one. What must not differ is the fact.
      detail: `${HOST_RAM_UNCHECKED} What does spill crosses the bus every token — usually the whole explanation for "why is it so slow".`,
    };
  }

  const utilization = placement.utilization;
  return {
    key: 'capacity',
    label: 'Capacity',
    value: gibLabel(headroom),
    unit: 'free',
    tone: utilization > CAPACITY_TIGHT ? 'warning' : 'good',
    verdict: utilization > CAPACITY_TIGHT ? 'Tight' : 'Fits',
    detail:
      utilization > CAPACITY_TIGHT
        ? 'Fits, with little room to raise context or add a user.'
        : // At the model's own ceiling the spare memory is real but the growth is not: Qwen3 4B
          // on a 5090 sits at 40,960 with plenty free, and "40K would still fit" reads as an
          // invitation to raise a slider that has nothing left to give. Headroom is only room to
          // *grow* while there is somewhere to grow to.
          evaluation.maxContextTokens > evaluation.contextTokens
          ? `Room to grow — ${tokens(evaluation.maxContextTokens)} context would still fit at this concurrency.`
          : `Comfortable at ${tokens(evaluation.contextTokens)}, which is as far as this model goes.`,
  };
}

function decodeReading(evaluation: Evaluation): Reading {
  const { shown, word, tone } = classifyDecode(evaluation.decode.perUserTokensPerSec);
  const { kvBound, offloadPenalty } = evaluation.decode;

  return {
    key: 'decode',
    label: 'Decode',
    value: shown,
    unit: 'tok/s per user',
    tone,
    verdict: word,
    // `kvBound` is the engine's own comparison of weight seconds against cache seconds, so it
    // outranks the mere *existence* of a spill. Testing the spill first meant a 0.08% offload
    // blamed the host bus while the cache was costing six times as much time — sending someone
    // to fix the wrong thing, which is the error this whole tile exists to avoid.
    detail: kvBound
      ? 'KV traffic now costs more time per step than the weights — at this context the cache, not the model, sets the speed.'
      : offloadPenalty
        ? `Weights crossing the host bus set the pace — ${percent(offloadPenalty.fraction)} of them spill every token.`
        : 'Bound by weight bandwidth. Lower quantization or faster memory is what moves this.',
  };
}

function prefillReading(evaluation: Evaluation): Reading {
  const {
    ttftSeconds,
    prefillTokensPerSec,
    linearSeconds,
    attentionSeconds,
    offloadPenalty,
    concurrencyPenalty,
  } = evaluation.prefill;

  /**
   * The rate is machine-wide; the wait is per user. Both have to be labelled or neither can be
   * checked against the other.
   *
   * `prefillTokensPerSec` covers every prompt in the pass, so at 32 users it stays put while
   * `ttftSeconds` grows 32x — divide one into the other and the arithmetic is off by the batch.
   * The decode tile immediately to the left says "per user" in as many words, which primes exactly
   * the wrong reading of this one.
   */
  const across =
    concurrencyPenalty === undefined
      ? ''
      : ` across all ${concurrencyPenalty.prompts} prompts in flight`;

  /**
   * Classified on the displayed figure, exactly as decode is. `seconds()` rounds, so 10.27s
   * prints as "10 s" and, judged raw, is labelled "Slow start" against a visible threshold of
   * 10 — the same disagreement the decode tile had, in the function next door. Fixing one and
   * not the other is how it survived a round.
   */
  const { shown, word, tone } = classifyTtft(ttftSeconds);

  /**
   * The largest of the three terms wins — a strict maximum, not a majority.
   *
   * "More than everything else put together" was the old test, and it cannot identify a largest
   * of three. A pass split 40% streaming / 35% attention / 25% linear has streaming as the clear
   * maximum and the old test still failed it, handing the verdict to attention — a term costing
   * a seventh less. Anything short of a majority was unattributable, which is most real splits.
   */
  const streaming = offloadPenalty?.streamingSeconds ?? 0;
  const largest = Math.max(streaming, attentionSeconds, linearSeconds);

  return {
    key: 'prefill',
    label: 'Time to first token',
    value: shown,
    unit: '',
    tone,
    verdict: word,
    detail:
      largest === streaming && offloadPenalty !== undefined
        ? `${rate(prefillTokensPerSec)} tok/s prompt processing${across}, dominated by streaming ${percent(
            offloadPenalty.fraction
          )} of the weights across the host bus before the prompt can start.`
        : largest === attentionSeconds
          ? `${rate(prefillTokensPerSec)} tok/s prompt processing${across}. Quadratic attention now dominates the pass, so this degrades faster than linearly as the prompt grows.`
          : `${rate(prefillTokensPerSec)} tok/s prompt processing${across}, bound by compute on the linear layers.`,
  };
}

export function Telemetry({
  evaluation,
  canOffload,
  tunableCeiling,
}: {
  evaluation: Evaluation;
  /** True for discrete GPUs, the only class with a slower tier to spill to. */
  canOffload: boolean;
  /**
   * True when raising the allocation ceiling could actually make this fit: the ceiling is
   * user-raiseable *and* the model is inside the machine's physical capacity. DeepSeek V3 at
   * BF16 wants 1,253 GiB on a 512 GiB Mac, and no sysctl fixes that.
   */
  tunableCeiling: boolean;
}) {
  /**
   * The strip's own heading, and the fix for a claim this layout was built to refuse (#74).
   *
   * Each tile is an `h3` and the section carried an `aria-label`, so the nearest `h2` above them was
   * the *memory budget's* — and the outline read `h1 bench → h2 Memory budget → h3 Capacity`, which
   * says capacity is a subsection of the budget. It is not. The docstring at the top of this file is
   * explicit that these are three independent axes and that collapsing them into one score is the
   * move this panel exists to avoid; the outline collapsed two of them under the third's neighbour.
   *
   * So the `h2` is here to *re-parent* the three, not to add a title. Which is also why it is the
   * section's `aria-labelledby` rather than an `aria-label` kept alongside: one string, one source,
   * and the landmark and the outline cannot come to name this panel two different things.
   */
  const headingId = useId();

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
        capacityReading(evaluation, canOffload, tunableCeiling),
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
    : [
        capacityReading(evaluation, canOffload, tunableCeiling),
        decodeReading(evaluation),
        prefillReading(evaluation),
      ];

  return (
    <section aria-labelledby={headingId} className="grid gap-3 sm:grid-cols-3">
      {/* `sr-only`, so the outline gains a parent and the strip gains no chrome. The three tiles
          already title themselves and a fourth title above them would read as a summary of the very
          kind this panel refuses to compute. Absolutely positioned, so it is not a grid item and
          takes none of the three columns. */}
      <h2 id={headingId} className="sr-only">
        Verdicts
      </h2>
      {readings.map((reading) => {
        const tone = TONE_STYLE[reading.tone];
        return (
          <article key={reading.key} className="panel flex flex-col gap-1 p-[min(1rem,4vw)]">
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
