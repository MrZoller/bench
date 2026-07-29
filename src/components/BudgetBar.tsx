import { useId, useState } from 'react';
import type { Evaluation } from '@/engine';
import { gibLabel, multiple, percent } from '@/lib/format';
import { marks } from '@/design/tokens';
import { DisclosureToggle } from './DisclosureToggle';

/**
 * The memory budget, as a stacked bar against the allocatable ceiling.
 *
 * This is the hero. The whole argument of the tool is visible in one shape: weights are a fixed
 * block, KV grows as you drag context and concurrency, and the ceiling does not move. When the
 * stack passes the ceiling the bar says so structurally — an overflow region beyond the line —
 * rather than by turning red, because "it turned red" does not tell you *by how much*.
 *
 * **That structural claim inverts at a large overshoot, and the sentence has to take over** (#73).
 * `scale` follows the stack, so past roughly 3x the ceiling the *budget* becomes the sliver and the
 * overflow becomes the whole picture: at 14x, DeepSeek V3 on a 5090 draws a full bar of segments
 * with the ceiling 7% from the left edge, which reads as "nearly full" rather than "fourteen times
 * over". The fix is not a different scale — keeping the stack on screen is still right — it is that
 * the multiple is stated in words past the point the shape can carry it. See `OVERSHOOT_STATED`.
 *
 * Colour is never the only channel here: every segment carries a direct label and a 2px surface
 * gap, and the same figures are available as a table for anyone who cannot use the bar at all.
 */

/**
 * Where the drawn shape stops conveying the overshoot and the sentence has to state it.
 *
 * 3x, because that is where the two halves of the picture change places: at 3x the budget occupies
 * the left third of the bar and the gap is still legible as a proportion, and past it the ceiling
 * rule closes on the left edge while the segments fill everything. Below the threshold the multiple
 * is the absolute overage said a second way — "over by 3.1 GiB" *and* "1.1x the ceiling" — and a
 * clause that appears on every overflow is a clause people learn to skip, including on the
 * configurations where it is the only thing telling them the scale.
 */
const OVERSHOOT_STATED = 3;

interface Segment {
  key: string;
  label: string;
  bytes: number;
  color: string;
  hint: string;
}

export function BudgetBar({
  evaluation,
  canOffload,
}: {
  evaluation: Evaluation;
  /** Whether this device has a slower tier to spill weights to at all — discrete GPUs only. */
  canOffload: boolean;
}) {
  const [hovered, setHovered] = useState<string | null>(null);
  const tableId = useId();
  const [showTable, setShowTable] = useState(false);

  const { placement } = evaluation;
  const ceiling = placement.allocatableBytesPerDevice;

  /**
   * Every figure in this bar is runtime-specific — the ceiling carries vLLM's 90% pre-allocation
   * and the overhead band is its 1.5 GiB of framework state. For a pair the runtime cannot
   * drive, those are not merely unknown, they are assumptions about software that will never
   * load. Drawing a confident stack from them beside three tiles reading "Unsupported" is the
   * same overclaim the tiles already refuse.
   */
  if (placement.unsupported) {
    return (
      <section className="panel p-[min(1.25rem,5vw)]">
        <h2 className="text-sm font-semibold tracking-wide">
          Memory budget
          <span className="ml-2 font-normal text-[var(--color-text-faint)]">per device</span>
        </h2>
        <p className="mt-3 text-sm text-[var(--color-text-muted)]">
          No budget to show — {placement.unsupported} The ceiling and the overhead band are
          properties of the runtime, so there is nothing here to measure against.
        </p>
      </section>
    );
  }

  const segments: Segment[] = [
    {
      key: 'weights',
      label: 'Weights',
      bytes: placement.weightBytesPerDevice,
      color: 'var(--color-weights)',
      hint: 'Fixed. Set by parameter count and quantization — the one part that does not move when you change usage.',
    },
    {
      key: 'kv',
      label: 'KV cache',
      bytes: placement.kvBytesPerDevice,
      color: 'var(--color-kv-cache)',
      hint: 'Grows with context x concurrency. The term that turns a comfortable fit into an OOM.',
    },
    {
      key: 'overhead',
      label: 'Overhead',
      bytes: placement.activationBytesPerDevice,
      color: 'var(--color-overhead)',
      hint: 'Runtime context, kernels and activation workspace. Small, but it is why 100% of nominal is never available.',
    },
  ];

  // Total spacer width the segments have to give back, so the row still measures 100%.
  const gapTotal = marks.gap * (segments.length - 1);

  const used = placement.usedBytesPerDevice;
  // Scale to whichever is larger, so an over-budget stack stays on screen and its overflow is
  // legible as a proportion rather than clipped at the edge.
  const scale = Math.max(used, ceiling) || 1;
  const overflows = used > ceiling;

  /**
   * How many times over the ceiling the stack is — the figure the shape stops carrying.
   *
   * Computed from this panel's own two numbers rather than read from `placement.utilization`, which
   * is the same quotient one level up. That is deliberate: this is exactly the reciprocal of where
   * the ceiling rule is drawn (`ceiling / scale`), so the sentence and the line are two readings of
   * one expression and cannot come apart. A second source for a figure whose whole job is to
   * describe the picture is how the two would eventually disagree — the failure this file already
   * carries a long comment about, one paragraph down.
   *
   * Finite-checked rather than merely compared: a degenerate zero ceiling makes this `Infinity`,
   * which passes the threshold and would print an em dash where a multiple belongs.
   */
  const overshoot = used / ceiling;
  const statesOvershoot = Number.isFinite(overshoot) && overshoot >= OVERSHOOT_STATED;

  /**
   * What the overflow line should say, which is not always "spill the weights".
   *
   * `offloadFraction` is capped at 1, so a placement where the cache and overhead *alone* pass
   * the ceiling reported "100% of weights would spill" — an instruction that reads like a remedy
   * and contradicts the capacity verdict a few pixels away. Removing every weight still leaves
   * this configuration over, and the bar has the figures to say so.
   *
   * Split on `canOffload`, for the same reason `capacityReading` in Telemetry does: a discrete GPU
   * and a Mac both reach `impossible`, by different routes. Testing the non-offloadable floor
   * instead read that Mac as an overflowing 5090 and told it to spill, on a machine with no tier
   * to spill to — the two panels sit one above the other and described the same placement two
   * different ways.
   *
   * This says what the overflow *is* and stops there. Whether the ceiling can be raised is a
   * remedy, and Telemetry states it a few pixels below; saying it twice in adjacent panels is how
   * one of the two copies later drifts.
   *
   * `floorBytesPerDevice` rather than this panel's own `kvBytesPerDevice + activationBytesPerDevice`,
   * which is the same rule one level down: the figure has to come from the device the predicate
   * refused. Those two agree on every rig whose devices hold the same amount, and part company under
   * a layer split — Gemma 3 12B on three 4090s at 128K and 8 users is impossible because two cards
   * need 24.6 GiB of cache and workspace against a 23 GiB ceiling, while the card the rest of this
   * bar describes needs 19.1. Rebuilt here, the sentence read "the cache and overhead alone need
   * 19.1 GiB" under a header reading 23.0 GiB, and disproved the claim it was making.
   *
   * Which leaves the figure true of a device the segments beside it are not drawing, so the sentence
   * says whose it is. Naming the card is cheaper than the alternatives — redrawing the bar for a
   * device the user did not ask about, or going back to a figure that reconciles with the segments
   * by being wrong about the refusal.
   */
  const floorBytes = placement.floorBytesPerDevice;
  /** Whether that floor belongs to some other card than the one this bar is drawing. */
  const floorIsElsewhere =
    floorBytes > placement.kvBytesPerDevice + placement.activationBytesPerDevice + 1;
  const overflowDetail = placement.impossible
    ? canOffload
      ? ` — ${floorIsElsewhere ? 'the busiest card by cache needs' : 'the cache and overhead alone need'} ${gibLabel(floorBytes)}, and neither can be offloaded, so spilling every weight would still leave it over`
      : ' — and this memory is the machine’s own, so there is nowhere faster to spill to'
    : placement.offloadFraction > 0
      ? ` — ${percent(placement.offloadFraction)} of weights would spill to host RAM`
      : '';

  return (
    <section aria-labelledby={`${tableId}-title`} className="panel p-[min(1.25rem,5vw)]">
      <header className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <h2 id={`${tableId}-title`} className="text-sm font-semibold tracking-wide">
          Memory budget
          <span className="ml-2 font-normal text-[var(--color-text-faint)]">per device</span>
        </h2>
        {/* Same shape as `PanelCount`, and deliberately not that component: this is two
            quantities rather than a count out of a total, and the unbreakable unit is each
            figure — "120 GiB" — rather than the pair. Breaking at the slash keeps both readable
            at a scaled root, where the blanket `whitespace-nowrap` this replaces was a floor on
            the whole line and scrolled the page sideways (#35). */}
        <p className="tabular text-sm text-[var(--color-text-muted)]">
          <span
            className={`whitespace-nowrap ${
              overflows ? 'text-[var(--color-critical)]' : 'text-[var(--color-text)]'
            }`}
          >
            {gibLabel(used)}
          </span>{' '}
          <span className="whitespace-nowrap text-[var(--color-text-faint)]">
            / {gibLabel(ceiling)}
          </span>
        </p>
      </header>

      {/* The bar. role=img with a full text alternative: the shape carries the meaning, and a
          screen reader should get that meaning as a sentence rather than as eleven divs. */}
      <div
        role="img"
        aria-label={`${gibLabel(used)} of ${gibLabel(ceiling)} allocatable used. ${segments
          .map((s) => `${s.label} ${gibLabel(s.bytes)}`)
          .join(', ')}.${overflows ? ' Over budget.' : ''}`}
        className="relative mt-4 h-10 w-full overflow-hidden rounded-md bg-[var(--color-free)]"
      >
        <div className="flex h-full w-full">
          {segments.map((segment, index) => {
            const width = (segment.bytes / scale) * 100;
            if (width <= 0) return null;
            return (
              <div
                key={segment.key}
                className="h-full transition-[width] duration-200 ease-out"
                style={{
                  /**
                   * The gap is taken *out* of each width rather than added beside it. When the
                   * stack overflows, `scale` equals `used`, so the percentages already sum to
                   * 100 — adding fixed margins on top then pushes the row past its container,
                   * and `overflow-hidden` silently clips the trailing segment. A small overhead
                   * band could disappear entirely while still being listed in the legend.
                   */
                  width: `calc(${width}% - ${(gapTotal * width) / 100}px)`,
                  background: segment.color,
                  marginRight: index < segments.length - 1 ? marks.gap : 0,
                  flexShrink: 0,
                }}
                onMouseEnter={() => setHovered(segment.key)}
                onMouseLeave={() => setHovered(null)}
              />
            );
          })}
        </div>

        {/* The ceiling. Drawn over the stack so it reads as a limit the bar is measured against,
            not as another segment — and inside a track-coloured halo, because whenever this line
            exists it is drawn *on top of a fill*. `scale` is `used` while the bar overflows, so the
            segments occupy the whole width and no empty track is left for the rule to sit on; which
            fill it lands on is the only thing the overshoot changes. Dashed critical red over
            saturated blue at 14x is the worst pairing this palette contains, and 1.1x is no better
            — there the rule falls at 91%, inside the amber cache that took the configuration over.
            Either way the gaps between the dashes show the fill, and the line half disappears into
            the thing it is supposed to be measuring.

            A halo rather than a heavier or solid line: the same 2px of track colour the segments
            already put between themselves via `marks.gap`, and the same mechanism the Envelope's
            "you are here" ring and the Matrix's selected cell already use — a mark that overlaps
            another is separated by surface, never by more ink. The dashes then read against the
            track wherever the line falls. */}
        {overflows && (
          <div
            className="absolute inset-y-0 flex justify-center"
            style={{
              // The halo is centred on the rule, so the *line* stays at the true ceiling
              // position: it starts one gap early and is one gap wider on each side.
              left: `calc(${(ceiling / scale) * 100}% - ${marks.gap}px)`,
              width: marks.lineWidth + marks.gap * 2,
              background: 'var(--color-free)',
            }}
            aria-hidden="true"
          >
            <div className="h-full w-0 shrink-0 border-l-2 border-dashed border-[var(--color-critical)]" />
          </div>
        )}
      </div>

      {overflows && (
        <p className="mt-2 text-sm text-[var(--color-critical)]">
          <span aria-hidden="true">▲ </span>
          Over the ceiling by {gibLabel(used - ceiling)}
          {overflowDetail}.
          {/* The clause the picture can no longer supply. `gibLabel(used - ceiling)` above answers
              "by how much" in absolute terms, which is the answer at 1.1x; the multiple is the
              answer at 14x, where 417 GiB past a 31 GiB ceiling is a magnitude nobody derives from
              two figures in a header. Its own sentence rather than an apposition to the overage,
              because it is a different quantity: 417 GiB is the overflow, 14x is the stack. */}
          {statesOvershoot && ` The stack is ${multiple(overshoot)} the ceiling.`}
        </p>
      )}

      {/* Legend, always present for two or more series, doubling as the direct labels. */}
      <ul className="mt-4 flex flex-wrap gap-x-5 gap-y-2">
        {segments.map((segment) => (
          <li
            key={segment.key}
            tabIndex={0}
            aria-describedby={`${tableId}-hint`}
            className={`flex items-center gap-2 rounded text-sm transition-opacity focus:ring-1 focus:ring-[var(--color-accent)] focus:outline-none ${
              hovered && hovered !== segment.key ? 'opacity-50' : 'opacity-100'
            }`}
            onMouseEnter={() => setHovered(segment.key)}
            onMouseLeave={() => setHovered(null)}
            onFocus={() => setHovered(segment.key)}
            onBlur={() => setHovered(null)}
          >
            <span
              aria-hidden="true"
              className="inline-block h-3 w-3 shrink-0 rounded-sm"
              style={{ background: segment.color }}
            />
            <span className="text-[var(--color-text-muted)]">{segment.label}</span>
            <span className="tabular text-[var(--color-text)]">{gibLabel(segment.bytes)}</span>
          </li>
        ))}
        <li className="flex items-center gap-2 text-sm">
          <span
            aria-hidden="true"
            className="inline-block h-3 w-3 shrink-0 rounded-sm border border-[var(--color-border)]"
            style={{ background: 'var(--color-free)' }}
          />
          <span className="text-[var(--color-text-muted)]">Free</span>
          <span className="tabular text-[var(--color-text)]">
            {gibLabel(Math.max(0, ceiling - used))}
          </span>
        </li>
        {/* A key for the ceiling rule, which is the one mark in this bar that never had one.
            Every fill is named here and the line was not, so at a large overshoot the reader met a
            dashed red rule buried in a blue fill with nothing on the page saying what it was — the
            legend is the dependable identity channel precisely because the mark may be hard to
            see. Keyed only when it is drawn, which is on overflow alone, and it carries the ceiling
            figure for the same reason every other row carries its bytes. */}
        {overflows && (
          <li className="flex items-center gap-2 text-sm">
            <span
              aria-hidden="true"
              className="inline-block h-3 w-0 shrink-0 border-l-2 border-dashed border-[var(--color-critical)]"
            />
            <span className="text-[var(--color-text-muted)]">Ceiling</span>
            <span className="tabular text-[var(--color-text)]">{gibLabel(ceiling)}</span>
          </li>
        )}
      </ul>

      {/* aria-live so the hint is announced on focus rather than only appearing visually. */}
      <p
        id={`${tableId}-hint`}
        aria-live="polite"
        className="mt-3 min-h-[1.25rem] text-sm text-[var(--color-text-muted)]"
      >
        {hovered ? segments.find((s) => s.key === hovered)?.hint : ''}
      </p>

      <DisclosureToggle
        expanded={showTable}
        onToggle={() => setShowTable((v) => !v)}
        controls={tableId}
      >
        {showTable ? 'Hide' : 'Show'} figures as a table
      </DisclosureToggle>

      {showTable && (
        <table id={tableId} className="mt-3 w-full text-left text-sm">
          <caption className="sr-only">Memory budget breakdown per device</caption>
          <thead>
            <tr className="text-[var(--color-text-faint)]">
              <th scope="col" className="py-1 font-normal">
                Component
              </th>
              <th scope="col" className="py-1 text-right font-normal">
                Size
              </th>
              <th scope="col" className="py-1 text-right font-normal">
                Share of ceiling
              </th>
              <th scope="col" className="py-1 font-normal">
                What it is
              </th>
            </tr>
          </thead>
          <tbody className="text-[var(--color-text-muted)]">
            {segments.map((segment) => (
              <tr key={segment.key} className="border-t border-[var(--color-border)]">
                <th scope="row" className="py-1 font-normal text-[var(--color-text)]">
                  {segment.label}
                </th>
                <td className="tabular py-1 text-right">{gibLabel(segment.bytes)}</td>
                <td className="tabular py-1 text-right">{percent(segment.bytes / ceiling)}</td>
                <td className="py-1 pl-4">{segment.hint}</td>
              </tr>
            ))}
            <tr className="border-t border-[var(--color-border)]">
              <th scope="row" className="py-1 font-normal text-[var(--color-text)]">
                Allocatable ceiling
              </th>
              <td className="tabular py-1 text-right">{gibLabel(ceiling)}</td>
              <td className="tabular py-1 text-right">100%</td>
              <td className="py-1 pl-4">What the runtime can actually hand the model.</td>
            </tr>
          </tbody>
        </table>
      )}
    </section>
  );
}
