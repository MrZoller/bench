import { useId, useState } from 'react';
import type { Evaluation } from '@/engine';
import { gibLabel, percent } from '@/lib/format';
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
 * Colour is never the only channel here: every segment carries a direct label and a 2px surface
 * gap, and the same figures are available as a table for anyone who cannot use the bar at all.
 */

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
            not as another segment. */}
        {overflows && (
          <div
            className="absolute inset-y-0 border-l-2 border-dashed border-[var(--color-critical)]"
            style={{ left: `${(ceiling / scale) * 100}%` }}
            aria-hidden="true"
          />
        )}
      </div>

      {overflows && (
        <p className="mt-2 text-sm text-[var(--color-critical)]">
          <span aria-hidden="true">▲ </span>
          Over the ceiling by {gibLabel(used - ceiling)}
          {overflowDetail}.
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
