import { useId, useMemo, useState } from 'react';
import {
  computeMatrix,
  measureMax,
  measureValue,
  type MatrixCell,
  type MatrixMeasure,
} from '@/engine/matrix';
import { DEVICES, MODELS, getDevice, getModel } from '@/data/catalog';
import { getQuant } from '@/data/quants';
import { getRuntime, substitutionFor } from '@/data/runtimes';
import { FALLBACK_QUANT_ID, quantApplies } from '@/lib/quantChoice';
import { sequential } from '@/design/tokens';
import { kvLabel } from '@/lib/stops';
import { params, percent, rate, seconds, tokens } from '@/lib/format';
import { useConfig, type Config } from '@/store/config';

/**
 * Every model against every device — the surface for "what are my options", which is the
 * question that comes before the one the Bench answers.
 *
 * Three measures, switchable over the same grid, because that is the clearest way to show that
 * "fits" and "usable" are different questions. Toggling between them visibly rearranges which
 * hardware looks good, which is the capacity/bandwidth/compute triangle made concrete rather
 * than asserted in prose.
 */

const MEASURES: readonly { value: MatrixMeasure; label: string; hint: string }[] = [
  { value: 'fit', label: 'Does it fit', hint: 'Headroom left after weights, cache and overhead.' },
  { value: 'decode', label: 'How fast', hint: 'Tokens per second for one user.' },
  { value: 'ttft', label: 'How responsive', hint: 'Time until the first token appears.' },
];

/**
 * The sequential ramp, reversed.
 *
 * `sequential` runs light to dark for a light surface. On this chassis the darkest step is the
 * one that recedes into the panel, so higher values have to be *brighter* — otherwise the best
 * cells are the ones you cannot see.
 */
const RAMP = [...sequential].reverse();

/**
 * What a row is evaluated at when the selected format does not apply to it, in preference order.
 *
 * Q4_K_M leads rather than the store's BF16 fallback. That one is chosen for *safety* — it always
 * applies — but on a grid meant to compare hardware it makes every dense model look far worse
 * than anyone would actually run it, which is the opposite of informative. Q4_K_M is the
 * default local trade and the honest stand-in.
 *
 * A list rather than one constant, because the substitute has to be a format the *runtime* can
 * load. Returning Q4_K_M unconditionally handed vLLM a GGUF K-quant it does not read, and since
 * the substitution bypassed the runtime check entirely, those rows were sized, coloured and
 * ranked as runnable — then produced different figures when clicked, because the Bench coerces
 * the selection to something loadable. The order runs 4-bit, then 8-bit, then BF16: comparable
 * quality first, universality last, so a row only falls back as far as it has to.
 */
const SUBSTITUTE_QUANT_IDS = ['q4_k_m', 'awq_4bit', 'int8', 'q8_0', FALLBACK_QUANT_ID] as const;

/**
 * The element a Matrix click scrolls back to — the detail that click just loaded.
 *
 * Exported so the Bench holds the anchor and this file only names it. A `getElementById` reaching
 * for a string the other component happens to use is the kind of coupling that breaks silently.
 */
export const DETAIL_ANCHOR_ID = 'bench-detail';

export function Matrix({ config }: { config: Config }) {
  const headingId = useId();
  const [measure, setMeasure] = useState<MatrixMeasure>('fit');
  const set = useConfig((s) => s.set);

  const quant = getQuant(config.quantId);
  const runtime = getRuntime(config.runtimeId);

  /**
   * What this runtime calls the selected cache precision.
   *
   * `kvPrecision` is an internal width, not a user-facing name: vLLM has no integer-Q8 cache, and
   * the catalog maps that value to FP8 precisely because "Q8" names a flag its users cannot type.
   * Upper-casing the internal value here described a vLLM setting that does not exist — in the
   * heading, and so in any screenshot taken from this panel. The Bench's own control has always
   * resolved it properly; this was the one place that did not, and the resolution is now one
   * function so a third surface cannot invent a fourth answer.
   */
  const kv = kvLabel(runtime, config.kvPrecision);

  const models = useMemo(
    () =>
      [...MODELS].sort((a, b) => (b.popularity?.downloads ?? 0) - (a.popularity?.downloads ?? 0)),
    []
  );
  // Shipping hardware only: a rumoured row would put speculative specs into a comparison people
  // read as a shortlist, and `status` exists precisely so that never happens silently.
  const devices = useMemo(() => DEVICES.filter((d) => d.status === 'shipping'), []);

  /**
   * The selected format where it applies, and a universal one where it does not.
   *
   * Forcing one format across the grid blanked more than half the rows at the default config:
   * MXFP4 is expert-only, so every dense model reported "does not apply" — a quantization fact
   * standing in for a hardware comparison, on the one surface whose job is comparing hardware.
   * Substituting keeps every row informative, and the substitution is stated rather than hidden.
   */
  const quantFor = useMemo(
    () => (model: (typeof models)[number], device: (typeof devices)[number]) => {
      if (quantApplies(quant, model, device, runtime)) return quant;
      // `runtime` is passed to both checks. Omitting it from the first let an unloadable
      // selection through as though it applied; omitting it from the second chose an unloadable
      // stand-in. The last entry always applies, so `find` cannot come back empty.
      return (
        SUBSTITUTE_QUANT_IDS.map(getQuant).find((q) => quantApplies(q, model, device, runtime)) ??
        getQuant(FALLBACK_QUANT_ID)
      );
    },
    [quant, runtime]
  );

  /**
   * The formats actually standing in, named so the header can state them.
   *
   * A set rather than a flag: the substitute now depends on what the runtime can load and what
   * the device can run, so one grid can carry more than one. Saying "Q4_K_M where it does not
   * apply" when half the rows were really evaluated at BF16 would misdescribe the comparison
   * being shown.
   */
  const substitutes = useMemo(() => {
    const used = new Set<string>();
    for (const m of models) {
      for (const d of devices) {
        const chosen = quantFor(m, d);
        if (chosen.id !== quant.id) used.add(chosen.label);
      }
    }
    return [...used];
  }, [models, devices, quant, quantFor]);

  const cells = useMemo(
    () =>
      computeMatrix({
        models,
        devices,
        runtime,
        usage: {
          contextTokens: config.contextTokens,
          concurrency: config.concurrency,
          promptTokens: config.promptTokens,
          kvPrecision: config.kvPrecision,
        },
        deviceCount: 1,
        quantFor,
      }),
    [models, devices, quantFor, runtime, config]
  );

  const max = measureMax(cells, measure);
  const runnable = cells.flat().filter((c) => c.runs).length;

  /**
   * Whether a cell is the scenario the Bench is currently showing.
   *
   * The device count is part of that question, and leaving it out made the mark a lie on any
   * linked rig: every cell here is scored with `deviceCount: 1`, so with the Bench on 2–8 devices
   * the ring and `aria-current` claimed it was showing a cell whose capacity and speed describe a
   * different machine. Clicking the already-marked cell then silently reset the configuration to
   * one device — the one thing a cell that says "you are here" should not do.
   *
   * One predicate rather than the two copies this had, which is the same rule as everywhere else
   * here: a mark and the thing it marks are one claim, and two hand-written copies of it are how
   * the ring and the screen-reader state come to disagree.
   */
  const isCurrent = (cell: MatrixCell) =>
    cell.modelId === config.modelId &&
    cell.deviceId === config.deviceId &&
    config.deviceCount === 1;

  /**
   * Whether any cell on this grid was scored at a format the runtime cannot actually load.
   *
   * Two routes reach it: the *selected* format may itself be a stand-in (every Apple-silicon row
   * under MLX), or a row falling back through `SUBSTITUTE_QUANT_IDS` may land on one.
   *
   * Only the first is reachable with today's catalog — none of MLX's formats carries a `requires`
   * or a `denseBpw`, so `quantApplies` is true for every model and `quantFor` never falls back, and
   * a test asserting the second would have nothing to drive it with. Scanning the cells anyway
   * costs one pass and closes the route before a catalog change opens it, which is cheaper than
   * noticing later that the grid was marked honestly on a Mac and silently on a fallback row.
   *
   * `evaluated`, not `runs`. A cell that was measured and did not fit is still a figure derived
   * from the stand-in: its verdict, its tooltip and — the sharp end — its "past the default
   * allocation, which this machine lets you raise" recommendation all rest on the stand-in's bit
   * width. Gating on `runs` hid the mark exactly when the grid was most confidently wrong: at 128K
   * over 128 users on MLX, every Apple cell fails placement, so the grid published 85 verdicts and
   * a raise-the-ceiling recommendation with nothing saying what they were computed from. Since
   * Q4_K_M's 4.85 bpw is the *heavier* stand-in, a borderline "past the default" is the verdict
   * most likely to flip. Raised by Codex on PR #32.
   */
  const substitutedCells = useMemo(
    () => cells.flat().some((cell) => cell.evaluated && substitutionFor(runtime, cell.quantId)),
    [cells, runtime]
  );

  /**
   * Tall enough for the longest label this grid actually renders.
   *
   * A fixed 96px was set for the names that existed then and is short for several shipping ones —
   * the catalog reaches 40 characters. The table sits in an `overflow-x-auto` container, which
   * clips vertically rather than scrolling, and the Mac Studio variants differ only in the
   * trailing capacity suffix that got cut, so two columns became indistinguishable — the exact
   * failure the rotation was introduced to fix.
   *
   * Rotated 45 degrees, so the vertical extent is the label's width times sin(45).
   *
   * 8px per character is an estimate rather than a measurement, and the point of the estimate is
   * that it errs *long*: the cost of erring is whitespace, where the cost of erring short is a
   * header clipped by the `overflow-x-auto` container, which clips vertically rather than
   * scrolling. The first version claimed to err long at 6.5 and did not — the app's font stack
   * renders the widest catalogued label at **7.03px per character**, so the constant was 8% short
   * and the `+20` was absorbing all of it, leaving 1.08px of clearance on a 204px row.
   *
   * That was invisible until `e2e/matrix-header.spec.ts` measured it, and it was not merely tight:
   * `--font-sans` resolves to `system-ui`, which is SF on macOS and whatever fontconfig picks on a
   * CI runner. A metrics difference of 1% either way decided whether the header clipped. The spec
   * now asserts the clearance directly, so this constant cannot quietly go short again.
   */
  const headerHeight = useMemo(() => {
    const longest = Math.max(0, ...devices.map((d) => shortName(d.name).length));
    return Math.ceil(longest * 8 * Math.SQRT1_2) + 20;
  }, [devices]);

  return (
    <section aria-labelledby={headingId} className="panel p-5">
      <header className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-2">
        {/* The grid moves materially with context, concurrency, prompt and KV precision — the fit
            counts and the throughput colours all change — and the heading named only the format
            and the runtime. The sliders that control it sit below this section, so a screenshot of
            the Matrix carried no record of the request it answers, and two of them taken at
            different settings are indistinguishable. Every input that moves a cell is stated. */}
        <h2 id={headingId} className="text-sm font-semibold tracking-wide">
          Every model on every machine
          <span className="ml-2 font-normal text-[var(--color-text-faint)]">
            at {quant.label}
            {substitutes.length > 0 &&
              `, ${substitutes.join(' or ')} where it does not apply`}, {runtime.label} —{' '}
            {tokens(config.contextTokens)} context, {tokens(config.promptTokens)} prompt,{' '}
            {config.concurrency} {config.concurrency === 1 ? 'user' : 'users'}, {kv} KV
            {/* Every cell is scored at one device, and until this said so a Bench configured for a
                linked rig showed a grid describing hardware the user had not asked about — with
                nothing on screen to reveal the substitution. Stated only when it differs from what
                the Bench holds, since on a single-device configuration it is not news. */}
            {config.deviceCount > 1 && ', one device per cell'}
          </span>
        </h2>
        <p className="text-sm whitespace-nowrap text-[var(--color-text-muted)]">
          <span className="tabular text-[var(--color-text)]">{runnable}</span> of{' '}
          {cells.flat().length} combinations run
        </p>
        {/* Outside the h2, which is this section's `aria-labelledby` target — the accessible name
            is computed from its whole subtree, so a sentence nested in there is read out every
            time the landmark is announced. The workload belongs in the name; the caveat does not. */}
        <p className="basis-full text-sm text-[var(--color-text-faint)]">
          Rows are capped at each model’s own context limit, so a model that stops short of{' '}
          {tokens(config.contextTokens)} is scored at whatever it does accept.
        </p>
      </header>

      {/* One filter row above the grid, as the dataviz guidance puts it. */}
      <fieldset className="mt-4">
        <legend className="sr-only">Colour the grid by</legend>
        <div className="flex flex-wrap gap-1 rounded-md border border-[var(--color-control-border)] bg-[var(--color-surface-raised)] p-1">
          {MEASURES.map((m) => (
            <button
              key={m.value}
              type="button"
              aria-pressed={m.value === measure}
              onClick={() => setMeasure(m.value)}
              className={`rounded px-3 py-1 text-sm transition-colors ${
                m.value === measure
                  ? 'bg-[var(--color-accent-dim)] text-[var(--color-text)]'
                  : 'text-[var(--color-text-muted)] hover:text-[var(--color-text)]'
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
        <p className="mt-1.5 text-xs text-[var(--color-text-muted)]">
          {MEASURES.find((m) => m.value === measure)?.hint} Switching between these rearranges which
          hardware looks good — that disagreement is the whole point.
        </p>
      </fieldset>

      <div className="mt-4 overflow-x-auto">
        <table className="w-full border-separate border-spacing-0.5 text-left text-xs">
          <caption className="sr-only">
            Every catalogued model against every shipping device, coloured by{' '}
            {MEASURES.find((m) => m.value === measure)?.label}. {runnable} of {cells.flat().length}{' '}
            combinations run.
          </caption>
          <thead>
            <tr>
              <th scope="col" className="sticky left-0 bg-[var(--color-surface)] pr-2 font-normal">
                <span className="sr-only">Model</span>
              </th>
              {devices.map((d) => (
                <th
                  key={d.id}
                  scope="col"
                  // Fixed width, and the label taken out of flow below, so a long name cannot
                  // stretch its own column — "RTX PRO 6000 Blackwell" was three times the width
                  // of its neighbours and skewed the whole grid.
                  className="relative w-7 min-w-7 p-0 align-bottom font-normal text-[var(--color-text-faint)] [@media(pointer:coarse)]:w-11 [@media(pointer:coarse)]:min-w-11"
                  style={{ height: headerHeight }}
                >
                  {/*
                    Rotated rather than truncated. Horizontally these clipped to "GeForc…" four
                    times over — a header that cannot distinguish its own columns is worse than
                    none, and the names are what make the grid readable at all.
                  */}
                  <span
                    className="absolute bottom-1 left-1/2 origin-bottom-left -rotate-45 whitespace-nowrap"
                    title={d.name}
                  >
                    {shortName(d.name)}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {models.map((model, r) => (
              <tr key={model.id}>
                <th
                  scope="row"
                  className="sticky left-0 max-w-[9rem] truncate bg-[var(--color-surface)] pr-2 font-normal text-[var(--color-text-muted)]"
                  title={`${model.name} — ${params(model.totalParams)}`}
                >
                  {model.name}
                </th>
                {cells[r].map((cell, c) => (
                  <td key={devices[c].id} className="p-0">
                    <button
                      type="button"
                      /**
                       * Loads the *whole* scenario the cell was scored under, not just the pair.
                       *
                       * Setting only model and device meant a dense-model cell scored at the
                       * Q4_K_M substitute landed in a Bench still holding MXFP4 — which `coerce`
                       * then replaced with BF16, so the grid and the detail view disagreed about
                       * the same square. The context and the single-device count matter for the
                       * same reason.
                       */
                      onClick={() => {
                        set('modelId', cell.modelId);
                        set('deviceId', cell.deviceId);
                        set('quantId', cell.quantId);
                        set('deviceCount', 1);
                        set('contextTokens', cell.contextTokens);
                        /**
                         * The detail this loads sits several sections above, and a cell already
                         * matching the current selection changes nothing the Matrix renders — so
                         * clicking one left the viewport on an unchanged grid and the action
                         * appeared to do nothing at all. The selected square is now marked too,
                         * so the click is acknowledged where it happened as well as where it
                         * landed.
                         */
                        // Optional on the method as well as the element: `scrollIntoView` is
                        // absent in jsdom and in some embedded browsers, and a click that throws
                        // here would abandon the selection it had just made — trading a scroll
                        // that did not happen for a scenario that did not load.
                        //
                        // The animation is gated on the motion preference, which the stylesheet's
                        // reduced-motion block cannot do for it: that neutralises CSS animation
                        // and transition durations and has no effect on a scroll asked for in JS.
                        // A multi-section animated jump is exactly the motion it exists to
                        // suppress, so the preference is read here instead.
                        const reduce = window.matchMedia?.(
                          '(prefers-reduced-motion: reduce)'
                        )?.matches;
                        document.getElementById(DETAIL_ANCHOR_ID)?.scrollIntoView?.({
                          behavior: reduce ? 'auto' : 'smooth',
                          block: 'start',
                        });
                      }}
                      title={tooltip(cell, measure, quant.id, config.deviceCount)}
                      aria-label={tooltip(cell, measure, quant.id, config.deviceCount)}
                      aria-current={isCurrent(cell) ? 'true' : undefined}
                      // 28px squares two pixels apart are under the 44px `marks.hitTarget` this
                      // repo declares, and with hundreds of neighbours a touch user loading the
                      // wrong scenario is the likely outcome rather than the unlucky one. Coarse
                      // pointers get the full target; a mouse keeps the dense grid, which is what
                      // makes the comparison legible in one screen.
                      className={`h-7 w-full rounded-sm focus:ring-2 focus:ring-[var(--color-accent)] focus:outline-none [@media(pointer:coarse)]:h-11 ${
                        cell.runs ? '' : 'border border-dashed border-[var(--color-border)]'
                      } ${cell.raiseCeilingWouldHelp ? 'border-[var(--color-warning)]' : ''} ${
                        isCurrent(cell)
                          ? 'ring-2 ring-[var(--color-accent)] ring-offset-1 ring-offset-[var(--color-surface)]'
                          : ''
                      }`}
                      style={{ background: fill(cell, measure, max) }}
                    />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* A ramp legend, since a continuous scale has no discrete keys to list.
          `flex-wrap`, because two of the four entries are prose and the row does not fit a phone.
          Unlike the grid above it this div has no scroll container of its own, so a row that
          overran did not scroll itself — it scrolled the page. At 320px the legend measured 299px
          inside a 246px box and took the document to 336/320. Issue #34; guarded by
          `e2e/matrix-legend.spec.ts`, since jsdom reports every one of those widths as 0. */}
      <div className="mt-4 flex flex-wrap items-center gap-x-3 gap-y-2 text-xs text-[var(--color-text-muted)]">
        {/* The ramp and its two ends are one item, with a floor under it.
            `flex-1` is `flex: 1 1 0%`, so the gradient's flex basis is zero and it is the only
            thing here that yields when the row is tight — which it was at every width below about
            1280px once the third key appeared. The ramp measured 0×12 on a 1024px laptop: the
            legend's entire subject, absent, while the prose explaining the exceptions sat at full
            width. Wrapping alone does not fix that, because a zero-basis item on a full line still
            gets no space: it survives wherever a line breaks early (139.8px at 390) and collapses
            wherever the keys nearly fill one (13.6px at 1024). A floor is what makes the ramp
            claim a width and, failing that, take a line of its own — `min-width` is resolved into
            the hypothetical main size, so it governs line-breaking and shrinking alike.
            Capped at `100%`, not left at a bare `12rem`, because a rem floor is a floor the
            viewport cannot argue with: under browser text scaling the root grows while the
            viewport does not, and at 320px with a 24px root the floor alone took the document to
            343/320 — reintroducing the sideways scroll this whole block exists to remove, in the
            one setting a reader most needs it not to. `min()` yields instead. */}
        <span className="flex min-w-[min(12rem,100%)] flex-1 items-center gap-2">
          <span>worse</span>
          <span aria-hidden="true" className="flex h-3 flex-1 overflow-hidden rounded-sm">
            {RAMP.map((step) => (
              <span key={step} className="flex-1" style={{ background: step }} />
            ))}
          </span>
          <span>better</span>
        </span>
        <span className="flex items-center gap-1.5">
          <span
            aria-hidden="true"
            className="inline-block h-3 w-3 rounded-sm border border-dashed border-[var(--color-border)]"
          />
          will not run
        </span>
        {/* A state the grid is really in gets a line, and only when it is in it.
            No glyph, unlike its two neighbours. Theirs key a swatch that literally matches a cell
            border, so a reader scans the grid and finds it; this is about where the numbers came
            from, and nothing on the grid can be pointed at — which is the whole problem it reports.
            A key to a mark that appears nowhere is worse than prose. */}
        {substitutedCells && (
          <span className="text-[var(--color-warning)]">
            some rows scored at a stand-in format {runtime.label} cannot load
          </span>
        )}
        {/* A default allocation and a hardware limit are not the same answer, and this grid is
            read as a shortlist. DeepSeek V3 at Q5_K_M is past the 512 GB Mac Studio's 384 GiB
            default and inside the 512 it can be tuned to — struck off the list over a checkbox,
            when the Envelope and Telemetry both kept the distinction. Shown only when the grid
            actually contains one, so the legend does not explain a state nobody is looking at. */}
        {cells.flat().some((c) => c.raiseCeilingWouldHelp) && (
          <span className="flex items-center gap-1.5">
            <span
              aria-hidden="true"
              className="inline-block h-3 w-3 rounded-sm border border-dashed border-[var(--color-warning)]"
            />
            past the default allocation, which this machine lets you raise
          </span>
        )}
      </div>
    </section>
  );
}

/**
 * A device name with the parts every row shares removed.
 *
 * "GeForce RTX 5090" and "GeForce RTX 5080" differ in one character at the end, so a truncating
 * header shows the same string for both. The vendor line is the redundant part.
 */
function shortName(name: string): string {
  return name.replace(/^(GeForce|Instinct|Radeon)\s+/, '');
}

/** Colour for a cell: a step of the ramp, or the recessive "did not run" fill. */
function fill(cell: MatrixCell, measure: MatrixMeasure, max: number): string {
  const value = measureValue(cell, measure);
  // Absence is not a low score. A pair that cannot run gets the empty fill, so the ramp is only
  // ever read across things that actually ran.
  // A hole, not a dark value: the panel surface, so an unrunnable pair is never mistaken for a
  // poor score at the bottom of a ramp whose darkest step is also nearly black.
  if (value === undefined || max <= 0) return 'transparent';

  /**
   * Log-scaled, because these ranges span orders of magnitude: decode runs from ~2 tok/s on a
   * CPU host to ~300 on a B200, and a linear ramp spends every step but the last on the top
   * device while the whole rest of the grid sits in one indistinguishable dark band. The
   * comparison people need is "is this twice as fast", not "what fraction of the best is it".
   */
  const index = Math.min(
    RAMP.length - 1,
    Math.floor((Math.log1p(value) / Math.log1p(max)) * RAMP.length)
  );
  return RAMP[index];
}

/**
 * What a cell says on hover, and to a screen reader. Never colour alone.
 *
 * Both substitutions this grid makes are named here as well as in the heading, because the heading
 * says which formats stand in *somewhere* while only the cell knows whether it is one of them. The
 * device count is the same kind of claim: every cell is scored at one device, so on a linked rig
 * the figures describe hardware the reader did not ask about — and clicking the cell adopts that
 * substitution rather than merely displaying it.
 */
function tooltip(
  cell: MatrixCell,
  measure: MatrixMeasure,
  selectedQuantId: string,
  selectedDeviceCount: number
): string {
  const model = getModel(cell.modelId).name;
  const device = getDevice(cell.deviceId).name;
  // Stated even for a blocked cell: "does not run" is a claim about a machine, and on a linked rig
  // it would otherwise read as a verdict on the rig the Bench is holding.
  const rig = selectedDeviceCount > 1 ? `${device}, one device` : device;
  if (!cell.runs) return `${model} on ${rig}: ${cell.blockedBy ?? 'does not run'}.`;

  const detail =
    measure === 'fit'
      ? cell.offloadFraction > 0
        ? `runs only by spilling ${percent(cell.offloadFraction)} of its weights to host RAM`
        : `${percent(Math.max(0, 1 - cell.utilization))} of the ceiling free`
      : measure === 'decode'
        ? `${rate(cell.tokensPerSec)} tok/s per user`
        : `${seconds(cell.ttftSeconds)} to first token`;

  const at = cell.quantId === selectedQuantId ? '' : ` at ${getQuant(cell.quantId).label}`;
  return `${model} on ${rig}${at}: ${detail}.`;
}
