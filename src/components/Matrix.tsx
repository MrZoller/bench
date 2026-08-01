import { useCallback, useId, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import {
  computeMatrix,
  measureRange,
  measureValue,
  type MatrixCell,
  type MatrixMeasure,
} from '@/engine/matrix';
import { comparisonGrid, getDevice, getModel } from '@/data/catalog';
import { getQuant } from '@/data/quants';
import { getRuntime, kvSubstitutionFor, runtimeDrives, substitutionFor } from '@/data/runtimes';
import { FALLBACK_QUANT_ID, quantApplies } from '@/lib/quantChoice';
import { magnitudeFill, magnitudeRamp } from '@/design/tokens';
import { MEASURE_DIRECTION } from '@/engine/measure';
import { DEVICE_CLASS_LABELS, MEASURES, kvLabel } from '@/lib/stops';
import { HOST_RAM_UNCHECKED } from '@/lib/verdicts';
import { PanelCount } from './PanelCount';
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

/**
 * What separates one class band of columns from the next: two spacing steps of the panel's own
 * surface, which is 8px at the default root.
 *
 * **A gap rather than a rule, because the grid's existing separator between squares is already a
 * gap.** The table is `border-separate` with `border-spacing-0.5`, so every square is bounded by half
 * a spacing step of surface — the `dataviz` guidance's "2px surface gap between fills" — and a band
 * boundary is that same channel, five times wider: `2 × --spacing` on top of the `0.5 × --spacing`
 * gutter. Ink was the alternative and it has nowhere to come from:
 * `tokens.ts` records `--color-border` at 1.18:1 on this fill, which is why it is the panel edge and
 * not a control boundary, and anything strong enough to read here would compete with the two borders
 * a cell already uses to mean something (dashed for "does not fit", warning for "past the default
 * allocation").
 *
 * **A border rather than padding, and that is geometry rather than taste.** The rotated column label
 * is absolutely positioned at `right-1/2` of its `th`, which resolves against the *padding* box —
 * padding would shift the label off the centre of the square it names, a border does not. The column
 * is widened by exactly the gap (`w-7` → `w-9`, and the coarse-pointer `w-11` → `w-13`) so the square
 * inside keeps its width and its 44px touch target: the gap is added to the column, not taken out of
 * the cell.
 *
 * **`calc(var(--spacing) * 2)` rather than `8px`, and that is the whole of what "exactly the gap"
 * means.** Every length this is compensated against is a multiple of `--spacing`, which is `0.25rem`:
 * `.w-9{width:calc(var(--spacing) * 9)}` against `.w-7{width:calc(var(--spacing) * 7)}` is a
 * difference of two steps, and `border-spacing-0.5` is half a step. Written as `8px` the arithmetic
 * held at a 16px root and nowhere else — at 200% text the column grew by 16px while the border stayed
 * at 8, so two of the 42 squares painted 8px (14%) wider than every other square, and the boundary
 * gutter fell from 5x an ordinary one to 3x. That is #44 exactly ("a length measured from text belongs
 * in the same units as the text"), one token away from the `marks.lineWidth` lesson in `BudgetBar`:
 * two literals for one quantity, here across a *unit* boundary rather than across a file. In the
 * spacing unit the identity `w-7 + gap = w-9` is true at every root, and
 * `e2e/catalog-order.spec.ts` measures it at 16px and at 32px for that reason.
 *
 * Two boundaries at 8px is 16px on a grid whose columns already reach ~1405px, which is the other
 * reason it is spent this way. #64 and #34 are both this header overflowing, so a separator that
 * reserved space per column, or leaned on free space that does not exist, would reopen them; this is
 * the whole cost, it is in flow, and `e2e/catalog-order.spec.ts` measures it at 320px.
 *
 * `data-band-start` carries no style and exists so a spec can find these columns without naming the
 * utility that draws them — the e2e locators named that border class, so the unit fix above would have
 * silently emptied every one of them and passed.
 */
const BAND_GAP = 'border-l-[calc(var(--spacing)*2)] border-l-[var(--color-surface)]';

/**
 * The current-cell mark — the two-tone inset frame the #67 rationale below derives — as one
 * string the cell and its legend key both read.
 *
 * One fact, once, because the two had already drifted (#130): the mark was redesigned from an
 * offset outer ring to this inset frame, and the legend's swatch went on drawing the retired
 * ring — accent painted outside the square with a surface gap, the exact bleed-over geometry the
 * redesign removed. It was the one key on the panel whose sample was not the mark. A swatch that
 * shares the mark's classes cannot depict a shape the grid no longer paints.
 */
const CURRENT_CELL_MARK =
  'inset-ring-2 inset-ring-[var(--color-accent)] shadow-[inset_0_0_0_3px_var(--color-surface)]';

/**
 * What the readout under the grid is pointed at.
 *
 * A position rather than a sentence, so the line is derived from the same state the grid is drawn
 * from and cannot describe a cell the grid has stopped showing — argued where it is used.
 *
 * Three kinds because three things on this grid hid text behind a native tooltip, not one: a cell's
 * figures, the device name a shortened column heading stands for, and the model name the row heading
 * truncates at 9rem along with its parameter count. #71 names the cell; the two headings are the
 * same defect with less arithmetic in it.
 */
type Readout =
  | { kind: 'cell'; row: number; col: number }
  | { kind: 'column'; deviceId: string }
  | { kind: 'row'; modelId: string };

export function Matrix({ config }: { config: Config }) {
  const headingId = useId();
  /** The measure switch's own description — see the `fieldset` below. */
  const measureHintId = useId();
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

  /**
   * What this grid covers — rows most-downloaded first, columns the shipping rows in file order.
   *
   * Both from `comparisonGrid()` rather than derived here, which is the same rule the ordering
   * itself was fixed under: this file and `Bench.tsx` each held a hand-written copy of the
   * popularity `sort` while `modelsByPopularity()` — the named helper that does exactly this — was
   * called by nothing outside its own test, so one ordering rule had three definitions and the
   * canonical one was dead (#79). The two grids agreed by coincidence. The shipping filter was the
   * other half of that shape, live rather than latent: it was a `status` rule enforced in a
   * component, where `catalog.test.ts` could not see it.
   *
   * Memoised on `[]` because the extent does not change while the page is mounted, and because the
   * arrays are identities every dependency array below is keyed on.
   */
  const { models, devices } = useMemo(() => comparisonGrid(), []);

  /**
   * The class bands: which columns open one, and what to call them.
   *
   * `devices.json` is grouped by class and both surfaces render it in file order, so the bands were
   * already there and nothing marked them — the discrete-GPU, unified-memory and CPU columns ran
   * together as one 42-column strip, and the left-to-right progression that makes 42 rotated headings
   * readable at all was invisible (#79). The gap below is the channel a sighted reader gets and the
   * caption is the channel a screen reader gets, which is the same split the struck headings already
   * make; both come from this one walk, because a mark and the sentence naming it are one claim.
   *
   * Derived from *adjacency* rather than from the declared order in `DEVICE_CLASS_LABELS`, for the
   * reason `Select` groups by adjacency: this marks the runs the catalog actually has, so a row out of
   * its band renders as an extra gap rather than as a grid quietly re-sorted. `catalog.test.ts` is
   * what fails first.
   *
   * The first column opens the first band and gets no separator — there is nothing to its left to be
   * separated from, and a gap there would be indistinguishable from padding. The label comes verbatim
   * from the table the picker's `<optgroup>` headings come from, so a reader who has met "Discrete
   * GPUs" in the Hardware control hears the same words here.
   */
  const bands = useMemo(() => {
    const opens = devices.filter((d, i) => i === 0 || d.class !== devices[i - 1].class);
    return {
      labels: opens.map((d) => DEVICE_CLASS_LABELS[d.class]),
      separated: new Set(opens.slice(1).map((d) => d.id)),
    };
  }, [devices]);

  /**
   * The columns this runtime cannot drive at all — one fact about the scenario, not one per row.
   *
   * Select vLLM and every Mac, every Strix Halo and every CPU host empties out completely — 16 of
   * the 42 shipping columns as the catalog stands at this commit — and every cell in them used to be
   * drawn exactly like a cell that *was* measured and did not fit: `transparent` behind a dashed
   * border, keyed "will not run". A uniformly empty column is the pattern that reads as a confident
   * result, so the picture said "this hardware cannot hold the model" where the truth is "this
   * software does not run here" — and the two need opposite advice. A 256 GB Mac Studio holds
   * Qwen3 8B many times over; the fix a reader would derive from the old picture (buy more memory)
   * is not the fix (change runtime). Every other surface already split them: the Envelope has a
   * separate `unsupported` state with its own sentence, Telemetry says `Unsupported` rather than
   * `Will not run`, and BudgetBar refuses to draw a stack at all. This was the one that collapsed
   * them, on the surface people read as a shortlist (#72).
   *
   * **From `runtimeDrives`, not from "every cell in this column is empty".** At #72's own URL the
   * two happen to agree — the columns with no runnable cell there are exactly these 10, while the
   * DGX Spark still runs a good share of its rows — and that coincidence is precisely what makes
   * deriving the mark from emptiness look safe. It is one slider from wrong. Take that same vLLM
   * grid to 32 concurrent users and the RTX 3090, 4090 and 5080 columns empty out as well, every
   * cell in them refused on counted bytes, and vLLM drives all three fine. Striking those headings
   * would be exactly the misattribution being fixed, only pointed the other way. Model-independent
   * by construction, which is why it is a set of device ids rather than a scan of the grid: "at any
   * size" is the claim, and it is true of the column before any row is scored.
   *
   * **What agrees with what, and what does not.** `planPlacement` turns a cell away on five
   * categorical grounds, and only two of them are facts about a *column*: this one, and "this device
   * has no interconnect, so a model cannot be split across N of them". The other three — a cache
   * precision the runtime cannot store, a weight format it cannot load, a format needing silicon the
   * device lacks — vary with the scenario or with the row, so a column heading is the wrong place to
   * state them. The second column-wide ground is unreachable from here because this grid scores
   * every cell at one device (said in the header above), which leaves this the only reason a column
   * can close today. So the cells' ink and the heading's mark are deliberately *not* the same
   * predicate: the ink asks `evaluated` — was this judged on its numbers — which is the broader
   * question, and the heading asks the narrower one it has wording for.
   *
   * `App.test.tsx` pins the gap shut rather than trusting it, and pins it as an exhaustive claim
   * rather than an example: no cell anywhere may be refused before the arithmetic unless its column
   * is struck. The day a device count is threaded through here, or a runtime is handed a format the
   * catalog can offer it, that assertion fails — instead of quietly producing 17 unexplained holes,
   * which is the failure this whole block exists to prevent.
   */
  const undrivable = useMemo(
    () => new Set(devices.filter((d) => !runtimeDrives(runtime, d)).map((d) => d.id)),
    [devices, runtime]
  );

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

  /**
   * What the ramp spans, which is both what it is scaled against and what the legend now names.
   *
   * One call rather than a maximum here and a pair of endpoints beside the legend: the ramp's top
   * and the figure printed at its bright end are the same quantity, and this file's own history is
   * mostly two derivations of one number disagreeing.
   */
  const range = useMemo(() => measureRange(cells, measure), [cells, measure]);
  // `range.domain` rather than a domain rebuilt here, which is the hazard `MeasureRange` names in its
  // own docstring: two derivations of one quantity are how a scale and its legend come to describe
  // different grids. `undefined` is the empty grid, where `fill` has no ramp to place anything on and
  // paints every cell as a hole.
  const domain = range?.domain;

  /**
   * The grid-wide counts and flags the heading, the caption and the legend all read.
   *
   * One pass rather than five `cells.flat()` walks, and memoised on `cells` because that is what
   * they are about. Since the readout re-renders this component on every pointer move across the
   * grid, an unmemoised scan changed from "once per scenario" to "once per hover" the moment the
   * line was added — 714 cells apiece as the catalog stands at this commit.
   */
  const summary = useMemo(() => {
    const flat = cells.flat();
    return {
      total: flat.length,
      runnable: flat.filter((c) => c.runs).length,
      // Whether *anything* was priced. A caveat about arithmetic nobody performed is noise that
      // teaches people to skip the caveat that matters.
      anyEvaluated: flat.some((c) => c.evaluated),
      anyRaiseable: flat.some((c) => c.raiseCeilingWouldHelp),
      // Whether "combinations run" is counting any cell that only runs by spilling — the
      // condition for the qualifier HOST_RAM_UNCHECKED exists to enforce (#127).
      anySpilled: flat.some((c) => c.runs && c.offloadFraction > 0),
    };
  }, [cells]);

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
  const isCurrent = useCallback(
    (cell: MatrixCell) =>
      cell.modelId === config.modelId &&
      cell.deviceId === config.deviceId &&
      config.deviceCount === 1,
    [config.modelId, config.deviceId, config.deviceCount]
  );

  // Whether the grid contains the marked cell at all, which is what the legend keys — and the fifth
  // of the per-render scans `summary` above exists to stop. Separate from it because this one is a
  // question about the Bench's selection as well as about the grid.
  const marksCurrent = useMemo(() => cells.flat().some(isCurrent), [cells, isCurrent]);

  /**
   * The grid is one tab stop, and the arrow keys move within it.
   *
   * Every cell is a `<button>` with a full-sentence `aria-label`, so before this the grid was one tab
   * stop per cell — 408 of them at the catalog #52 was measured against, 714 as it stands now — and
   * it sat *above* the Usage controls in DOM order, which meant 422 presses of Tab between the top
   * of the page and the context slider that drives every figure on it. A screen-reader user heard
   * every one of those sentences on the way. #66 has since moved those controls above this grid, and
   * that does not make this pattern optional: this is the last panel with a tab stop in it, so
   * without the roving index a reader who Tabs *into* the grid needs 714 presses to get out of the
   * document — nothing downstream is being protected, the reader is. That is the one accessibility
   * affordance this repo had no spec behind, which is exactly why it survived: touch targets, reflow
   * at 200%, coarse-pointer queries and palette contrast all have tokens and tests, and nothing was
   * looking at focus order.
   *
   * The ARIA grid pattern is what this is for: one element in the tab sequence, arrows to move
   * between cells, Home/End for the ends of a row and Ctrl+Home/End for the ends of the grid. One
   * stop per cell becomes one for the grid, whatever the catalog does next.
   *
   * A skip link was the cheaper alternative and is deliberately not here as well — past this fix
   * it would save a single keypress, and it never addressed the screen-reader traversal at all.
   */
  const [active, setActive] = useState<[row: number, col: number]>([0, 0]);
  const cellRefs = useRef(new Map<string, HTMLButtonElement | null>());

  /**
   * What the line was showing when the current pointer gesture *started* — or `null` when there is no
   * pointer gesture, which is how the keyboard is told apart
   * ([#102](https://github.com/MrZoller/bench/issues/102)).
   *
   * The gap #71 left: on a touch-only device the only gesture a cell offers is a tap, and that tap
   * *is* `onClick` — five store keys rewritten and a scroll several sections away. So the readout
   * either filled while navigation was already happening or never filled at all, and a touch reader
   * could not compare two cells, which is one of the three readers #71's rationale names. Every other
   * channel is unavailable to them: there is no hover, the figures live in each cell's `aria-label`,
   * and unlike the Envelope and the budget bar this panel has no table behind a disclosure to fall
   * back to — it *is* the table, and its cells show colour.
   *
   * **The rule is a state, not a gesture: you may commit to a cell whose figures you have already
   * been shown.** Three drafts got here, and the two it replaced are worth keeping because each was
   * a plausible thing to key on and each was wrong for its own reason.
   *
   * Keying on `pointerType === 'touch'` was the first. It reads as the `any-pointer` lesson (#43)
   * pointed the useful way — the event says what happened, where a media query only says what the
   * device has — and it assumes `pen` hovers. Direct-contact styluses do not, so a pen tap fell
   * straight through to activation exactly as the unfixed touch path did.
   *
   * Comparing against the *live* readout target was the second, and a browser makes that comparison
   * always true: tapping a button focuses it, `onFocus` runs before `click`, so by the time the
   * handler is reached the readout is already this cell's. The guard never fired.
   *
   * Asking at `pointerdown` answers both. It lands before focus moves, so the answer is about what
   * the reader was looking at when they reached for the cell — and every input then falls out of one
   * question rather than a list of pointer types. A mouse hovers, so one click commits. A hovering
   * pen is a mouse. A finger and a contact-only pen are not, so the first tap inspects and the second
   * commits. The keyboard has no `pointerdown` at all, which is the `null` case and always commits —
   * focus has already filled the line, and Enter on a focused cell is not a gesture that could have
   * meant anything else.
   *
   * The question is asked of {@link pointerOver} as well as of the readout state, because a mouse
   * that arrives and clicks within one frame has fired `mouseenter` and not yet been rendered.
   *
   * **Consumed on every click**, which is the other half of the keyboard case: the ref would
   * otherwise still hold the last tap's snapshot, so Enter after a tap would be read as that tap
   * still in progress and refuse to activate — for ever, since nothing else would clear it.
   */
  const gesture = useRef<{ inspected: boolean } | null>(null);

  /**
   * The cell a **hovering** pointer is inside, kept in a ref so `pointerdown` can read it *now*.
   *
   * The readout target is React state, and a mouse that arrives and clicks inside one frame fires
   * its enter and its `pointerdown` before the render that state produces — so reading the committed
   * target turned an ordinary mouse click into two, which two browser specs caught immediately.
   *
   * **Written only when the pointer entered without being pressed, which is what "hovered" means.**
   * Three rounds of review went into that clause, each one narrowing where the provenance question
   * gets asked, and the answer is: at the moment the record is written, never inferred afterwards.
   * Keying on the *reading* gesture's `pointerType` was the last wrong version — it takes a
   * contact-only stylus for a mouse, because the browser emits a compatibility enter as the pen
   * touches down and that record then exists before `pointerdown` runs. `buttons` is the thing that
   * actually differs: a mouse and a hovering pen arrive with none pressed, a finger and a
   * contact-only pen arrive already in contact. So the record means "this pointer has been shown the
   * figures", which is the rule, rather than "a pointer of a kind that usually can".
   *
   * **And it names the pointer, because on a hybrid there is more than one** (found in review). A
   * mouse hovers cell A, the keyboard moves the readout to B, and a finger then lands on A: without
   * an identity the finger inherits the mouse's record and commits on its first tap, while the line
   * still describes B. `pointerId` is stable for the mouse and distinct per contact, so the record
   * answers "*this* pointer has been shown the figures" — which is what the sentence meant all along
   * and what three rounds of narrowing were converging on.
   */
  const pointerOver = useRef<{ cell: string; pointerId: number } | null>(null);

  const rowCount = cells.length;
  const colCount = devices.length;
  // Clamped on read rather than reset in an effect: the grid's size follows the catalog and the
  // runtime filter, so a remembered position can fall outside it between renders.
  const activeRow = Math.min(active[0], Math.max(0, rowCount - 1));
  const activeCol = Math.min(active[1], Math.max(0, colCount - 1));

  const focusCell = useCallback((row: number, col: number) => {
    setActive([row, col]);
    // Focus moved here rather than in an effect keyed on `active`, which would pull focus into
    // the grid on first render and on every unrelated re-render that reset it.
    cellRefs.current.get(`${row}:${col}`)?.focus();
  }, []);

  const onCellKeyDown = useCallback(
    (event: KeyboardEvent<HTMLButtonElement>, row: number, col: number) => {
      // A page is most of a screenful of rows without being the whole grid; the catalog is 17
      // models, so five keeps PageUp/PageDown meaningful rather than equivalent to Ctrl+Home.
      const PAGE = 5;
      let nextRow = row;
      let nextCol = col;

      switch (event.key) {
        case 'ArrowRight':
          nextCol = Math.min(colCount - 1, col + 1);
          break;
        case 'ArrowLeft':
          nextCol = Math.max(0, col - 1);
          break;
        case 'ArrowDown':
          nextRow = Math.min(rowCount - 1, row + 1);
          break;
        case 'ArrowUp':
          nextRow = Math.max(0, row - 1);
          break;
        case 'Home':
          nextCol = 0;
          if (event.ctrlKey) nextRow = 0;
          break;
        case 'End':
          nextCol = colCount - 1;
          if (event.ctrlKey) nextRow = rowCount - 1;
          break;
        case 'PageDown':
          nextRow = Math.min(rowCount - 1, row + PAGE);
          break;
        case 'PageUp':
          nextRow = Math.max(0, row - PAGE);
          break;
        default:
          // Enter and Space are the button's own business, and everything else belongs to the
          // page — swallowing keys a grid does not use is how Tab stops working.
          return;
      }

      // Only once a key this grid owns has been recognised, so the arrow keys still scroll the
      // page when the grid has no cell to move to.
      if (nextRow === row && nextCol === col) return;
      event.preventDefault();
      focusCell(nextRow, nextCol);
    },
    [colCount, rowCount, focusCell]
  );

  /**
   * The reserved line under the grid, and the two pieces of state that fill it.
   *
   * Every figure this grid computes used to be reachable only through a native `title` — a mouse,
   * about a second of dwell, one cell at a time, and gone the moment the pointer moves (#71). That
   * leaves three readers with the colour and nothing else. A sighted keyboard user arrowing a row
   * sees the ring move and no numbers, because the sentence is in the accessible name and they are
   * not listening to it. Touch has no hover at all, so the only way to read a cell is to *commit* to
   * it — `onClick` rewrites five store keys and scrolls several sections away. And anyone comparing
   * two cells gets one tooltip at a time, on dwell, dismissed by the move to the second one.
   *
   * It costs the grid more than it would cost another chart, because `fill` log-scales onto seven
   * steps deliberately — the ranges span orders of magnitude — so the colour is explicitly a rank
   * and not a magnitude. Ranks without a readout answer "which is better" and never "by how much",
   * which is the question a shortlist is for.
   *
   * `BudgetBar` already solved this shape: a reserved line that fills on `onMouseEnter` **and**
   * `onFocus`. The focus half is what fixes the keyboard, and it is free once the hover half exists.
   *
   * **Below the grid, not above it.** A line that grows where it can move the cells is a hover trap:
   * the square under the pointer shifts as the line fills, so the readout describes a cell the
   * pointer has left. Under the grid, the only thing a filling line can push is the legend.
   *
   * **Two states, and whichever input moved last wins.** One `hovered` flag would blank the line on
   * `mouseleave` while the focus ring is still sitting on a cell — a mark and its readout
   * disagreeing, which is the failure `isCurrent` above exists to prevent in the other direction. So
   * the keyboard's cell is what the line falls back to when the pointer leaves: the ring is still
   * visible, so its sentence should still be there.
   *
   * **`hovered ?? focused` on its own is that same disagreement pointed the other way**, and this
   * shipped with it. A pointer resting anywhere on the grid fires `mouseenter` and — because the
   * mouse never moves — never fires `mouseleave`, so its cell outranked the keyboard *indefinitely*:
   * a reader tabbing in and arrowing across a row moved the ring while the line went on printing a
   * different model on a different machine, for every keystroke. Cells carry no `hover:` style, so
   * the ring is the only mark on screen, and #71's first named reader — the sighted keyboard user —
   * got a *wrong* figure where before they got none. A cell's `onFocus` therefore drops the
   * pointer's claim, which makes the rule "the input that moved last", with hover free to take the
   * line back on its next `mouseenter`. `BudgetBar`'s single-state hint cannot have this, because
   * one state means last-event-wins by construction.
   *
   * The response to a hover is this line rather than a mark on the cell, deliberately. Anything
   * drawn on a square inherits the two-tone contrast obligation the selection ring carries — the
   * accent measures 1.06:1 on one step of this ramp — and a mark would still say nothing to the
   * reader who cannot use a pointer.
   */
  const [hovered, setHovered] = useState<Readout | null>(null);
  const [focused, setFocused] = useState<Readout | null>(null);

  /**
   * What a column heading says beyond its own shortened label.
   *
   * One derivation, two channels: the `th`'s `aria-label` when the runtime cannot drive the machine,
   * and this line whenever a reader points at the heading. Two hand-written copies of a refusal is
   * how the strike-through and the spoken column name come to describe different columns.
   */
  const columnReadout = (device: (typeof devices)[number]) =>
    undrivable.has(device.id)
      ? `${device.name} — ${runtime.label} does not support this hardware, at any size.`
      : device.name;

  /**
   * And what a row heading says: the model name the 9rem column truncates, with the parameter count
   * that only ever existed in its `title`.
   */
  const rowReadout = (model: (typeof models)[number]) =>
    `${model.name} — ${params(model.totalParams)}`;

  /**
   * Each cell's sentence, computed once per grid and read by three things.
   *
   * The `title`, the `aria-label` and the readout are one string by construction rather than by three
   * calls agreeing — which is this file's recurring lesson in the small. It is also the render cost
   * that matters most here: two `tooltip()` calls per cell was 1,428 per render as the catalog stands
   * at this commit, and the readout means a render now happens on every pointer move across the grid
   * rather than only when the scenario changes.
   */
  const sentences = useMemo(() => {
    // The heading each column actually shows, which is what the narrow form names — one lookup
    // rather than `headerColumns` again, since a second derivation of this list is how the readout
    // and the header come to disagree about what a column is called.
    const shown = new Map(headerColumns(devices).map((c) => [c.device.id, c.label]));
    return cells.map((row) =>
      row.map((cell) => ({
        full: tooltip(cell, measure, quant.id, config.deviceCount),
        brief: tooltip(cell, measure, quant.id, config.deviceCount, {
          shortDevice: shown.get(cell.deviceId) ?? getDevice(cell.deviceId).name,
          runtime: runtime.label,
        }),
      }))
    );
  }, [cells, devices, measure, quant.id, config.deviceCount, runtime.label]);

  /**
   * The sentence itself, derived at render rather than stored when the pointer arrives.
   *
   * Storing the string would freeze it: the measure switch, the four Usage sliders and the runtime
   * all move what a cell says, and a line still showing "310 tok/s per user" after the grid has been
   * recoloured for TTFT is worse than an empty one. Storing *where* the reader is costs nothing and
   * cannot go stale.
   *
   * A heading is held by id and a cell by its indices, which is each one's own address rather than
   * two conventions: `cells` is indexed the way the roving tab stop and `cellRefs` already index it,
   * and read defensively because the grid's size follows the catalog and the runtime filter — the
   * same reason `activeRow` is clamped. A heading has no such pair, and an id spares this the
   * assumption that `headerBand.columns` and `devices` are still in the same order.
   */
  const target = hovered ?? focused;
  /**
   * Two strings rather than one, and CSS decides which is shown — see `tooltip`'s `brief`.
   *
   * A heading readout is already short and is the same either way: a device or model name is what
   * the truncated heading was hiding, so there is no preamble to drop.
   */
  const readout = (() => {
    if (!target) return { full: '', brief: '' };
    if (target.kind === 'column') {
      const said = columnReadout(getDevice(target.deviceId));
      return { full: said, brief: said };
    }
    if (target.kind === 'row') {
      const said = rowReadout(getModel(target.modelId));
      return { full: said, brief: said };
    }
    return sentences[target.row]?.[target.col] ?? { full: '', brief: '' };
  })();

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
   * Whether the cache precision itself is a stand-in — a claim about the scenario, not about any
   * row, so it needs no `some` over the cells.
   *
   * Gated on a cell having been evaluated for the same reason `substitutedCells` is: a grid where
   * nothing ran produced no figure to caveat, and a warning explaining arithmetic that was never
   * performed is noise that teaches people to skip the warning that matters.
   */
  const kvSubstituted =
    kvSubstitutionFor(runtime, config.kvPrecision) !== undefined && summary.anyEvaluated;

  /**
   * The header: its labels, and the space the rotation needs for them in *both* axes.
   *
   * One object, computed once, because the bug this replaces was two derivations of one quantity.
   * A 45-degree rotation costs `sin(45) × label` of height and `cos(45) × label` of width, and
   * those are the same number — but only the height was ever reserved. So the band was 246px tall
   * at every viewport while the four longest names leaned up-and-*right* past the last column, out
   * of the `overflow-x-auto` container: a grid that fits its panel exactly at 1440 and 1024 got
   * 142px of overflow and a scrollbar anyway, and the default view hid the names the 246px was
   * calculated from. The app paid a phone screen of vertical space for labels it then cut off (#64).
   *
   * The two lengths are the same number and are now written once. Which direction the width is spent
   * in is the other half of the repair, and it is argued under **Lean** below.
   *
   * **Height.** A fixed 96px was set for the names that existed then and is short for several
   * shipping ones — the catalog reaches 40 characters. The table sits in an `overflow-x-auto`
   * container, which clips vertically rather than scrolling, and the Mac Studio variants differ
   * only in the trailing capacity suffix that got cut, so two columns became indistinguishable —
   * the exact failure the rotation was introduced to fix.
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
   *
   * **The result is in `rem`, and that is the second time this number has been wrong for the same
   * reason.** The labels are `text-xs`, so their width scales with the root font size — while a
   * height in CSS pixels does not. At a 32px root the text doubled and the row it has to fit in
   * did not, so the container clipped the names again: the exact failure the rotation exists to
   * prevent, reintroduced at the one setting a low-vision reader would be using. 0.5rem per
   * character is 8px at the default root, so nothing moves there. Raised by Codex on PR #36 (#44).
   *
   * **Lean** is the same length, spent sideways — and the labels are turned to spend it *leftward*,
   * over the model-name column, which is the half of this that had to be got right.
   *
   * A trailing lane on the right cannot be made to work, and both versions were built and measured.
   * As `padding-right` it is non-negotiable, so at 1024px — the grid's own min-content is 857px
   * inside a 934px panel — it forces 65px of scrolling onto a grid that fits. As a yielding grid
   * track, `minmax(0, lean)`, it takes only the free space that happens to exist: fine at 1440 and
   * 1280, and between a 857px grid and its 920px painted extent there is *less free space than the
   * labels need*. Measured at 960px of viewport: container 870px, grid 857px — it fits — scrollWidth
   * 920px anyway, with "Threadripper PRO 7995WX" painted 50px outside the visible right edge. That
   * is #64 narrowed to a 60px window of viewport, not repaired.
   *
   * Leaning the other way has no width dependence at all. The space a left-leaning label needs is
   * the model-name column, which is already in flow and already inside the container, and — unlike
   * free space — it is measured from *text*, so it grows with the font exactly as the labels do.
   * The reservation below (`minWidth` on the stub header cell) is what makes that safe rather than
   * lucky: today the longest model name asks for 133px and the lean for 141px, so the guard is worth
   * 8px, and without it a catalog of short model names would lean labels past the container's left
   * edge — where, unlike the right, the overflow is not scrollable and the name is simply gone.
   */
  const headerBand = useMemo(() => {
    const columns = headerColumns(devices);
    const longest = Math.max(0, ...columns.map((c) => c.label.length));
    // sin(45) and cos(45) are one number, which is the whole point of computing it once.
    const lean = longest * 0.5 * Math.SQRT1_2;
    return { columns, height: `${lean + 1.25}rem`, lean: `${lean}rem` };
  }, [devices]);

  return (
    <section aria-labelledby={headingId} className="panel p-[min(1.25rem,5vw)]">
      <header className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-2">
        {/* The grid moves materially with context, concurrency, prompt and KV precision — the fit
            counts and the throughput colours all change — and the heading named only the format
            and the runtime. So a screenshot of the Matrix carried no record of the request it
            answers, and two of them taken at different settings are indistinguishable. Every input
            that moves a cell is stated.

            Still stated now that the sliders sit at the top of the page rather than below this
            section (#66). Being able to scroll up to the controls is not the same as the panel
            saying what it was computed at: this grid is 17 rows tall, a screenshot carries no
            scrollback at all, and the heading is also the section's `aria-labelledby` target. */}
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
        <PanelCount count={summary.runnable} total={summary.total}>
          combinations run
        </PanelCount>
        {/* Outside the h2, which is this section's `aria-labelledby` target — the accessible name
            is computed from its whole subtree, so a sentence nested in there is read out every
            time the landmark is announced. The workload belongs in the name; the caveat does not. */}
        <p className="basis-full text-sm text-[var(--color-text-faint)]">
          Rows are capped at each model’s own context limit, so a model that stops short of{' '}
          {tokens(config.contextTokens)} is scored at whatever it does accept.
        </p>
      </header>

      {/* One filter row above the grid, as the dataviz guidance puts it.

          `aria-describedby` on the group, not merely a paragraph under it: the hint below says what
          the selected measure *is*, and until it was wired up a screen-reader user entering this
          group heard "Colour the grid by, Does it fit, pressed" and nothing about headroom — the
          same gap the five Usage controls had, in the one place on the page that already had the
          sentence written (#80). Once per group rather than once per button, so switching measures
          does not re-read it three times. */}
      <fieldset className="mt-4" aria-describedby={measureHintId}>
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
        <p id={measureHintId} className="mt-1.5 text-xs text-[var(--color-text-muted)]">
          {MEASURES.find((m) => m.value === measure)?.hint} Switching between these rearranges which
          hardware looks good — that disagreement is the whole point.
        </p>
      </fieldset>

      <div className="mt-4 overflow-x-auto">
        {/* `role="grid"` rather than the native table role, because the cells are widgets a
            keyboard drives rather than data a reader browses — which is the distinction the two
            roles exist to draw, and what tells a screen reader to hand the arrow keys over. */}
        <table role="grid" className="w-full border-separate border-spacing-0.5 text-left text-xs">
          <caption className="sr-only">
            Every catalogued model against every shipping device, coloured by{' '}
            {MEASURES.find((m) => m.value === measure)?.label}. {summary.runnable} of{' '}
            {summary.total} combinations run.
            {/* The count's qualifier, in the channel where the legend line below is invisible —
                same conditional, same constant (#127). */}
            {summary.anySpilled &&
              ` Some of those run only by spilling weights to host RAM. ${HOST_RAM_UNCHECKED}`}
            {/* The same fact the struck headings and the legend carry, in the channel that has
                neither. This caption is the grid's only summary for a reader who cannot see it, and
                at #72's own URL it read "232 of 408 combinations run" with no further explanation —
                the collapsed reading the issue is about, said in the one place where the
                strike-through, the empty columns and the legend key are all invisible. (232, not the
                235 the same grid shows at Q4_K_M: the store coerces that selection to BF16 under
                vLLM, since vLLM does not read GGUF K-quants.) Stated only when the grid is in that
                state, the same rule the legend below follows. */}
            {undrivable.size > 0 &&
              ` ${undrivable.size} of the ${devices.length} device columns are hardware ${runtime.label} does not support at any size, struck through in the header: their cells are empty because of the runtime, not for want of memory.`}{' '}
            {/* **What each axis is sorted by, which is the one fact this grid's own headings cannot
                carry** (#79). Both orders were deliberate and unstated: 35 rows in download order
                with nothing naming the criterion — a row heading here is name-only, where the Bench's
                Model picker at least prints "N downloads/mo" under the selection — and a column
                sequence whose gaps now say *that* there is a boundary without saying what divides it.
                This is also the whole of the band channel for a reader who cannot see the gap, which
                is why it is one sentence with the mark rather than prose somewhere else. */}
            Columns run left to right in {bands.labels.length} bands — {bands.labels.join(', ')} —
            with a gap between them, and rows run most-downloaded first. This grid is a single tab
            stop: use the arrow keys to move between cells, Home and End for the ends of a row, and
            Control with Home or End for the ends of the grid.
          </caption>
          <thead>
            <tr>
              {/*
                  The model column, and the lane the rotated labels lean into — the same cell, which
                  is the point. `minWidth` is the sideways half of `headerBand`, so the column that
                  the leftmost label leans over is never narrower than the lean itself.

                  On today's catalog it is worth 8px: the longest model name asks for 133px and the
                  lean for 141px. It is not decoration. Overflow to the *right* of a scroll container
                  is at least reachable by panning; overflow to the left is not scrollable at all, so
                  a catalog of short model names would put the first device's name somewhere no
                  reader can get to. A reservation measured from text, guarding space measured from
                  text, both of which grow together when the font does.
                */}
              <th
                scope="col"
                className="sticky left-0 bg-[var(--color-surface)] pr-2 font-normal"
                style={{ minWidth: headerBand.lean }}
              >
                <span className="sr-only">Model</span>
              </th>
              {/* Iterated from `headerBand.columns` rather than `devices`, so the label a column
                  renders is the same string the band was measured from. Two loops over two lists
                  is how a reservation and the thing it reserves for come to disagree. */}
              {headerBand.columns.map(({ device, label }) => (
                <th
                  key={device.id}
                  scope="col"
                  // Fixed width, and the label taken out of flow below, so a long name cannot
                  // stretch its own column — "RTX PRO 6000 Blackwell" was three times the width
                  // of its neighbours and skewed the whole grid.
                  //
                  // A column opening a class band carries the separator, and it is `BAND_GAP`:
                  // whitespace in the panel's own colour, four times the `border-spacing` between
                  // squares. See the constant for why it is a border, why it is a gap, and why both
                  // lengths are in the spacing unit.
                  data-band-start={bands.separated.has(device.id) ? '' : undefined}
                  className={`relative p-0 align-bottom font-normal text-[var(--color-text-faint)] ${
                    bands.separated.has(device.id)
                      ? `${BAND_GAP} w-9 min-w-9 [@media(pointer:coarse)]:w-13 [@media(pointer:coarse)]:min-w-13`
                      : 'w-7 min-w-7 [@media(pointer:coarse)]:w-11 [@media(pointer:coarse)]:min-w-11'
                  }`}
                  style={{ height: headerBand.height }}
                  /**
                   * The runtime refusal, for a reader who is hearing this column rather than seeing
                   * it struck — an `aria-label` on the `th` itself, so it is announced once as the
                   * column's own name instead of appearing as a second child element.
                   *
                   * It keeps the device name, because the label beside it is deliberately shortened
                   * and the header's job is still to say which machine this is; a name that stated
                   * only the runtime would trade one missing fact for another.
                   *
                   * Deliberately *not* a `title`. Part of what #72 is about is that the distinction
                   * already existed in a native tooltip, one cell at a time, on a surface whose
                   * whole point is scanning — so a second tooltip would be the same mistake with a
                   * new subject. The strike-through is the channel a sighted reader gets, keyed in
                   * the legend below; this is the channel a screen reader gets.
                   *
                   * The same string the readout prints, from `columnReadout` rather than written out
                   * again here, so the sentence a reader hears and the one they can read are one
                   * claim about the column.
                   */
                  aria-label={undrivable.has(device.id) ? columnReadout(device) : undefined}
                  // The pointer's half of the same channel. The label below is shortened and its
                  // full form lived only in a `title`, which is #71's defect with a device name in
                  // it rather than a figure; the heading has no focus of its own to offer, so a
                  // sighted keyboard reader gets the full name from any cell in the column instead.
                  onMouseEnter={() => setHovered({ kind: 'column', deviceId: device.id })}
                  onMouseLeave={() => setHovered(null)}
                  /* The same reclaim the cells carry, for the same reason and by the same guard —
                     see the cell's `onMouseMove`. A heading is a hover-only surface with no focus of
                     its own, so a pointer resting in one and outranked by a keyboard step had *no*
                     route back except leaving the heading: worse here than on a cell, where at least
                     the reader can arrow to the thing they are pointing at. Fixing the cell alone
                     left two live instances of one defect, which is the shape this repo keeps
                     recording. */
                  onMouseMove={() =>
                    setHovered((claim) =>
                      claim?.kind === 'column' && claim.deviceId === device.id
                        ? claim
                        : { kind: 'column', deviceId: device.id }
                    )
                  }
                >
                  {/*
                      Rotated rather than truncated. Horizontally these clipped to "GeForc…" four
                      times over — a header that cannot distinguish its own columns is worse than
                      none, and the names are what make the grid readable at all.

                      The full name stays in the `title`, and in every cell's `aria-label` below, so
                      what the shortening drops is one hover or one screen-reader cell away.

                      Anchored bottom-*right* and turned clockwise, so each label ends at its own
                      column and runs up-and-left over the grid it belongs to, rather than starting
                      at its column and running up-and-right past the last one. Geometry, not taste:
                      text that ascends left-to-right has to lean right, and to the right of the last
                      column there is nothing but the edge of the scroll container — which is how a
                      grid that fits its panel came to report 142px of overflow (#64). Leaning the
                      other way spends the same length over the model-name column, which is inside
                      the container and reserved for it above.

                      Struck through where the runtime cannot drive the machine. Dimming was the
                      other half of the suggestion in #72 and there is nothing left to dim with:
                      these labels are already `--color-text-faint`, the faintest ink the palette
                      has, so a second step down would be unreadable rather than recessive. A strike
                      is a non-colour channel in any case, which is what this needs — the cells
                      beneath it carry no ink at all now, and colour alone was never going to be the
                      thing that says why.
                    */}
                  <span
                    className={`absolute right-1/2 bottom-1 origin-bottom-right rotate-45 whitespace-nowrap ${
                      undrivable.has(device.id) ? 'line-through' : ''
                    }`}
                    title={device.name}
                  >
                    {label}
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
                  // Truncated at 9rem, so the name of any model past that length — and the parameter
                  // count of every one of them — was a mouse-only fact. Same string in all three
                  // channels.
                  title={rowReadout(model)}
                  /**
                   * The third channel, and the one the readout below cannot serve.
                   *
                   * The line under the grid is `aria-hidden` on purpose — it copies a cell's own
                   * accessible name, so announcing it would say every cell twice — and these headings
                   * are not focusable, so the *parameter count* was left with a pointer and nothing
                   * else: a cell's sentence names the model but never its size. The same shape as the
                   * column heading's own `aria-label` above, from the same one derivation, and it is
                   * announced once per row change rather than once per cell, since a grid announces a
                   * row header when the reader enters a new row.
                   *
                   * A sighted keyboard reader still has only the model *name*, which every cell in the
                   * row states. Closing that properly means making row headers part of the arrow-key
                   * grid — which is what the pattern says they are, and it moves the column indexing
                   * `matrix-grid.spec.ts` and #52's roving stop are pinned to. Filed rather than
                   * smuggled into this change.
                   */
                  aria-label={rowReadout(model)}
                  onMouseEnter={() => setHovered({ kind: 'row', modelId: model.id })}
                  onMouseLeave={() => setHovered(null)}
                  /* See the column heading above: one class, three surfaces. */
                  onMouseMove={() =>
                    setHovered((claim) =>
                      claim?.kind === 'row' && claim.modelId === model.id
                        ? claim
                        : { kind: 'row', modelId: model.id }
                    )
                  }
                >
                  {model.name}
                </th>
                {cells[r].map((cell, c) => (
                  // The band separator, on the same columns as the header's and by the same
                  // mechanism, so the gap runs the full height of the grid rather than stopping at
                  // the labels. On the `td` rather than on the button inside it: the button's borders
                  // are already two other channels — dashed for "measured and does not fit", warning
                  // for "past the default allocation" — and a third meaning on the same property is
                  // how a legend comes to key a mark that means two things.
                  <td
                    key={devices[c].id}
                    role="gridcell"
                    data-band-start={bands.separated.has(cell.deviceId) ? '' : undefined}
                    className={`p-0 ${bands.separated.has(cell.deviceId) ? BAND_GAP : ''}`}
                  >
                    <button
                      type="button"
                      ref={(node) => {
                        cellRefs.current.set(`${r}:${c}`, node);
                      }}
                      // The roving half of the pattern: exactly one cell is in the tab sequence,
                      // and it is wherever the reader last was. `activeRow`/`activeCol` are the
                      // clamped pair, so a grid that shrank under a remembered position still
                      // offers a stop rather than none at all.
                      tabIndex={r === activeRow && c === activeCol ? 0 : -1}
                      onKeyDown={(event) => onCellKeyDown(event, r, c)}
                      /**
                       * Loads the *whole* scenario the cell was scored under, not just the pair.
                       *
                       * Setting only model and device meant a dense-model cell scored at the
                       * Q4_K_M substitute landed in a Bench still holding MXFP4 — which `coerce`
                       * then replaced with BF16, so the grid and the detail view disagreed about
                       * the same square. The context and the single-device count matter for the
                       * same reason.
                       */
                      // Before the browser's own focus moves the line — see `gesture`.
                      onPointerDown={(event) => {
                        gesture.current = {
                          inspected:
                            // The record only exists if a pointer hovered this cell without being
                            // pressed, and it has to be *this* pointer — see `pointerOver`.
                            (pointerOver.current?.cell === `${r}:${c}` &&
                              pointerOver.current.pointerId === event.pointerId) ||
                            (target?.kind === 'cell' && target.row === r && target.col === c),
                        };
                      }}
                      onClick={(event) => {
                        // So the tab stop follows the reader: leaving the grid and coming back
                        // returns to the cell they last used, not to the top-left corner. Done
                        // before the guard below, because a closed cell is still where the keyboard
                        // should resume from — it is inert, not absent.
                        setActive([r, c]);
                        /**
                         * A pointer that has not been shown this cell's figures inspects first — see
                         * `gesture`.
                         *
                         * Only when the readout is not already this cell's, so the second tap falls
                         * through and loads it. `focused` rather than `hovered` because a tap moves
                         * focus too and `onFocus` clears the pointer's claim, so writing the hover
                         * channel here would be overwritten by the browser's own focus a moment
                         * later — the two-state rule the readout already documents, met from a
                         * direction it had not been.
                         */
                        /**
                         * A keyboard activation is identified positively, not by the absence of a
                         * snapshot (found in review, twice).
                         *
                         * `click` from Enter or Space carries `detail === 0`; every pointer-generated
                         * click carries a count. Inferring "keyboard" from `gesture.current === null`
                         * meant any pointer sequence that ended without a click left a snapshot that
                         * the next keypress inherited — and the exits are more numerous than they
                         * look: `pointercancel` when a touch becomes a scroll, and a release outside
                         * the cell, where implicit pointer capture delays `pointerleave` until after
                         * `pointerup` so it arrives with `buttons === 0` and is indistinguishable
                         * from a completed tap. Two rounds went into chasing those exits. Asking the
                         * click what produced it ends the class: a stale snapshot cannot reach a
                         * keyboard press, and a pointer press overwrites it at `pointerdown`.
                         */
                        const started = event.detail === 0 ? null : gesture.current;
                        gesture.current = null;
                        if (started !== null && !started.inspected) {
                          setHovered(null);
                          setFocused({ kind: 'cell', row: r, col: c });
                          return;
                        }
                        // A closed column has no scenario to load. Argued at `aria-disabled` below;
                        // the short version is that adopting this pair sets the Bench to a
                        // configuration it can only blank, several sections above where the click
                        // happened, from a square with nothing drawn in it.
                        if (undrivable.has(cell.deviceId)) return;
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
                      /**
                       * Both halves of the readout, and the reason there are two.
                       *
                       * `onFocus` is the half that fixes the keyboard: the roving tab stop and the
                       * arrow keys move focus, so this fires on every step across a row and the
                       * sentence appears without a pointer, a dwell or a click. `onMouseEnter` is
                       * what a mouse reader already expected from the `title`, minus the second of
                       * delay and minus covering the neighbours they are comparing against.
                       *
                       * `onBlur` and `onMouseLeave` clear only their own channel, so the line falls
                       * back rather than blanking; `onFocus` clears both, because a pointer that has
                       * stopped moving never reports leaving. Both halves of that are what keep the
                       * line agreeing with the visible ring — see the two states above.
                       */
                      onFocus={() => {
                        setFocused({ kind: 'cell', row: r, col: c });
                        // And the pointer's claim expires here, because a resting pointer fires no
                        // `mouseleave` of its own — argued at the two states above. Without this the
                        // line prints one cell while the ring is drawn on another, indefinitely.
                        setHovered(null);
                        /*
                         * `pointerOver` deliberately survives this, and the round trip is worth
                         * recording. Clearing it here fixed a returning *tap* — a completed tap
                         * leaves a compatibility hover behind, so tapping back after a keyboard move
                         * read as already-inspected and committed — and broke a stationary *mouse*,
                         * which emits no new `mouseenter` and would then need two clicks after any
                         * keyboard use. Both are real, and expiry cannot tell them apart because it
                         * is asking the wrong question. The record's *provenance* is the question,
                         * and it is asked at `pointerdown` instead.
                         */
                      }}
                      onBlur={() => setFocused(null)}
                      // `pointerenter` rather than `mouseenter` for the record, because only the
                      // pointer event carries `buttons` — see `pointerOver`. The readout state stays
                      // on the mouse events, which are what the hover rule was written against.
                      onPointerEnter={(event) => {
                        if (event.buttons === 0) {
                          pointerOver.current = {
                            cell: `${r}:${c}`,
                            pointerId: event.pointerId,
                          };
                        }
                      }}
                      onPointerLeave={(event) => {
                        if (pointerOver.current?.pointerId === event.pointerId) {
                          pointerOver.current = null;
                        }
                      }}
                      onMouseEnter={() => setHovered({ kind: 'cell', row: r, col: c })}
                      onMouseLeave={() => setHovered(null)}
                      /**
                       * How the pointer gets its claim *back*, which `onMouseEnter` alone cannot do
                       * (found in review).
                       *
                       * `onFocus` expires the hover claim, for the reason argued above. But a pointer
                       * already resting inside this cell fires no further `mouseenter` — that event
                       * needs a boundary crossing — so after focus moved elsewhere by keyboard, moving
                       * the mouse within the cell it is already in reclaimed nothing and the line went
                       * on printing the focused cell. Last-input-wins held in one direction only, and
                       * the reader had to leave the cell and come back to fix it.
                       *
                       * The functional update is what makes this affordable: `mousemove` fires per
                       * pixel, and returning the *same* object when the claim is already this cell
                       * lets React bail out of the re-render rather than reconciling 700 cells on
                       * every mouse move. Only the first move after a loss allocates.
                       */
                      onMouseMove={() =>
                        setHovered((claim) =>
                          claim?.kind === 'cell' && claim.row === r && claim.col === c
                            ? claim
                            : { kind: 'cell', row: r, col: c }
                        )
                      }
                      /**
                       * The native tooltip stays, now as an echo rather than the only route.
                       *
                       * It is the one channel that appears *at* the cell, and this grid is 17 rows
                       * tall — a reader hovering the top of it on a laptop can have the readout below
                       * the fold. What made it a defect was being alone, not being present.
                       */
                      title={sentences[r][c].full}
                      // The full form always, at every width: the accessible name is the one channel
                      // with no axis headings beside it, so the preamble the readout can drop below
                      // `sm` is exactly what a screen-reader user has instead of them.
                      aria-label={sentences[r][c].full}
                      aria-current={isCurrent(cell) ? 'true' : undefined}
                      /**
                       * A square with nothing drawn in it is not a control.
                       *
                       * Narrowing the dashed border to `evaluated` (argued in the class list below)
                       * leaves a closed column's cells with no ink whatsoever — `fill` already
                       * returns `transparent` for anything that does not run — so on today's
                       * catalog vLLM produced 272 enabled 28px buttons a reader cannot see, each
                       * one in the arrow-key sequence and each one silently adopting a scenario
                       * several sections above that the Bench can only blank.
                       *
                       * Restoring a hairline would not have answered it. `tokens.ts` states the
                       * rule: "a control's boundary is what identifies it as interactive, so it
                       * needs the 3:1 non-text minimum *before* it is focused" — and it records
                       * `--color-border` at 1.18:1 on the raised fill, which is why that token is
                       * the panel edge and `--color-control-border` exists separately at 3.41:1.
                       * At that contrast there is no ink here that both reads as "closed" and stays
                       * clear of "measured and over the ceiling", so the honest channel is state
                       * rather than ink — and it is the channel the rest of the app already uses on
                       * an unsupported pairing: BudgetBar draws no stack, the Bench blanks its
                       * tiles.
                       *
                       * `aria-disabled` rather than `disabled`, because a disabled button takes no
                       * focus: the arrow keys would stop dead at a struck column and a screen
                       * reader would lose the per-cell sentence, which is the one channel that says
                       * *which* machine and *which* runtime. Focusable, hoverable, announced
                       * unavailable, and inert.
                       */
                      aria-disabled={undrivable.has(cell.deviceId) ? 'true' : undefined}
                      // 28px squares two pixels apart are under the 44px `marks.hitTarget` this
                      // repo declares, and with hundreds of neighbours a touch user loading the
                      // wrong scenario is the likely outcome rather than the unlucky one. Coarse
                      // pointers get the full target; a mouse keeps the dense grid, which is what
                      // makes the comparison legible in one screen.
                      //
                      // The selected square is marked *inside* its own box, and the focus ring
                      // stays outside it, because the two used to be the same mark. Selection was
                      // `ring-2 ring-[accent] ring-offset-1` and focus is `focus:ring-2
                      // ring-[accent]` — the same channel, the same width and the same colour, so
                      // focusing the marked square changed nothing whatsoever: a 1:1 change
                      // contrast, which is #67's 1.95:1 select border in its most extreme form. And
                      // it is not a corner case. Clicking a cell makes it both the selection and
                      // the roving tab stop, so the marked square is exactly where Tab lands when a
                      // reader comes back to the grid.
                      //
                      // An inner frame now says "this is the scenario the Bench is showing" and an
                      // outer ring says "this is where the keyboard is", and neither can stand in
                      // for the other. It also stops the mark bleeding over the 2px `border-spacing`
                      // onto the neighbouring squares, which the offset ring did.
                      //
                      // **The inner frame is two tones, and the second one is not decoration.**
                      // Moving it inside the cell moves it off the panel surface and onto the ramp,
                      // and the accent is not readable there: against the seven steps of
                      // `sequential` it measures 2.00, 1.48, 1.06, 1.38, 2.04, 3.07 and 4.52:1, so
                      // an accent-only frame sits below the 3:1 non-text minimum on **304 of the 408
                      // squares the grid held when that was measured** — including the default
                      // selection, on `#3987e5` at 1.38:1, and the grid is 714 squares now. On
                      // `#6da7ec` the two sit 0.022 apart in relative luminance (0.347
                      // against 0.369) and measure 1.06:1 — a pure hue difference, which is #67's
                      // own failure mode restated as a resting state, and gone in greyscale or to a
                      // deuteranope. So the accent band carries a 1px `--color-surface`
                      // separator on its inner edge — the dataviz surface ring, and the same
                      // two-tone trick `Envelope.tsx` uses for its "you are here" mark on the same
                      // ramp ("A ring, not a filled dot: the cell's own colour has to stay readable
                      // underneath it"). The accent is then bounded by surface on both sides — the
                      // 2px `border-spacing` outside, the separator inside — at 7.14:1, and *one of
                      // the two tones* clears 3:1 against every step of the ramp: the separator on
                      // the five light steps (14.26 to 3.50:1), the accent on the two dark ones
                      // where the separator disappears (3.07 and 4.52:1). Worst case 3.07:1, zero
                      // squares below the bar. `tokens.ts` validates the accent against `surface`
                      // and never against the ramp, so a mark drawn on a cell has to bring its own
                      // guarantee; `App.test.tsx` measures it over every fill the grid paints.
                      //
                      // The separator rides the `--tw-shadow` slot rather than a second inset ring
                      // because Tailwind composes one box-shadow chain in a fixed order —
                      // `inset-shadow, inset-ring, ring-offset, ring, shadow` — and only the last
                      // slot paints *under* the accent. A 3px inset there shows through in the 2–3px
                      // band the 2px accent does not cover, which is what keeps the accent 2px wide
                      // instead of 1px. (Utility names are spelled out only in the class list below:
                      // Tailwind scans comments too, and a bracketed example in prose compiles to a
                      // rule of dead CSS.)
                      //
                      // **The dashed border keys one refusal, not both** (#72). `cell.runs` was the
                      // condition, so a pair the runtime cannot load at all wore the same swatch as
                      // a pair whose bytes were counted and came up short — byte-identical markup
                      // for two states that need opposite advice, repeated down a whole column until
                      // it read as a finding about the machine. `evaluated` is the question the
                      // border actually answers: was this judged on its numbers. A cell refused on a
                      // categorical ground was not, so it gets no ink at all, and the struck column
                      // heading above says why once instead of once per row. That also makes the
                      // "will not run" key below true of exactly the cells that wear it.
                      //
                      // The cursor is the pointer's half of the `aria-disabled` above — the one
                      // channel a mouse user gets before they click, on a square that has no
                      // boundary to look at.
                      className={`h-7 w-full scroll-mb-20 rounded-sm focus:ring-2 focus:ring-[var(--color-accent)] focus:outline-none sm:scroll-mb-15 lg:scroll-mb-10 [@media(pointer:coarse)]:h-11 ${
                        undrivable.has(cell.deviceId) ? 'cursor-not-allowed' : ''
                      } ${
                        !cell.runs && cell.evaluated
                          ? 'border border-dashed border-[var(--color-border)]'
                          : ''
                      } ${cell.raiseCeilingWouldHelp ? 'border-[var(--color-warning)]' : ''} ${
                        isCurrent(cell) ? CURRENT_CELL_MARK : ''
                      }`}
                      style={{ background: fill(cell, measure, domain) }}
                    />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/*
          The readout. Reserved at one line so filling it moves nothing, the way `BudgetBar`'s hint
          is — `min-h` on an empty paragraph, rather than a paragraph that appears.

          The reservation is in `rem` and so is the line it reserves for: `min-h-[1.25rem]` is exactly
          `text-sm`'s line height, so the two grow together when the root does. A height in pixels
          under text that scales is the reservation that quietly stops being one, which is the header
          band's #44 all over again in a smaller box.

          `aria-hidden`, and this is the one place this file departs from that pattern. `BudgetBar`
          marks its hint `aria-live="polite"` because the sentence exists nowhere else: its legend
          items are named "Weights 14 GiB" and the explanation is not part of that name. Here the
          sentence *is* the focused cell's accessible name, verbatim — so a live region carrying it
          would say every cell twice, once as the name and once as the announcement, on every press
          of an arrow key across a row — 42 columns as the catalog stands at this commit, so a walk
          across one goes from 42 sentences to 84. The channel this adds is the visual one, which is
          precisely the channel the three readers in #71 were missing; the spoken one was already
          there, and is the reason the duplicate would be so expensive.

          `tabular`, because the figures change under the sliders and a readout whose digits shift
          width as it counts is the thing `index.css` reserves that class for. */}

      {/* A ramp legend, since a continuous scale has no discrete keys to list.
          `flex-wrap`, because several of the seven entries are whole sentences and the row does not
          fit a phone.
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
            one setting a reader most needs it not to. `min()` yields instead.

            **The endpoints are what make this a scale rather than an ordering** (#71). `worse` and
            `better` alone say which way to read the ramp and nothing about what it spans, so a
            mid-blue under "How fast" could be 20 tok/s or 200 — and since `fill` log-scales, the
            colour is deliberately a rank, which leaves no other route to a magnitude. They are the
            extremes of the values actually on the grid, not a per-step key — and for `fit` and
            `decode`, whose ramps anchor at zero, the lowest cell need not paint the first step at
            all. Under `ttft` the domain is the grid's own span and it does (#97). The claim is the
            span either way.

            The two throughput figures come off the cell at that end rather than out of the ramp value,
            which is argued at `MeasureRange` — for TTFT they would otherwise be a floating-point round
            trip, and printed the wrong way round if anyone sorted them by the number instead of by
            rank. `fit`'s low figure is the ramp's own value and says so at `rampEnd`: every offloaded
            cell scores exactly zero headroom, so the dark end of that ramp is a population rather than
            a cell.

            **Two figures in this row is why the floor moves from 12rem to 20rem, and why the ramp now
            carries one of its own.** The ramp is still the only item here with a zero flex basis, so
            it is still the only thing that collapses, and it now shares its line with about 190px of
            text: "worse 0.4 tok/s" and "1449 tok/s better" at the widest the catalog reaches. 20rem
            leaves it ~100px on that line instead of the 13.6px #34 measured, `min(…, 100%)` keeps
            that from being a floor a 320px viewport cannot meet, and the ramp's own floor is what
            makes it take a line rather than vanish where even 20rem is too tight — a zero-basis item
            on a shared line takes only the space left over, which is the whole lesson of #34's second
            half.

            **The `nowrap` protects the figure and its unit, and nothing more — `PanelCount`'s rule
            from #35, which is where that shape was settled.** The first version of this put the
            direction word and the figure in one non-wrapping item, which makes each end's min-content
            the whole label: 17 characters at the widest the scenario space reaches ("1011 tok/s
            better", measured across three runtimes, every catalogued format, 4K/32K/128K of context
            and 1/8/128 users). A 320px viewport leaves about 262px inside this panel — 12.8px of page
            padding and 16px of panel padding a side — and 17 characters of the 24px monospace a 200%
            reader gets is roughly 245px of it: true today, true by about a character, and on a
            quantity that moves whenever a device is added. Protecting the figure alone takes the
            floor to the longest *figure* instead ("1011 tok/s", 10 characters), and lets "worse" fall
            to its own line in the one setting where the alternative is a label leaving the panel —
            which is exactly the trade `PanelCount` makes for "12 of 425" and the noun after it.
            Measured in `e2e/matrix-legend.spec.ts` at the decode measure, where the longest labels
            are, and at 200% in `e2e/reflow.spec.ts` with the stress font on `--font-mono` as well,
            since `.tabular` resolves that variable and the sweep only ever stressed `--font-sans`. */}
        <span className="flex min-w-[min(20rem,100%)] flex-1 flex-wrap items-center gap-x-2 gap-y-1">
          {/* The word is prose and wraps; the figure is a figure and does not. */}
          <span className="tabular">
            worse{' '}
            {range && <span className="whitespace-nowrap">{rampEnd(range.low, measure)}</span>}
          </span>
          <span
            aria-hidden="true"
            className="flex h-3 min-w-[min(6rem,100%)] flex-1 overflow-hidden rounded-sm"
          >
            {magnitudeRamp.map((step) => (
              <span key={step} className="flex-1" style={{ background: step }} />
            ))}
          </span>
          {/* Nothing runs on this grid under some scenarios — every cell a hole, `fill` returning the
              empty colour for all of them — and then there is no span to state. The words stay,
              because they still say which way the ramp reads. */}
          <span className="tabular">
            {range && <span className="whitespace-nowrap">{rampEnd(range.high, measure)}</span>}{' '}
            better
          </span>
        </span>
        <span className="flex items-center gap-1.5">
          <span
            aria-hidden="true"
            className="inline-block h-3 w-3 rounded-sm border border-dashed border-[var(--color-border)]"
          />
          will not run
        </span>
        {/* The band gap, keyed rather than left to be inferred.
            The gap is a mark this legend keys every neighbour of — the dashed border, the struck
            heading, the selection ring, the warning border — and it shipped with its only sentence
            inside a `sr-only` caption, so a screen-reader user was told the columns are grouped and a
            sighted reader met two channels of whitespace with nothing on the page naming them. That is
            #73's asymmetry, in the same direction and on the same surface. A boundary legible only to
            someone who already knows that `dgx-spark` is not a discrete GPU is not a channel a legend
            gets to rely on.

            The bands are named here in order, so the key also answers the question the gap raises
            rather than only labelling it — which makes this the visible half of the caption's column
            sentence rather than a second copy of it. The words come from `DEVICE_CLASS_LABELS`, the
            same table the picker's `<optgroup>` headings come from.

            The sample *is* the mark, like the struck heading above: two squares in the empty-cell
            colour with the real gap between them, `w-2` being the `2 × --spacing` the columns spend.
            Conditional like its neighbours — one class band is a grid with no boundary in it, and a
            key for a mark that appears nowhere is worse than prose. */}
        {bands.labels.length > 1 && (
          <span className="flex items-center gap-1.5">
            <span aria-hidden="true" className="inline-flex items-center">
              <span className="h-3 w-1.5 rounded-sm bg-[var(--color-grid)]" />
              <span className="h-3 w-2" />
              <span className="h-3 w-1.5 rounded-sm bg-[var(--color-grid)]" />
            </span>
            a gap between columns — the hardware class changes: {bands.labels.join(', ')}
          </span>
        )}
        {/* The other refusal — the one the swatch above used to absorb (#72).
            A state the grid is really in gets a line, and only when it is in it: the same rule the
            four conditional keys below follow, and the reason this cannot simply be listed
            unconditionally beside "will not run". Under llama.cpp there is no undrivable column and
            no strike on any heading, so a key for one would explain a mark that appears nowhere.

            The sample *is* the mark rather than a swatch beside it, which is the only honest key for
            a text treatment — and it is not `aria-hidden` like the swatches are, because struck text
            has no swatch to hide: hidden, the sentence would start mid-clause, and read out, it
            names the thing a screen reader cannot see. The `th`'s own `aria-label` is the other end
            of that channel.

            Wording borrowed verbatim from `Envelope.tsx`'s `unsupported` hint, with the runtime
            named because this grid spans hardware the Envelope's does not. That sentence has already
            been through review as the counterpart to "will not run"; writing a fourth version of it
            here is how the two surfaces would come to disagree about the same refusal. */}
        {undrivable.size > 0 && (
          <span>
            <span className="text-[var(--color-text-faint)] line-through">
              a struck column heading
            </span>{' '}
            — {runtime.label} does not support this hardware, at any size
          </span>
        )}
        {/* The selection mark, which had no key — the third mark in this app drawn on top of a fill
            with nothing on the page naming it (#73; the budget bar's ceiling rule and the Envelope's
            ring were the other two). `aria-current` names it for a screen reader and the accent hue
            says "live" to anyone who already knows the palette, which is not a channel a legend gets
            to rely on. It is not only a click acknowledgement either: the ring appears when the
            *controls* above put the Bench on a cell, so a reader can meet it without having touched
            the grid.

            Shown only when the grid contains it, the same rule as its neighbours — `isCurrent` is
            false for every cell on a linked rig, since these are all scored at one device. */}
        {marksCurrent && (
          <span className="flex items-center gap-1.5">
            {/* The sample is the mark — `CURRENT_CELL_MARK`, the same classes the cell paints —
                over a mid-ramp fill, since the frame's whole design is about staying readable on
                the ramp rather than on the panel surface. */}
            <span
              aria-hidden="true"
              className={`inline-block h-3 w-3 rounded-sm ${CURRENT_CELL_MARK}`}
              style={{ background: magnitudeRamp[Math.floor(magnitudeRamp.length / 2)] }}
            />
            the cell the Bench above is set to
          </span>
        )}
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
        {/* The cache axis, and a separate line rather than a clause on the one above.
            The two conditions are independent — this grid can be scored entirely at native
            formats and still charge an unmeasured width to every cache on it — so a combined
            sentence would be true of a state the grid is not in.

            "Every *scored* cell", and the qualifier is load-bearing rather than throat-clearing.
            The quantifier is stronger than the weight legend's "some rows", because the cache
            precision comes from the scenario rather than from the per-row format substitution —
            but it is not "every cell", which is a claim about the whole grid and false on most
            of it. Under MLX the grid still carries every shipping device while only the Apple
            columns are evaluated at all, so a sentence saying "every cell" describes NVIDIA and
            AMD columns that were never priced. Raised by Codex on PR #37. */}
        {kvSubstituted && (
          <span className="text-[var(--color-warning)]">
            every scored cell’s cache charged at {kv}’s nominal width, which {runtime.label} has not
            been measured at
          </span>
        )}
        {/* What "combinations run" is counting, when it counts a spilled cell (#127). The panel's
            count and every spilled cell's "runs" were the two surfaces promising a load the
            engine never checked — planPlacement sizes the spill with no host-RAM input at all —
            while the Envelope's legend and Telemetry's tile both carried the qualifier. One line
            at panel level, keyed like its neighbours on the state actually being in the grid;
            the constant rides verbatim, since a near-copy per panel is the drift it exists to
            prevent. */}
        {summary.anySpilled && (
          <span>
            some combinations run only by spilling weights to host RAM. {HOST_RAM_UNCHECKED}
          </span>
        )}
        {/* A default allocation and a hardware limit are not the same answer, and this grid is
            read as a shortlist. DeepSeek V3 at Q5_K_M is past the 512 GB Mac Studio's 384 GiB
            default and inside the 512 it can be tuned to — struck off the list over a checkbox,
            when the Envelope and Telemetry both kept the distinction. Shown only when the grid
            actually contains one, so the legend does not explain a state nobody is looking at. */}
        {summary.anyRaiseable && (
          <span className="flex items-center gap-1.5">
            <span
              aria-hidden="true"
              className="inline-block h-3 w-3 rounded-sm border border-dashed border-[var(--color-warning)]"
            />
            past the default allocation, which this machine lets you raise
          </span>
        )}
      </div>
      {/* **Last in the panel, and sticky, and those two together are the fix** (#71, both halves
          found in review).

          Sticky, because the paragraph used to sit after a 17-row table: a keyboard reader arrowing
          across the first rows had the figures rendered below the fold, and the cell's native `title`
          does not cover them — it needs a pointer and a dwell, so it serves the mouse reader and not
          the one who had nothing before this. `bottom-0` on an element in flow keeps it on screen
          without taking it out of the layout.

          Last in the panel, which is necessary and — as review pointed out — **not sufficient, and my
          first rationale for it was wrong.** "Nothing follows it to push" is true inside this panel
          and false on the page: a paragraph whose height changes changes this section's height and
          shoves whatever is downstream of it down. When #71 landed that was the Usage panel, which
          `Bench.tsx` rendered immediately after `<Matrix>`; #66 moved those controls to the top of the
          page, so today it is the MoE aside and the document's own height. Being last removes the
          legend from the blast radius; it does not remove the page, and which panel happens to be
          downstream is not something this reservation should depend on.

          So the height is **reserved per breakpoint**, from measurement rather than taste. The
          sentence is a full model-and-device line, and its widest rendering — a spilled cell's, which
          since #127 carries `HOST_RAM_UNCHECKED` as a second sentence — is 80px at 320 and 60 at
          390 (the brief form, unchanged by the qualifier), 60 at 640 and 768, 40 at 1024 and up:
          four lines, three, three, two. The reservation follows that: `5rem` below `sm`, `3.75rem`
          at `sm`, `2.5rem` at `lg`. It costs 60px of reserved blank space on a phone — and one
          more line at `sm` and `lg` than before #127, which is the measured price of the spilled
          qualifier — and buys a layout that does not move as the reader arrows across a row.

          A constant measured from today's device names is exactly the trap #44 records — #78
          lengthened them in this same sweep — so it is not left to hold by luck: `matrix-readout`
          sweeps every cell at 320 and 640 and fails if any sentence renders taller than the space
          reserved for it. A longer name added later breaks that test rather than the layout.

          `pointer-events-none`, because a pinned opaque bar over a grid of buttons otherwise
          swallows clicks meant for the cells behind it, and this paragraph is `aria-hidden`
          decoration with nothing to click. The cells carry `scroll-mb` to match the reservation, so
          the browser's own focus scrolling leaves the bar's height clear instead of parking the
          focused cell behind it.

          No vertical padding: `min-h` is a border-box floor, so padding sits inside it while the
          line is empty and adds to it once there is text — 4px of movement per sentence, which is
          the thing this is all for. */}
      {/* **Sticky to the bottom of the viewport while the grid is on screen** (found in review).

          The comment on the cell's `title` above concedes that this grid is 17 rows tall and a reader
          at the top of it can have the readout below the fold — and treats the native tooltip as the
          channel that covers that case. It does not: `title` needs a pointer and a dwell, so a sighted
          *keyboard* reader arrowing across the first rows had the figures rendered somewhere they
          could not see, which is #71's defect surviving its own fix for the reader who had the least
          before it.
          `sticky` rather than `fixed`, and `bottom-0` on the element that is already in flow, so it
          occupies its own space in the layout and reflows nothing: the paragraph sits where it always
          did and merely stops scrolling out of view while the panel is. `min-h` already reserved the
          line against text scaling, which is what keeps this from shifting the legend below it as the
          sentence appears and disappears.
          A surface of its own — `--color-surface` plus the panel's horizontal padding pulled back out
          with negative margins — because a line of text pinned over a scrolling grid of coloured
          squares is unreadable against half of them. No vertical padding with it: `min-h` is a
          border-box floor, so padding is inside it while the line is empty and *adds* to it once
          there is a line of text — 4px of legend movement every time a sentence appears, which is
          precisely the reflow the reservation exists to prevent. The sibling test caught it. */}
      <p
        aria-hidden="true"
        className="tabular pointer-events-none sticky bottom-0 -mx-[min(1.25rem,5vw)] mt-3 min-h-[5rem] bg-[var(--color-surface)] px-[min(1.25rem,5vw)] text-sm text-[var(--color-text)] sm:min-h-[3.75rem] lg:min-h-[2.5rem]"
      >
        {/*
          Both forms in the DOM, one displayed. See `tooltip`'s `brief` for the arithmetic.

          `data-readout` carries no style and exists so a spec can find each form without naming the
          utility that hides it — the same reason `data-band-start` exists on the class-band columns.
          It also gives jsdom, which resolves no media query and therefore renders both, a way to ask
          about one: without it the unit suite reads the two sentences concatenated.
        */}
        <span data-readout="brief" className="sm:hidden">
          {readout.brief}
        </span>
        <span data-readout="full" className="hidden sm:inline">
          {readout.full}
        </span>
      </p>
    </section>
  );
}

/**
 * The trailing qualifier a catalog name carries — `(12-ch DDR5-4800)`, `(512 GB)`, `(GB10)`.
 *
 * What the brackets hold is a *spec* rather than an identity: the memory configuration, the capacity
 * variant, the SoC. It belongs in the name the tooltip and the cell labels use, and it is what took
 * the longest header label to 40 characters.
 */
const QUALIFIER = /\s*\(([^)]*)\)\s*$/;

/**
 * The label each column shows: as short as it can be while still naming its own column.
 *
 * Two rules, in order.
 *
 * **The vendor line goes**, always. "GeForce RTX 5090" and "GeForce RTX 5080" differ in one
 * character at the end, so a truncating header shows the same string for both.
 *
 * **The qualifier goes too — but only where the rest of the name is already unique.** That condition
 * is the whole function. Stripping it unconditionally is the obvious version and it reintroduces the
 * defect the rotation exists to prevent: the three Mac Studio M3 Ultra rows differ *only* in their
 * capacity, so they would collapse to one string three columns wide, and a header that cannot
 * distinguish its own columns is worse than none. Where the qualifier is load-bearing it comes back,
 * minus the brackets and the space before the unit — punctuation that carries nothing in a 45-degree
 * label and costs three characters of band on every column.
 *
 * Computed over the rendered set rather than name by name, because uniqueness is a property of the
 * set: the same row shortens differently depending on what else is on the grid, and a catalog
 * addition that collides with an existing stem lengthens *both* labels rather than quietly making
 * one of them ambiguous.
 *
 * On today's catalog this takes the longest label from 40 characters to 25 and the reserved band from
 * 246px to 161px with every column still distinguishable — asserted in `App.test.tsx`, and
 * geometrically in `e2e/matrix-header.spec.ts`.
 */
function headerColumns<T extends { name: string }>(
  devices: readonly T[]
): { device: T; label: string }[] {
  const parts = devices.map((device) => {
    const short = device.name.replace(/^(GeForce|Instinct|Radeon)\s+/, '');
    const qualifier = QUALIFIER.exec(short);
    return {
      device,
      stem: qualifier ? short.slice(0, qualifier.index) : short,
      qualifier: qualifier ? qualifier[1] : '',
    };
  });

  const shared = new Set(
    parts
      .filter((part, i) => parts.some((other, j) => j !== i && other.stem === part.stem))
      .map((part) => part.stem)
  );

  return parts.map(({ device, stem, qualifier }) => ({
    device,
    label:
      shared.has(stem) && qualifier ? `${stem} ${qualifier.replace(/\s+(?=[GT]B$)/, '')}` : stem,
  }));
}

/** Colour for a cell: a step of the ramp, or the recessive "did not run" fill. */
function fill(
  cell: MatrixCell,
  measure: MatrixMeasure,
  domain: { min: number; max: number } | undefined
): string {
  const value = measureValue(cell, measure);
  // Absence is not a low score. A pair that cannot run gets the empty fill, so the ramp is only
  // ever read across things that actually ran.
  // A hole, not a dark value: the panel surface, so an unrunnable pair is never mistaken for a
  // poor score at the bottom of a ramp whose darkest step is also nearly black.
  if (value === undefined || domain === undefined) return 'transparent';

  /**
   * The domain and the direction both come from the measure, and neither is decided here — see
   * `MeasureRange`, which argues why the zero floor is right for the two measures that have a
   * reachable zero and wrong for the one that does not (#97).
   */
  return magnitudeFill(value, domain, MEASURE_DIRECTION[measure]);
}

/**
 * The figure at one end of the ramp, said as the quantity that end is made of.
 *
 * A cell rather than a number, and it stays a cell after #97 removed the reason it first had to be
 * one. `measureValue` used to invert TTFT so larger was better, so recovering seconds from the ramp
 * value meant `1 / (1 / t)` — a second derivation of a figure the cell already holds, and not
 * reliably equal to `t`. It returns seconds now, so that particular round trip is gone; reading the
 * field is still what makes the legend's figure and that cell's own tooltip the same string, for
 * `decode` and `ttft`, by construction rather than by two call sites agreeing.
 *
 * **`fit` is the exception, and what it prints is the ramp's value rather than any one cell's
 * sentence.** There the ramp value *is* the quantity — headroom — so a second `1 - utilization`
 * written here would be the duplicate this function exists to avoid. But `measureValue` also
 * collapses *every* offloaded cell to zero by design, which makes the dark end of a fit ramp a
 * population rather than a cell: a pair that fits with nothing to spare and a pair spilling 99% of
 * its weights paint the same square, deliberately, because neither has headroom. So the label reads
 * "0% free", which is the one statement true of all of them. The worst spiller's own figure — "runs
 * only by spilling 99% of its weights to host RAM", which is what that cell's tooltip says — is true
 * of exactly one square wearing that colour, and printing it under `worse` would misdescribe every
 * other one. Worth knowing that the consequence is a legend figure no cell necessarily reports: on
 * the default grid the low cell is Qwen3 32B on an RTX 5080 spilling 66%, and no cell anywhere prints
 * "0% of the ceiling free". Which cell `measureRange` handed back is therefore immaterial here, and
 * that is load-bearing — it is whichever tied cell came first in row-major order.
 *
 * Short by intent: the unit and no more. The measure switch's hint above says which quantity is on
 * the grid, the two words beside these say which direction is better, and a legend that repeats
 * "per user" at both ends is the row that overflows at 320px.
 */
function rampEnd(cell: MatrixCell, measure: MatrixMeasure): string {
  switch (measure) {
    case 'fit':
      return `${percent(measureValue(cell, 'fit') ?? 0)} free`;
    case 'decode':
      return `${rate(cell.tokensPerSec)} tok/s`;
    case 'ttft':
      return seconds(cell.ttftSeconds);
  }
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
  selectedDeviceCount: number,
  /**
   * Name the machine and state the figure, dropping the model and the full device name
   * ([#102](https://github.com/MrZoller/bench/issues/102)).
   *
   * The narrow form, and the reason it exists is arithmetic rather than taste. The readout's
   * reservation is in `rem`, so at a 32px root it doubles — and the glyphs double with it, so the
   * same sentence wraps into roughly twice as many lines each twice as tall. Reserved height grows
   * linearly, required height closer to quadratically. Measured at 320px with the browser default at
   * 32px: the longest sentence needs **280px against 160px reserved**, and 240 against 160 at 390.
   * Above `sm` it fits at both root sizes with nothing to spare and nothing over.
   *
   * The preamble is what makes it long, and **half of it is a part the reader already has and half
   * is not** — which the first draft of this got wrong by dropping both (found in review). The row
   * heading is `sticky left-0`, so the model name is on screen at every scroll position. The column
   * headings are not: they sit at the top of a 35-row grid, so a reader inspecting a cell in a lower
   * row would have been shown `69% of the ceiling free` with nothing anywhere saying which machine.
   *
   * So the narrow form keeps the device and drops the model, which is exactly "drop what is sticky,
   * keep what scrolls" — and keeps it under the *header's* own label rather than the catalog name,
   * since that is the string the column shows and is as short as it can be while still naming its
   * own column.
   *
   * Both forms are rendered and CSS picks one, because the choice is a *layout* question and this
   * component has no viewport to read. The paragraph is `aria-hidden`, so nothing hears both.
   *
   * **An options object with a required field rather than a boolean and a defaulted string**, which
   * is this file's own doctrine about a guard that fails open: `shortDevice = ''` would let a caller
   * ask for the narrow form and get ": 69% of the ceiling free" — a sentence naming no machine at
   * all, which is the defect the narrow form was just corrected for. Present means narrow, and the
   * type will not let it be present and empty-handed.
   */
  narrow?: { shortDevice: string; runtime: string }
): string {
  const model = getModel(cell.modelId).name;
  const device = getDevice(cell.deviceId).name;
  /**
   * Stated even for a blocked cell: "does not run" is a claim about a machine, and on a linked rig it
   * would otherwise read as a verdict on the rig the Bench is holding.
   *
   * **And it is stated in the narrow form too** (found in review). The panel heading carrying "one
   * device per cell" is at the top of a 35-row grid, so a touch reader inspecting a lower row has it
   * off screen — and dropping the qualifier there would make the line appear to describe the
   * multi-device rig they configured, while tapping the cell resets `deviceCount` to 1. That is the
   * misattribution this clause exists to prevent, reintroduced at the width where the heading is
   * least reachable.
   */
  const machine = narrow ? narrow.shortDevice : device;
  /**
   * `1x RTX 5070 Ti` in the narrow form rather than `RTX 5070 Ti, one device`, and the three
   * characters are the whole reason.
   *
   * ", one device" is twelve, which at 320px with a 32px root is a fifth line and 40px past the
   * reservation — measured, and the reason this was not simply the same clause in both forms. The
   * prefix says the same thing in the space available and reads as a count rather than as an aside,
   * which is what it is: the figure is for one of the devices the reader configured.
   */
  const rig =
    selectedDeviceCount > 1 ? (narrow ? `1x ${machine}` : `${machine}, one device`) : machine;
  if (!cell.runs) {
    /**
     * The full stop is added only where the reason has not brought one.
     *
     * `blockedBy` is two unlike things: the Matrix's own short verdicts ("Does not fit") and every
     * categorical refusal `planPlacement` writes, each of which is already a sentence ending in a
     * period. Appending one unconditionally produced "MLX (Apple) does not run on Desktop CPU (2-ch
     * DDR5-6000).." — invisible enough in a native tooltip, and this sentence is now printed under
     * the grid at `text-sm`.
     */
    const reason = cell.blockedBy ?? 'does not run';
    const stop = reason.endsWith('.') ? '' : '.';
    /**
     * The narrow form of a refusal is not the same string with the preamble removed.
     *
     * `blockedBy` is two unlike things — this grid's own short verdicts ("Does not fit"), which are
     * already phrases, and every categorical refusal `planPlacement` writes, which are whole
     * sentences naming the runtime *and* the machine. The second is the longest thing this line can
     * render, and both of the names in it are on the axes or in the Runtime picker at this width.
     * `evaluated` is exactly the split: false for a refusal that never consulted the arithmetic.
     */
    if (narrow) {
      return cell.evaluated
        ? `${rig}: ${reason}${stop}`
        : /*
           * The runtime is named, not elided (found in review). `blockedBy` here is a whole sentence
           * naming the runtime *and* the machine, and dropping it wholesale left "the runtime does not
           * drive this" — which is the one fact a reader cannot recover from the axes, since the
           * Runtime picker is above the grid and off screen for a lower row. The machine comes from the
           * column, so it is the runtime that has to stay.
           */
          `${rig}: ${narrow.runtime} does not drive this.`;
    }
    return `${model} on ${rig}: ${reason}${stop}`;
  }

  const detail =
    measure === 'fit'
      ? cell.offloadFraction > 0
        ? // The other long one, and the narrow form says the same fact in a third of the characters.
          // "of its weights" and "runs only by" are the sentence around the figure, and the figure is
          // what a reader who cannot hover tapped the cell for.
          narrow
          ? `spills ${percent(cell.offloadFraction)} to host RAM`
          : `runs only by spilling ${percent(cell.offloadFraction)} of its weights to host RAM`
        : `${percent(Math.max(0, 1 - cell.utilization))} of the ceiling free`
      : measure === 'decode'
        ? `${rate(cell.tokensPerSec)} tok/s per user`
        : `${seconds(cell.ttftSeconds)} to first token`;

  const at = cell.quantId === selectedQuantId ? '' : ` at ${getQuant(cell.quantId).label}`;
  /**
   * The spilled cell's "runs" carries the qualifier `HOST_RAM_UNCHECKED` exists to enforce
   * (#127): `planPlacement` sizes the spill with no host-RAM input, so "runs only by spilling
   * 99% of its weights to host RAM" was promising a load the engine never checked — on the one
   * surface read as a shortlist. The constant rides verbatim as its own sentence, after the
   * period the composition already writes.
   *
   * The narrow readout stays bare, and the decision is measured rather than aesthetic (argued
   * in review, both ways): its sentence — "spills 66% to host RAM" — states the spill and makes
   * no load claim, the tooltip and aria-label carry the full constant at every width, and
   * qualifying the visual narrow line costs a reservation nobody should pay. Measured at 200%
   * text on a 320px viewport, where the rem floor doubles while the line width halves: the full
   * constant needs a 10rem floor and the inline-clause register still needs 7.5rem — 120px of
   * permanently reserved blank space on a phone, against 80px today. The legend's keyed line
   * and the caption remain the panel-level channels.
   */
  const unchecked =
    measure === 'fit' && cell.offloadFraction > 0 && !narrow ? ` ${HOST_RAM_UNCHECKED}` : '';
  // The stand-in stays in the brief form: it is the one part of the preamble the axes do not carry,
  // and a figure derived from a format the runtime cannot load has to say so at every width.
  return narrow ? `${rig}: ${detail}${at}.` : `${model} on ${rig}${at}: ${detail}.${unchecked}`;
}
