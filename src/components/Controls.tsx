import { Fragment, useId, useState, type ReactNode } from 'react';
import { DisclosureToggle } from './DisclosureToggle';

/**
 * The input layer. Plain `select` and `range` rather than custom widgets, because both are
 * keyboard-operable, screen-reader-legible and touch-friendly for free, and none of that is
 * worth re-implementing for a visual flourish.
 *
 * Every control is labelled and every live value is shown next to its control in the accent
 * colour — the accent's one job is marking what responds to you.
 *
 * Every control can also carry a `note`: one line of `text-xs` prose under it, wired through
 * `aria-describedby` so it is part of the control's accessible description rather than text that
 * merely happens to sit nearby. All three take one now. `Select` had the mechanism and the other two
 * had nothing, which is how the five controls driving every figure on the page came to explain none
 * of themselves (#80) — a call site cannot say what a component has no way to render.
 *
 * **A note is one line, and `Select` now has somewhere else to put the rest.** An option can also
 * carry a `detail` — reference prose about the thing you picked, rather than a claim that helps you
 * pick — which renders behind a disclosure and stays out of the accessible description. Without that
 * second slot the Hardware picker had one string for both jobs, so 180 words of catalog provenance
 * were read out on every focus and drawn as five lines of 12px text inside a two-column grid (#68).
 *
 * **And an option can carry a `group`, which is the one structural thing a native `select` gives you
 * for free and this had no way to ask for** (#79). The Hardware picker was a flat list of 43 options
 * whose order was grouped, deliberate and completely unmarked — discrete cards, whole machines and
 * CPU hosts running together with nothing to say where one kind ended. `<optgroup>` is the browser's
 * own answer: a heading a reader sees while scanning, announced by a screen reader on entering the
 * run, and free on touch, where a custom listbox would have cost all three.
 */

/**
 * `*emphasis*`. The innermost mark: nothing renders inside it.
 *
 * One capture group, like the two below, because `String.split` with a single group alternates
 * unmatched text and matched spans — so the odd indices are the marked ones, and the parts in
 * between cannot contain a pair of their own.
 */
const EMPHASIS = /(\*[^*]+\*)/;

/**
 * `**strong**`, whose content is prose and may carry an emphasis of its own: an interior `*` is
 * part of the content precisely when it is not the closing pair, which is what `\*(?!\*)` says.
 */
const STRONG = /(\*\*(?:[^*]|\*(?!\*))+\*\*)/;

/** A backticked identifier. Outermost, because its content is literal — marks and all. */
const CODE = /(`[^`]+`)/;

/** `*emphasis*` within one already-classified span. The last pass: nothing renders inside it. */
function emphasised(text: string, key: string): ReactNode[] {
  return text
    .split(EMPHASIS)
    .map((part, i) => (i % 2 === 0 ? part : <em key={`${key}.${i}`}>{part.slice(1, -1)}</em>));
}

/** `**strong**`, whose content is prose and goes back through the emphasis pass. */
function strengthened(text: string, key: string): ReactNode[] {
  return text.split(STRONG).flatMap((part, i): ReactNode[] =>
    i % 2 === 0
      ? emphasised(part, `${key}.${i}`)
      : [
          <strong key={`${key}.${i}`} className="font-semibold text-[var(--color-text)]">
            {emphasised(part.slice(2, -2), `${key}.${i}b`)}
          </strong>,
        ]
  );
}

/**
 * The inline marks the curated catalog notes are written with, rendered rather than printed.
 *
 * `devices.json` is prose for a reader — `docs/ROADMAP.md`'s register — and it uses `**strong**`,
 * `*emphasis*` and backticked identifiers throughout: "**This row is the 80-core GPU**", "the FP16
 * *matrix* rate", "`iogpu.wired_limit_mb` accepts a value up to physical memory". Nothing rendered
 * any of them, so rows printed literal asterisks in the picker, and moving the prose to its own
 * region would have moved the glitch with it. Same root cause as the punctuation this issue is
 * about: reference prose emitted as UI copy without the transformation it needs.
 *
 * **One pass per mark, outermost first**, rather than one regex with three alternatives. A single
 * pass has to decide `**a *b* c**` on a lookahead, and the version of this that went to review
 * documented that input as falling through to literal text while actually emitting the whole
 * fragment as bold — the two-marks-and-a-docstring shape could not state its own rule. Code is
 * outermost and keeps its content literal, since an identifier is a thing you type; then `**`,
 * whose content goes through the emphasis pass; then `*`. A marker with no partner matches nothing
 * and reaches the output as the character the curator typed.
 *
 * Still three patterns and no parser: a curator who wants a list or a link is asking for something
 * this field is not for. `src/App.test.tsx` renders every note in the catalog through this and
 * asserts nothing of the markup survives as text, so a fourth mark fails a test rather than
 * printing itself at a reader.
 */
function inlineProse(text: string): ReactNode[] {
  return text.split(CODE).flatMap((part, i): ReactNode[] =>
    i % 2 === 0
      ? strengthened(part, `p${i}`)
      : [
          // `tabular` is the app's only mono utility — named for the figures it was written for, but
          // it is `--font-mono` and this is `iogpu.wired_limit_mb`, a thing you type. A second class
          // that set the same family would be the drift this repo keeps removing.
          <code key={`p${i}`} className="tabular">
            {part.slice(1, -1)}
          </code>,
        ]
  );
}

/** One option of a `Select`, in the order the control renders them. */
interface Option<T extends string> {
  value: T;
  label: string;
  disabled?: boolean;
  note?: string;
  /**
   * Reference prose about the selected option, shown behind a disclosure and *not* part of the
   * control's accessible description. See `devicePickerNote`.
   */
  detail?: string;
  /**
   * The heading this option sits under, or `undefined` for an option that belongs to no group.
   *
   * A `<select>` groups by *adjacency* — an `<optgroup>` is a run of contiguous options — so this is
   * a label on each option rather than a list of groups with options inside them. That is not a
   * shortcut: it means the component cannot reorder the list to make its groups, so the sequence a
   * call site passes is exactly the sequence a reader gets. The Hardware picker's order is
   * `devices.json`'s row order, which `$comment-order` states and `catalog.test.ts` enforces (#79);
   * a component that grouped by filtering three times would silently own that order instead.
   */
  group?: string;
}

/**
 * The options split into the runs an `<optgroup>` can be made of.
 *
 * A run per *change* of `group`, walking the list once — so a list whose groups are interleaved
 * renders as two `<optgroup>`s carrying one label rather than being tidied into one. That is the
 * honest rendering of it: the sequence the call site passed survives, and the fact that its groups are
 * not runs is visible instead of being repaired behind its back.
 */
function optionRuns<T extends string>(
  options: readonly Option<T>[]
): { group?: string; options: Option<T>[] }[] {
  const runs: { group?: string; options: Option<T>[] }[] = [];
  for (const option of options) {
    const last = runs.at(-1);
    if (last && last.group === option.group) last.options.push(option);
    else runs.push({ group: option.group, options: [option] });
  }
  return runs;
}

/** One `<option>`, written once so a grouped run and an ungrouped one cannot render differently. */
function renderOption<T extends string>(option: Option<T>) {
  return (
    <option key={option.value} value={option.value} disabled={option.disabled}>
      {option.label}
    </option>
  );
}

export function Select<T extends string>({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: T;
  onChange: (value: T) => void;
  options: readonly Option<T>[];
}) {
  const id = useId();
  /**
   * The selected option's note, and only that.
   *
   * There was a `hint` prop behind this as a fallback, which no call site ever passed — a dead
   * escape hatch, and one that would have misbehaved if used: it rendered only while the selected
   * option had no note of its own, so a control-level explanation would appear and vanish as the
   * choice moved. A fixed sentence per control is `note` on `StopSlider`/`Segmented` below, where
   * there are no per-option notes to fight with. Deleted rather than kept, because an unused prop
   * that duplicates a working one is how the two drift.
   */
  const selected = options.find((o) => o.value === value);
  const note = selected?.note;
  const detail = selected?.detail;

  /**
   * Open across a change of selection, rather than reset by it.
   *
   * Someone who opened the note on a Mac Studio is comparing it with the next machine down, and
   * closing it under them on every change would make the disclosure useless for exactly the reader
   * who opened it. The region simply disappears while the selection has no detail — which is honest
   * about there being nothing to show, and restores itself on the next option that has one.
   */
  const [showDetail, setShowDetail] = useState(false);

  return (
    <div className="flex flex-col gap-1">
      <label
        htmlFor={id}
        className="text-xs font-medium tracking-wide text-[var(--color-text-faint)] uppercase"
      >
        {label}
      </label>
      <select
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
        // The note carries load-bearing warnings ("Does not run on ..."), so it has to be part
        // of the control's accessible description rather than adjacent text.
        aria-describedby={note ? `${id}-note` : undefined}
        /**
         * The focus indicator is an outline. Both halves of that sentence were wrong before.
         *
         * It was `focus:border-[accent] focus:outline-none` — a 1px edge measuring **1.95:1 against
         * the unfocused edge**, where WCAG 2.2 SC 2.4.13 asks for 3:1 at a 2px minimum thickness,
         * and colour-only, so a deuteranope or a protanope loses most of what separates violet from
         * slate-blue at that size (#67). Every other control in the app already keeps a real 2px
         * indicator; these four were the outlier, and they are the two inputs — model and hardware —
         * that everything else on the page derives from.
         *
         * **`outline` rather than the `ring` its neighbours use**, because a ring is a `box-shadow`
         * and this is a native `menulist` select, which WebKit paints through the platform rather
         * than from the CSS box. A box-shadow is not reliably painted on one, so the fix that looks
         * like the rest of the app would have shipped no indicator at all in Safari. An outline is
         * drawn by the browser outside the control's box in every engine, and it is the mechanism
         * the success criterion is written around.
         *
         * The border swap stays, as a redundant second cue. It is simply no longer the indicator.
         */
        className="w-full rounded-md border border-[var(--color-control-border)] bg-[var(--color-surface-raised)] px-3 py-2 text-sm text-[var(--color-text)] focus:border-[var(--color-accent)] focus:outline-2 focus:outline-offset-2 focus:outline-[var(--color-accent)]"
      >
        {/* Grouped where a call site says so, flat where it does not — the two compose, so a
            picker with headings for some of its rows and none for the rest still renders in one
            pass and in one order. `Fragment` keyed on the run's first option, since a run has no
            id of its own and its heading may be absent. */}
        {optionRuns(options).map((run) =>
          run.group === undefined ? (
            <Fragment key={`ungrouped:${run.options[0].value}`}>
              {run.options.map(renderOption)}
            </Fragment>
          ) : (
            <optgroup key={`${run.group}:${run.options[0].value}`} label={run.group}>
              {run.options.map(renderOption)}
            </optgroup>
          )
        )}
      </select>
      {note && (
        <p id={`${id}-note`} className="text-xs text-[var(--color-text-muted)]">
          {note}
        </p>
      )}
      {/* Named for the control rather than "Show more", because the accessible name of a bare
          "Show more" is the same on every picker that grows one, and a screen-reader user listing
          the page's buttons would get a set of identical labels pointing at different regions. */}
      {detail && (
        <>
          <DisclosureToggle
            expanded={showDetail}
            onToggle={() => setShowDetail((v) => !v)}
            controls={`${id}-detail`}
          >
            {showDetail ? 'Hide' : 'Show'} the full {label.toLowerCase()} note
          </DisclosureToggle>
          {showDetail && (
            <p
              id={`${id}-detail`}
              className="text-xs leading-relaxed text-[var(--color-text-muted)]"
            >
              {inlineProse(detail)}
            </p>
          )}
        </>
      )}
    </div>
  );
}

/**
 * A slider over an explicit list of stops rather than a numeric range.
 *
 * Context and concurrency are read logarithmically — the interesting jumps are 4K to 32K to
 * 128K, not 4K to 4.1K — and a linear range would spend most of its travel in a region nobody
 * cares about. The index is the input; the value is looked up.
 */
export function StopSlider<T extends number | string>({
  label,
  stops,
  value,
  onChange,
  format,
  note,
}: {
  label: string;
  stops: readonly T[];
  value: T;
  onChange: (value: T) => void;
  format: (value: T) => string;
  /** One sentence on what this setting is. See `SETTING_NOTES`. */
  note?: string;
}) {
  const id = useId();
  const index = Math.max(0, stops.indexOf(value));

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-baseline justify-between gap-2">
        <label
          htmlFor={id}
          className="text-xs font-medium tracking-wide text-[var(--color-text-faint)] uppercase"
        >
          {label}
        </label>
        {/* Live value in the accent: this is the thing that moves when you drag. */}
        <output htmlFor={id} className="tabular text-sm text-[var(--color-accent)]">
          {format(value)}
        </output>
      </div>
      <input
        id={id}
        type="range"
        min={0}
        max={stops.length - 1}
        step={1}
        value={index}
        onChange={(e) => onChange(stops[Number(e.target.value)])}
        aria-valuetext={format(value)}
        // On the input rather than beside it, for the same reason `Select`'s note is: adjacent text
        // is text a screen-reader user reaches after the control, if they reach it at all, and a
        // slider is exactly where they will not — arrowing through the stops re-announces the value
        // and nothing else. The description is read once when the slider is focused.
        aria-describedby={note ? `${id}-note` : undefined}
        className="h-6 w-full cursor-pointer accent-[var(--color-accent)]"
      />
      {note && (
        <p id={`${id}-note`} className="text-xs text-[var(--color-text-muted)]">
          {note}
        </p>
      )}
    </div>
  );
}

/**
 * Segmented control for short, mutually exclusive choices where seeing all options helps.
 *
 * Native radio inputs under the styling, not buttons with `aria-pressed`. Toggle buttons say
 * "this one is on" independently; they never say "and choosing it turned the others off", so a
 * screen-reader user could not tell these were alternatives, and arrow-key navigation — which
 * people expect inside a group of radios — did not work at all.
 */
export function Segmented<T extends string>({
  label,
  value,
  onChange,
  options,
  note,
}: {
  label: string;
  value: T;
  onChange: (value: T) => void;
  options: readonly { value: T; label: string }[];
  /** One sentence on what this setting is. See `SETTING_NOTES`. */
  note?: string;
}) {
  const name = useId();

  return (
    /* The description belongs to the *group*, not to each radio. A screen reader announces a
       group's name and description on entry, so the sentence is heard once before the options; put
       on the radios it would be re-read on every arrow key — three times for three options — which
       is how a description earns itself a reputation for being noise. `fieldset` is the group. */
    <fieldset aria-describedby={note ? `${name}-note` : undefined} className="flex flex-col gap-1">
      <legend className="text-xs font-medium tracking-wide text-[var(--color-text-faint)] uppercase">
        {label}
      </legend>
      {/* `flex-wrap`, because a non-wrapping row's min-content is the *sum* of its options and
          that is a floor the viewport cannot argue with. It is not the control that overflowed:
          the row sets the width of its grid column, and every `w-full` slider sharing that column
          inherits it — so at a 32px root the whole Usage panel scrolled the page sideways on the
          strength of four KV options. Wrapping makes the floor the widest single option instead.
          The labels keep `flex-1`, whose zero basis is safe here only because each has a
          `min-width: auto` floor of its own text; that is what stops a wrapped line collapsing
          the way the Matrix legend's ramp did (#35). */}
      <div className="flex flex-wrap gap-1 rounded-md border border-[var(--color-control-border)] bg-[var(--color-surface-raised)] p-1">
        {options.map((option) => {
          const active = option.value === value;
          return (
            <label
              key={option.value}
              className={`flex-1 cursor-pointer rounded px-2 py-1 text-center text-sm transition-colors focus-within:ring-2 focus-within:ring-[var(--color-accent)] ${
                active
                  ? 'bg-[var(--color-accent-dim)] text-[var(--color-text)]'
                  : 'text-[var(--color-text-muted)] hover:text-[var(--color-text)]'
              }`}
            >
              <input
                type="radio"
                name={name}
                value={option.value}
                checked={active}
                onChange={() => onChange(option.value)}
                // Visually hidden rather than `hidden`: it has to stay focusable and reachable
                // by arrow keys, which is the whole reason for using a radio here.
                className="sr-only"
              />
              {option.label}
            </label>
          );
        })}
      </div>
      {note && (
        <p id={`${name}-note`} className="text-xs text-[var(--color-text-muted)]">
          {note}
        </p>
      )}
    </fieldset>
  );
}
