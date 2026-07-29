import { useId, useMemo, useState } from 'react';
import type { Evaluation } from '@/engine';
import { judgeWorkloads, type Fitness } from '@/engine/verdict';
import { estimateConfig, type Config } from '@/store/config';
import { PanelCount } from './PanelCount';
import { DisclosureToggle } from './DisclosureToggle';

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
  /**
   * Not a fourth grade — the absence of one, and styled to say so (#75).
   *
   * The three above are a status scale with reserved colour tokens: full, half, empty, and `fail`
   * shares `--color-critical` with Telemetry's "Will not run". This state is not on that scale, so it
   * takes an ink token rather than a status hue. A neutral row wearing the strongest negative in the
   * vocabulary made that colour mean two different things — "this machine cannot do it" and "you have
   * not configured this yet" — and the second one is a prompt to move a slider, not a verdict.
   *
   * `--color-text-faint` is the recessive ink, 5.10:1 on the panel surface, which clears the 4.5:1 a
   * 12px status word needs; the status tokens are reserved for states that are actually graded.
   *
   * The glyph leaves the circle family on purpose. `○` is the *bottom* of that ordered set, and a row
   * that was never measured does not sit at the bottom of an ordering it is not in — a dash reads as
   * "no reading", the same convention Telemetry's `value: '—'` already uses for a figure it cannot
   * report. And it is `aria-hidden` below, like the other three: the word beside it is the grading.
   *
   * An em dash, U+2014, which is the character Telemetry actually uses — the first version of this
   * said so in the comment and shipped an en dash, U+2013, which is a narrower glyph than any of the
   * three circles. That is not only a typographic quibble: the word beside it starts after the glyph,
   * so at 1440px "Not measured" began at x = 202.0 while "Yes", "Tight" and "No" all began at 206.1,
   * a 4.1px jog in the column #70 exists to have aligned. The glyph is now inside a fixed box at
   * `sm`, which is where the shared column exists, so the alignment no longer depends on which
   * character this is — see the icon span below.
   */
  unmeasured: { icon: '—', word: 'Not measured', color: 'var(--color-text-faint)' },
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

  /**
   * The headline counts what was graded — both sides of the fraction.
   *
   * `usable` was `fitness !== 'fail'` over all seven rows, so an *ungraded* row was subtracted
   * exactly as a failing one was: at the default concurrency of 1 this panel read "5 of 7 workloads"
   * on a Spark that would serve several users perfectly well, because nobody had touched the slider
   * (#75). And that is the first number most visitors read here, at the setting they arrive on.
   *
   * A row that was not measured is not evidence in either direction, so it leaves the numerator and
   * the denominator together — "5 of 6", with the seventh row still on screen saying what it is
   * waiting for. Dropping it from the numerator alone would report the same understatement; keeping
   * it in both would count it as a pass, which is the opposite lie.
   */
  const graded = verdicts.filter((v) => v.fitness !== 'unmeasured');
  const usable = graded.filter((v) => v.fitness !== 'fail').length;

  /**
   * When nothing can run, every row carries the same sentence — so it is said once, above the
   * list, and the rows keep only their status. Seven identical explanations read as seven
   * separate problems.
   *
   * `every` over *all* the verdicts rather than over `graded`, now that a row can be ungraded, and
   * the difference is load-bearing: this collapse blanks each row's own reason, so folding an
   * ungraded row into it would delete the one sentence saying what that row is waiting for. Only
   * `judgeWorkloads`' top-level refusal gives seven rows one sentence, and it grades all seven
   * `fail` — so an `unmeasured` row in the list is itself proof the rows are not all saying the same
   * thing, and it blocks the collapse rather than being swallowed by it.
   */
  const sharedReason =
    verdicts.every((v) => v.fitness === 'fail') && new Set(verdicts.map((v) => v.reason)).size === 1
      ? verdicts[0].reason
      : undefined;

  return (
    <section aria-labelledby={headingId} className="panel p-[min(1.25rem,5vw)]">
      <header className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <h2 id={headingId} className="text-sm font-semibold tracking-wide">
          What you could do with it
        </h2>
        {/*
          The noun qualifies the denominator whenever it is short of the list (found in review).

          `graded.length` deliberately omits an ungraded row, so that a row nobody asked about is
          not counted as a failure — but the fraction it produces is read as a headline, and
          "6 of 6 workloads" beside seven visible rows claims complete coverage. That is the same
          false implication the omission exists to avoid, arriving through the total instead of
          through the numerator.

          Naming the denominator is the smallest fix that removes the claim: "6 of 6 measured
          workloads" is true, and the row saying "Not measured" is on screen to account for the
          seventh. The alternative — counting all seven and marking the ungraded one as neither —
          needs a three-part fraction, which is a bigger change to a headline than this earns.
        */}
        <PanelCount count={usable} total={graded.length}>
          {graded.length === verdicts.length ? 'workloads' : 'measured workloads'}
        </PanelCount>
      </header>

      {sharedReason && (
        <p className="mt-3 text-sm text-[var(--color-critical)]">
          <span aria-hidden="true">▲ </span>
          {sharedReason}
        </p>
      )}

      {/*
        The three tracks live on the list, not on the row.

        With the grid on each `<li>` every row was its own grid container, so the middle `auto`
        track was sized from that row's own label — and the written reason, which is the third
        column, started at a different x on all seven rows: 444 to 503 at 1440px, 59px of rag. The
        first track is fixed at `9rem`, so the status word and the label lined up regardless, and
        that alignment is exactly what makes the third one read as a column rather than as prose.
        It is the column that carries the panel's whole argument — seven archetypes, seven answers,
        each explained in writing — and scanning those reasons against each other is what a
        wandering left edge stops the eye doing (#70).

        Subgrid on the row rather than `display: contents`, which is the other way to share one set
        of tracks. Two reasons, and neither is the one this repo was bitten by before (the Bench's
        `contents` scroll anchor, which generated no principal box): a row that generates no box is
        a `<li>` that shipping browsers drop from the accessibility tree, so a seven-item list is
        announced as an empty one; and `order` applies among siblings of one container, so with the
        rows dissolved the mobile stacking below would sort all twenty-one cells into a block of
        labels, a block of status words and a block of reasons instead of seven rows. Subgrid keeps
        the row a real box and keeps its `order` scoped to it.

        Fixing the middle track instead — a guessed 11rem in place of `auto` — aligns the columns
        today and silently overflows on the first archetype label longer than "Inline code
        completion". A subgrid over an `auto` track is that measurement taken rather than guessed.

        The support check the issue asked for: subgrid is Baseline widely available — Firefox 71,
        Safari 16, Chrome and Edge 117, which is September 2023 plus the thirty months Baseline
        waits. It is *not*, however, inside Vite's default build target, which floors at Chrome 111
        (`baseline-widely-available` resolves to chrome111/edge111/firefox114/safari16.4), so Chrome
        and Edge 111 to 116 are browsers this build targets and the feature is missing from. There
        the value is invalid, the declaration is dropped, and the row is left with no column
        template — which is the reason the row's own two-column template is `max-sm:`-scoped rather
        than unscoped. Unscoped it survives into those browsers past `sm` and, with `sm:order-none`
        cancelling the stacking, renders `● Yes` *before* the label — reversing both layouts we
        actually support. Scoped, the row falls back to one implicit column and the three cells
        stack in DOM order, which is the same sequence they read in left to right: ugly, complete,
        and confined to browsers older than the feature. No `@supports` branch to restore the
        three-column table for them, because that branch would have to re-declare the per-row
        tracks this issue is about, unexercised by any test, for browsers that update themselves.
      */}
      <ul className="mt-4 grid gap-y-2 sm:grid-cols-[9rem_auto_1fr] sm:gap-x-3">
        {verdicts.map(({ workload, fitness, reason }) => {
          const style = FITNESS[fitness];
          return (
            <li
              key={workload.id}
              /*
               * Below `sm` the row is its own two-column grid, exactly as it was: the stacked
               * layout is built from `order` and a spanning third cell, both of which are
               * relationships among one row's three children. Only at `sm`, where every child is
               * `order-none` anyway, does the row hand its columns back to the list.
               *
               * `max-sm:` and not bare, so that in a browser without subgrid the row is left with
               * no template rather than with this one — see the note above the list.
               */
              className="grid items-baseline gap-x-3 gap-y-0.5 max-sm:grid-cols-[auto_1fr] sm:col-span-3 sm:grid-cols-subgrid"
            >
              {/*
                Icon and word together, so the grading survives without colour.

                **The `nowrap` is `sm:`-scoped, and that is a reflow fix rather than a tidy-up.** At
                `sm` and above this cell sits in a *fixed* `9rem` track: nothing it contains can widen
                the layout, and a status word must not wrap there or the row's one line becomes two
                and the column stops being a column. Below `sm` the row is its own
                `grid-cols-[auto_1fr]` and this cell is the `1fr`, whose automatic minimum is its
                min-content — so an unbreakable string here is a hard floor on the width of the whole
                row, and the row is inside a panel inside the document.

                Three short words never reached that floor. "Not measured" does: measured in Chromium
                at `reflow.spec.ts`'s own configuration — 320x900, `defaultFontSize=32`, `--font-sans`
                set to Courier New — the seventh row's status cell had a min-content of 199px against
                55px for "○ No", which took `documentElement.scrollWidth` to 371px in a 320px viewport
                and put 29 elements outside the document. That is WCAG 1.4.4 (#35), the criterion this
                repo already shipped a failure of once. Letting the word wrap below `sm` drops the same
                cell to 142px and the document back to 320/320, because the floor becomes the longest
                *word* rather than the whole string.

                At the default text size it never wraps — on a 390px phone the cell holds 96px of
                content in a 188px track, and even at 320px 96px in 131px — so this costs nothing on
                the layout anyone actually arrives at.
              */}
              <span
                className="order-2 flex items-center gap-1.5 text-xs sm:order-none sm:whitespace-nowrap"
                style={{ color: style.color }}
              >
                {/*
                  A fixed box at `sm`, so the word beside it starts at the same x on all seven rows.

                  The circles are ~11.1px at this size and the dash is 10.5px; before this the word's
                  left edge moved with the glyph, which put "Not measured" 4.1px left of the other six
                  in the column #70 exists to have aligned — and every alignment assertion in
                  `e2e/workload-columns.spec.ts` kept passing, because they measure the *cell* box.

                  `w-3` is `0.75rem`, the same length as `text-xs`, so the box is one em at every root
                  size and a glyph wider than the box is centred rather than nudging the word.
                  `shrink-0` so a cell tighter than its track cannot take the width back out of it.

                  Scoped to `sm` because below it there is no shared column to align to — the status
                  word follows a label of the row's own width — and because a fixed box is +10px of
                  min-content at 200% text, which is most of the slack the wrap fix above just bought.
                */}
                <span aria-hidden="true" className="sm:w-3 sm:shrink-0 sm:text-center">
                  {style.icon}
                </span>
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

      <DisclosureToggle expanded={expanded} onToggle={() => setExpanded((v) => !v)}>
        {expanded ? 'Hide' : 'Show'} what each workload means
      </DisclosureToggle>
    </section>
  );
}
