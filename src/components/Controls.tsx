import { useId } from 'react';

/**
 * The input layer. Plain `select` and `range` rather than custom widgets, because both are
 * keyboard-operable, screen-reader-legible and touch-friendly for free, and none of that is
 * worth re-implementing for a visual flourish.
 *
 * Every control is labelled and every live value is shown next to its control in the accent
 * colour — the accent's one job is marking what responds to you.
 */

export function Select<T extends string>({
  label,
  value,
  onChange,
  options,
  hint,
}: {
  label: string;
  value: T;
  onChange: (value: T) => void;
  options: readonly { value: T; label: string; disabled?: boolean; note?: string }[];
  hint?: string;
}) {
  const id = useId();
  const note = options.find((o) => o.value === value)?.note ?? hint;

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
        {options.map((option) => (
          <option key={option.value} value={option.value} disabled={option.disabled}>
            {option.label}
          </option>
        ))}
      </select>
      {note && (
        <p id={`${id}-note`} className="text-xs text-[var(--color-text-muted)]">
          {note}
        </p>
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
}: {
  label: string;
  stops: readonly T[];
  value: T;
  onChange: (value: T) => void;
  format: (value: T) => string;
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
        className="h-6 w-full cursor-pointer accent-[var(--color-accent)]"
      />
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
}: {
  label: string;
  value: T;
  onChange: (value: T) => void;
  options: readonly { value: T; label: string }[];
}) {
  const name = useId();

  return (
    <fieldset className="flex flex-col gap-1">
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
    </fieldset>
  );
}
