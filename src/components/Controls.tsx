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
        className="w-full rounded-md border border-[var(--color-control-border)] bg-[var(--color-surface-raised)] px-3 py-2 text-sm text-[var(--color-text)] focus:border-[var(--color-accent)] focus:outline-none"
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

/** Segmented control for short, mutually exclusive choices where seeing all options helps. */
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
  return (
    <fieldset className="flex flex-col gap-1">
      <legend className="text-xs font-medium tracking-wide text-[var(--color-text-faint)] uppercase">
        {label}
      </legend>
      <div className="flex gap-1 rounded-md border border-[var(--color-control-border)] bg-[var(--color-surface-raised)] p-1">
        {options.map((option) => {
          const active = option.value === value;
          return (
            <button
              key={option.value}
              type="button"
              aria-pressed={active}
              onClick={() => onChange(option.value)}
              className={`flex-1 rounded px-2 py-1 text-sm transition-colors ${
                active
                  ? 'bg-[var(--color-accent-dim)] text-[var(--color-text)]'
                  : 'text-[var(--color-text-muted)] hover:text-[var(--color-text)]'
              }`}
            >
              {option.label}
            </button>
          );
        })}
      </div>
    </fieldset>
  );
}
