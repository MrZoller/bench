import { useEffect, useRef, useState } from 'react';

/**
 * A button that copies a string, and reports honestly about whether it did.
 *
 * Extracted from the masthead's share link when the launch panel (#136) needed a second one. It is
 * not extracted for tidiness: the four guards below are each a bug that was found and fixed once,
 * and a hand-written second copy is precisely how a codebase comes to have one fixed and one not —
 * the rule this repo states about `kvLabel`, `substitutionFor` and the host-RAM caveat, applied to
 * a component.
 *
 * The four, in the order they were found:
 *
 *   1. **The confirmation is derived, never stored.** "Copied" is a claim that the clipboard holds
 *      what is on screen, so it compares the two rather than setting a flag. A slider moved while
 *      a write is in flight re-derives `value`, and the button beside the new scenario would
 *      otherwise read "copied" while the clipboard holds the old one. Comparison also gets the odd
 *      case right for free: drag away and back, and the claim becomes true again.
 *   2. **Overlapping attempts are disowned by number.** `writeText` is not abortable, so a
 *      superseded write cannot be stopped — only ignored. Without the counter, two clicks could
 *      settle out of order and a stale success would unmount the fallback field a later refusal had
 *      just revealed.
 *   3. **The reset timer is cleared per attempt**, or a second click inherits the first's timer and
 *      a real failure is erased two seconds after being reported.
 *   4. **No clipboard API is a visible state, not a no-op.** `navigator.clipboard` is undefined on
 *      insecure origins and in some embedded browsers, and an optional chain there left the button
 *      looking like it had worked — the worst of the three outcomes. The fallback is the text
 *      itself, selected and ready.
 *
 * No `document.execCommand('copy')` fallback: deprecated, needs a document selection anyway, and
 * fails silently in exactly the same contexts.
 */
export function CopyButton({
  value,
  idleLabel,
  copiedLabel,
  fallbackLabel,
  fieldLabel,
  multiline = false,
  className = 'rounded-md border border-[var(--color-border)] px-3 py-1.5 text-xs text-[var(--color-accent)] hover:border-[var(--color-accent-dim)]',
}: {
  /** The text to copy. Re-derived by the caller on every render — see guard 1. */
  value: string;
  idleLabel: string;
  copiedLabel: string;
  /** What the button says once the clipboard has refused it. */
  fallbackLabel: string;
  /** Accessible name for the manual-copy field the fallback reveals. */
  fieldLabel: string;
  /** A `textarea` rather than an `input`, for anything with newlines in it. */
  multiline?: boolean;
  className?: string;
}) {
  const [state, setState] = useState<'idle' | 'copied' | 'unavailable'>('idle');
  const [copiedValue, setCopiedValue] = useState<string | null>(null);
  const resetTimer = useRef<number | undefined>(undefined);
  const attempt = useRef(0);

  /**
   * Selected once, when the field first appears.
   *
   * A callback ref is recreated on every render, so React re-invoked it on every configuration
   * change and `select()` pulled focus off whatever control the user was operating — a keyboard
   * user could press an arrow key once and lose the control, which is the fallback for one
   * accessibility problem creating a worse one.
   */
  const fieldRef = useRef<HTMLInputElement | HTMLTextAreaElement>(null);
  useEffect(() => {
    if (state === 'unavailable') fieldRef.current?.select();
  }, [state]);

  useEffect(() => () => window.clearTimeout(resetTimer.current), []);

  const confirmed = state === 'copied' && copiedValue === value;
  const label = confirmed ? copiedLabel : state === 'unavailable' ? fallbackLabel : idleLabel;

  const fieldClass =
    'min-w-0 flex-1 rounded-md border border-[var(--color-border)] bg-transparent px-2 py-1.5 text-xs text-[var(--color-text-muted)]';

  return (
    <div className="flex flex-wrap items-center gap-2">
      <button
        type="button"
        onClick={() => {
          window.clearTimeout(resetTimer.current);
          const id = ++attempt.current;
          const superseded = () => attempt.current !== id;
          // The value as it stood at the click, not as it stands when the promise settles. What
          // the clipboard ends up holding is this, and the confirmation compares it with whatever
          // is on screen by then.
          const writing = value;

          const writer = navigator.clipboard?.writeText(writing);
          if (writer === undefined) {
            setState('unavailable');
            return;
          }

          void writer.then(
            () => {
              if (superseded()) return;
              setCopiedValue(writing);
              setState('copied');
              resetTimer.current = window.setTimeout(() => setState('idle'), 2000);
            },
            // A rejected write — permission denied, document not focused — lands here, and means
            // the same thing to the user as no API at all.
            () => {
              if (superseded()) return;
              setState('unavailable');
            }
          );
        }}
        className={className}
      >
        {/* aria-live so the confirmation is announced, not just seen. */}
        <span aria-live="polite">{label}</span>
      </button>

      {state === 'unavailable' &&
        (multiline ? (
          <textarea
            readOnly
            aria-label={fieldLabel}
            value={value}
            rows={value.split('\n').length}
            onFocus={(e) => e.currentTarget.select()}
            ref={fieldRef as React.RefObject<HTMLTextAreaElement>}
            className={`${fieldClass} font-mono whitespace-pre`}
          />
        ) : (
          <input
            readOnly
            aria-label={fieldLabel}
            value={value}
            // Select on focus so one keystroke copies it — the closest thing to the button
            // working that a browser without clipboard access allows.
            onFocus={(e) => e.currentTarget.select()}
            ref={fieldRef as React.RefObject<HTMLInputElement>}
            className={fieldClass}
          />
        ))}
    </div>
  );
}
