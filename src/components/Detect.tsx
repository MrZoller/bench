import { useId, useState } from 'react';
import { DEVICES, useConfig } from '@/store/config';
import { detect, readSignals, type Detection } from '@/lib/detect';
import { deviceOptionLabel } from '@/lib/stops';

/**
 * "What can my machine run?" (#137).
 *
 * The front half of guided mode. The Hardware picker is 43 rows and assumes the visitor knows which
 * one they own; this reads what the browser exposes and narrows it — **to a shortlist and a
 * confirmation, never to a selection.**
 *
 * **The three states are the design, and the third is the one that matters.**
 *
 *   1. *A shortlist.* A few rows, each a button that sets the hardware.
 *   2. *Too many rows to be a shortlist.* On a redacting browser, vendor alone leaves seventeen
 *      shipping NVIDIA rows or ten Apple ones, so the panel says what it learned and asks the reader
 *      to use the picker — with the list already narrowed. A follow-up question is a first-class
 *      path here rather than a failure branch.
 *   3. *Nothing at all.* `navigator.gpu` is undefined in Safari behind a flag and in any hardened
 *      browser. That says so plainly and points at the picker; it is not an error, and nothing is
 *      logged.
 *
 * The evidence is always shown, in all three states, because it is what makes a guess visibly a
 * guess. "Your browser reports an Apple GPU — it reports the Metal feature family rather than the
 * chip" is the difference between a shortlist a reader can judge and one they have to trust.
 */
export function Detect() {
  const set = useConfig((s) => s.set);
  /**
   * Subscribed to, not just written: after a candidate is pressed the list gave no sign which one
   * had been chosen — the Hardware select that reflects it is off screen in a long list, so both a
   * sighted and a screen-reader user could sit on an unchanged button wondering. Raised by Codex on
   * #168.
   */
  const selectedDeviceId = useConfig((s) => s.deviceId);
  const headingId = useId();
  const [state, setState] = useState<'idle' | 'reading' | 'unavailable' | 'done'>('idle');
  const [result, setResult] = useState<Detection | null>(null);

  const run = async () => {
    setState('reading');
    const signals = await readSignals();
    if (signals === undefined) {
      setState('unavailable');
      return;
    }
    setResult(detect(signals, DEVICES));
    setState('done');
  };

  return (
    <div className="sm:col-span-2">
      <button
        type="button"
        onClick={() => void run()}
        disabled={state === 'reading'}
        /* 44px on a coarse pointer, like the disclosures: this is the affordance a reader who does
           not know their hardware depends on, and a target that is hard to hit on the accessibility
           path fails the people the path exists for. */
        className="inline-flex min-h-11 items-center rounded-md border border-[var(--color-border)] px-3 py-1.5 text-xs text-[var(--color-accent)] hover:border-[var(--color-accent-dim)] disabled:opacity-60"
      >
        {state === 'reading' ? 'Reading…' : 'What can my machine run?'}
      </button>

      {state === 'unavailable' && (
        /* Not an error. A browser that exposes no WebGPU adapter is a browser doing what it was
           configured to do, and the honest response is to name the picker rather than to report a
           failure the reader cannot act on. */
        <p aria-live="polite" className="mt-2 text-xs text-[var(--color-text-muted)]">
          This browser exposes no graphics adapter to the page — Safari without the flag, or a
          hardened browser. Pick your hardware above instead; nothing else on this page depends on
          it.
        </p>
      )}

      {state === 'done' && result !== null && (
        <section
          aria-labelledby={headingId}
          /* Announced, because the read is asynchronous and inserts only visual content — without
             this a reader who pressed the button got no indication that anything had happened
             (raised by Codex on #168).

             `aria-live` rather than `role="status"`, and the difference is not cosmetic: `role`
             *replaces* the implicit one, so `role="status"` took this element out of the region
             landmark it is named by — the panel stopped being addressable as "Which of these is
             yours?" at the same moment it became announceable. `aria-live` adds the behaviour and
             leaves the role alone. */
          aria-live="polite"
          className="mt-3 flex flex-col gap-2 rounded-md border border-[var(--color-border)] p-3"
        >
          {/* One heading, because there is now one shape of answer: candidates to choose from.
              It read "What the browser would say" on the long-list path, which was a heading for a
              panel that offered nothing to do. */}
          <h3 id={headingId} className="text-xs font-medium text-[var(--color-text)]">
            Which of these is yours?
          </h3>

          {/* Always, in every state. This is what makes a guess visibly a guess — and on a Mac it
              is the only thing that explains why the list is ten rows long. */}
          <ul className="flex list-none flex-col gap-1 text-xs leading-relaxed text-[var(--color-text-muted)]">
            {result.evidence.map((line) => (
              <li key={line} className="flex gap-2">
                <span aria-hidden="true">·</span>
                <span>{line}</span>
              </li>
            ))}
            {result.evidence.length === 0 && (
              <li>Your browser withheld everything identifying. That is a setting, not a fault.</li>
            )}
            {/* A signal was dropped because applying it would have left no machine at all — an
                Intel Mac is the reachable case. Said out loud, because the survivors are then
                narrowed by *some* of what the browser reported rather than all of it, and a reader
                comparing this list against their own machine deserves to know which. */}
            {result.conflicted === true && (
              <li className="flex gap-2 text-[var(--color-warning)]">
                <span aria-hidden="true">◐</span>
                <span>
                  Two of the readings cannot both be true of any machine in the catalog, so one was
                  ignored. The list below is narrowed by the rest.
                </span>
              </li>
            )}
          </ul>

          {/* A platform the catalog has no row for, and a state where narrowing found nothing.
              Both are terminal: offering "which of these is yours?" over forty-two rows is the
              picker with extra steps, and asking an iPhone which Mac it is has no right answer at
              all. Raised by Codex on #168. */}
          {result.unsupportedPlatform !== undefined || result.narrowedNothing === true ? (
            <p className="text-xs leading-relaxed text-[var(--color-text-muted)]">
              {result.unsupportedPlatform !== undefined
                ? 'There is nothing here to pick — every machine in the catalog is a desktop, a laptop or a server. You can still browse them with the Hardware list above.'
                : 'Nothing your browser said narrows the list, so this is the whole catalog. Use the Hardware list above; it is the same rows.'}
            </p>
          ) : (
            <>
              {/**
               * **The candidates are offered whether or not the list is short**, and the first version
               * hid them past six — discarding the narrowing exactly where it was worth most (raised by
               * Codex on #168). A vendor-only NVIDIA read leaves seventeen rows, and seventeen buttons
               * is still far better than searching forty-three: the reader is told it is a long list and
               * given it anyway, rather than sent back to the unfiltered picker.
               */}
              {result.askAbout !== undefined && (
                <p className="text-xs leading-relaxed text-[var(--color-text-muted)]">
                  That still leaves {result.candidates.length} machines, which is more than a
                  shortlist.{' '}
                  {result.askAbout === 'memory'
                    ? 'How much memory does it have? The memory is in every name below.'
                    : 'Everything below the vendor is a guess this browser will not let anyone make — but these are the rows it could be.'}
                </p>
              )}
              <ul className="flex list-none flex-wrap gap-2">
                {result.candidates.map((device) => (
                  <li key={device.id}>
                    <button
                      type="button"
                      /* A confirmation, and the only place detection ever writes to the store. The
                       reader picks; nothing here is applied on their behalf. */
                      onClick={() => set('deviceId', device.id)}
                      /* Which one was chosen, in the accessibility tree and not only in a border. The
                     Hardware select that otherwise reflects it is off screen in a seventeen-row
                     list, so pressing a button left both a sighted and a screen-reader user on an
                     unchanged control with no sign anything had happened. Raised by Codex on #168. */
                      aria-pressed={device.id === selectedDeviceId}
                      className={`inline-flex min-h-11 items-center rounded-md border px-3 py-1.5 text-xs text-[var(--color-text)] hover:border-[var(--color-accent-dim)] ${
                        device.id === selectedDeviceId
                          ? 'border-[var(--color-accent)]'
                          : 'border-[var(--color-border)]'
                      }`}
                    >
                      {deviceOptionLabel(device)}
                    </button>
                  </li>
                ))}
              </ul>
            </>
          )}
        </section>
      )}
    </div>
  );
}
