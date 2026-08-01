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
        <p className="mt-2 text-xs text-[var(--color-text-muted)]">
          This browser exposes no graphics adapter to the page — Safari without the flag, or a
          hardened browser. Pick your hardware above instead; nothing else on this page depends on
          it.
        </p>
      )}

      {state === 'done' && result !== null && (
        <section
          aria-labelledby={headingId}
          className="mt-3 flex flex-col gap-2 rounded-md border border-[var(--color-border)] p-3"
        >
          <h3 id={headingId} className="text-xs font-medium text-[var(--color-text)]">
            {result.askAbout === undefined
              ? 'Which of these is yours?'
              : 'What the browser would say'}
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

          {result.askAbout === undefined ? (
            <ul className="flex list-none flex-wrap gap-2">
              {result.candidates.map((device) => (
                <li key={device.id}>
                  <button
                    type="button"
                    /* A confirmation, and the only place detection ever writes to the store. The
                       reader picks; nothing here is applied on their behalf. */
                    onClick={() => set('deviceId', device.id)}
                    className="inline-flex min-h-11 items-center rounded-md border border-[var(--color-border)] px-3 py-1.5 text-xs text-[var(--color-text)] hover:border-[var(--color-accent-dim)]"
                  >
                    {deviceOptionLabel(device)}
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-xs leading-relaxed text-[var(--color-text-muted)]">
              That still leaves {result.candidates.length} machines, which is not a shortlist.{' '}
              {result.askAbout === 'memory'
                ? 'How much memory does it have? Pick the matching row above — the memory is in every name.'
                : 'Pick yours from the Hardware list above; everything below the vendor is a guess this browser will not let anyone make.'}
            </p>
          )}
        </section>
      )}
    </div>
  );
}
