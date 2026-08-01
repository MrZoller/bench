import { useId, useMemo } from 'react';
import type { Config } from '@/store/config';
import type { Placement } from '@/engine';
import { getDevice, getModel } from '@/data/catalog';
import { getQuant } from '@/data/quants';
import { getRuntime } from '@/data/runtimes';
import { launchCommands, type Emission, type LauncherCommands } from '@/lib/launch';
import { CopyButton } from './CopyButton';

/**
 * The command that runs the configuration above (#136).
 *
 * The panel every other one on this page was leading to: bench says a configuration works, and
 * then stops one step short of the thing the reader came for. The gap between "fits, 42 tok/s" and
 * a serving process is exactly the part people fumble — the offload count, the context flag, the
 * cache-type flags, vLLM's shard count — and every one of those is an answer the engine has
 * already committed to.
 *
 * **Nothing here derives a figure.** `src/lib/launch.ts` formats what `planPlacement` decided, and
 * this component chooses how much of it to show. The one judgement it makes is which refusals are
 * worth a reader's attention:
 *
 *   - A launcher whose *both* forms refuse for one reason — an unsupported placement, a checkpoint
 *     the catalog cannot name, a format the runtime does not load — gets that reason stated once,
 *     in full. It is the useful half of the feature: "vLLM cannot be pointed at an AWQ pack of this
 *     model, because the catalog has none" is what the reader needs to know.
 *   - A launcher refusing only the *other* form gets a muted pointer. `llama-server` does not
 *     measure and `llama-bench` does not serve, and saying so once beside each is navigation
 *     rather than a problem.
 *
 * Placed after the workload grades and before the two grids, which is where it sits in the
 * reader's own sequence: what fits, how it grades, how to run it — and only then the exploratory
 * fields that ask about other machines.
 */
export function Launch({ config, placement }: { config: Config; placement: Placement }) {
  const headingId = useId();

  const groups = useMemo(() => {
    const model = getModel(config.modelId);
    const quant = getQuant(config.quantId);
    const runtime = getRuntime(config.runtimeId);
    const rig = { device: getDevice(config.deviceId), count: config.deviceCount };

    return launchCommands({
      model,
      quant,
      runtime,
      rig,
      placement,
      usage: {
        contextTokens: config.contextTokens,
        concurrency: config.concurrency,
        promptTokens: config.promptTokens,
        kvPrecision: config.kvPrecision,
      },
    });
  }, [config, placement]);

  // A runtime with no launcher registered renders nothing at all rather than an empty panel
  // explaining that it has nothing to say.
  if (groups.length === 0) return null;

  return (
    <section aria-labelledby={headingId} className="panel p-[min(1.25rem,5vw)]">
      <header className="mb-4">
        <h2 id={headingId} className="text-sm font-medium text-[var(--color-text)]">
          Run it
        </h2>
        {/* Prose rather than a restatement of the settings: this sentence is about what the
            commands *are*, and the settings are named by the controls that set them. Same line the
            Envelope and Matrix subheads hold. */}
        <p className="mt-1 text-xs text-[var(--color-text-muted)]">
          The flags for this exact placement — the layer split, the cache precision and the window
          above, in each launcher’s own spelling.
        </p>
      </header>

      <div className="flex flex-col gap-5">
        {groups.map((group) => (
          <LauncherBlock key={group.launcher.id} group={group} />
        ))}
      </div>
    </section>
  );
}

function LauncherBlock({ group }: { group: LauncherCommands }) {
  const { launcher, serve, measure } = group;

  /**
   * Both forms refused for the same reason, which means the refusal belongs to the *placement* or
   * the *catalog* rather than to this binary. Stated once, in full — two identical warning blocks
   * read as two problems, which is the same rule the Bench's substitution note already follows.
   */
  const shared = !serve.ok && !measure.ok && serve.reason === measure.reason ? serve.reason : null;

  return (
    <div className="flex flex-col gap-3">
      <h3 className="text-xs font-medium tracking-wide text-[var(--color-text-muted)] uppercase">
        {launcher.label}
      </h3>

      {shared !== null ? (
        <p
          role="note"
          className="rounded-md border border-[var(--color-warning)] p-3 text-xs leading-relaxed text-[var(--color-text-muted)]"
        >
          <span aria-hidden="true" className="text-[var(--color-warning)]">
            ◐{' '}
          </span>
          {shared}
        </p>
      ) : (
        <>
          <Form emission={serve} launcher={launcher.label} kind="Serve" />
          <Form emission={measure} launcher={launcher.label} kind="Measure" />
        </>
      )}

      {/* "A command is a claim, and flags drift" — so every template says where its flags were read
          and when, in the posture every `devices.json` row takes for a device spec. A reader who
          finds a flag renamed has the page to check it against. */}
      <p className="text-[0.625rem] text-[var(--color-text-muted)]">
        Flags checked against{' '}
        {/* `inline-flex` with a 24px floor rather than the bare line box it started as: at
            `text-[0.625rem]` the anchor measured 120x12 and `e2e/touch-targets.spec.ts` failed it
            against WCAG 2.5.8, which is what that sweep is for. The spacing exception might well
            cover a link sitting alone on its own line, but `DisclosureToggle` already records why
            this repo takes the minimum instead: an exception has to be re-argued every time the
            layout moves, and a minimum is checked on every run. 24 and not 44 for the reason the
            sweep's own docblock gives — this is not a crowded target. */}
        <a
          href={launcher.source}
          className="inline-flex min-h-6 items-center underline underline-offset-2"
          target="_blank"
          rel="noreferrer"
        >
          upstream documentation
        </a>{' '}
        on {launcher.checkedOn}.
      </p>
    </div>
  );
}

function Form({
  emission,
  launcher,
  kind,
}: {
  emission: Emission;
  launcher: string;
  kind: 'Serve' | 'Measure';
}) {
  if (!emission.ok) {
    // The complementary refusal: this launcher is simply the other kind of binary. Muted, because
    // it is navigation rather than a problem — the command it points at is on this same panel.
    return <p className="text-xs text-[var(--color-text-muted)]">{emission.reason}</p>;
  }

  return (
    <div className="flex flex-col gap-2">
      {/* The commands are written one flag per line with shell continuations, so most lines are
          short — but a placeholder naming a model and a format is not, and neither is a repo id.
          `overflow-x-auto` keeps that inside its own box: a code block that cannot wrap must scroll
          itself, or it scrolls the document, which is exactly what #34 was.

          **The padding is on the `code`, not on the `pre`, and that is load-bearing.** A scroll
          container's `padding-right` is not honoured at the end of the scroll — the text runs flush
          to the edge — and it also makes `scrollWidth` exceed what any in-flow child reaches, which
          is the exact shape `matrix-header.spec.ts` sweeps for: a container offering to scroll to
          something the layout never asked for. It failed this block at 390px, 457 against 446.
          Moving the padding onto a `w-max` inner box fixes the rule and the visual at once, since
          the padding is then part of the content being scrolled. */}
      {/**
       * **A scrollable box is a tab stop in Chrome, deliberately** — a keyboard reader has to be
       * able to scroll it — and `e2e/focus-indicators.spec.ts` found this one taking focus with no
       * indicator and no name, as an element its sweep could not even see. The answer is to make
       * the stop intentional rather than to suppress it: `tabIndex` declares it, the `aria-label`
       * says which command it is, and the outline is the same 2px accent the four selects use.
       *
       * `outline` rather than a `ring`, for the reason `Controls.tsx` records: a ring is a
       * `box-shadow`, and Safari's forced-colors mode drops those while keeping outlines.
       */}
      <pre
        tabIndex={0}
        aria-label={`${launcher} ${kind.toLowerCase()} command`}
        className="overflow-x-auto rounded-md border border-[var(--color-border)] bg-[var(--color-bg)] text-xs leading-relaxed text-[var(--color-text)] focus:outline-2 focus:outline-offset-2 focus:outline-[var(--color-accent)]"
      >
        <code className="block w-max p-3">{emission.text}</code>
      </pre>

      <CopyButton
        value={emission.text}
        idleLabel={`Copy the ${launcher} ${kind.toLowerCase()} command`}
        copiedLabel="Command copied"
        fallbackLabel="Copy it from here"
        fieldLabel={`${launcher} ${kind.toLowerCase()} command`}
        multiline
      />

      {emission.notes.length > 0 && (
        /* Not behind a disclosure. Each of these is a place the command and the panel above it
           could be read as agreeing when they do not — a cache precision the server cannot be told,
           a concurrency the benchmark client cannot reproduce — and a caveat nobody opens is a
           caveat nobody has. */
        <ul className="flex list-none flex-col gap-1 text-xs leading-relaxed text-[var(--color-text-muted)]">
          {emission.notes.map((note) => (
            <li key={note} className="flex gap-2">
              <span aria-hidden="true">·</span>
              <span>{note}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
