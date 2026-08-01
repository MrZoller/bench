import { useId, useMemo, useState } from 'react';
import type { Evaluation } from '@/engine';
import { effectivePromptTokens } from '@/engine/placement';
import {
  CALIBRATION_BAND,
  compare,
  hasSubmittablePair,
  parseLlamaBench,
  submissionUrl,
  type Comparison,
} from '@/lib/calibrate';
import { useConfig, type Config } from '@/store/config';
import { configToShareSearch } from '@/store/url';
import { getDevice, getModel } from '@/data/catalog';
import { getQuant } from '@/data/quants';
import { DisclosureToggle } from './DisclosureToggle';

/**
 * Predicted versus measured (#139).
 *
 * The verify affordance. Every other calculator's numbers are rules of thumb; nobody can check them
 * and nobody is asked to. Paste `llama-bench` output beside the prediction for the same scenario,
 * see both, and — optionally — submit the pair as a pre-filled issue.
 *
 * **Behind a disclosure, and that is a judgement rather than a default.** This is the one panel on
 * the page a reader has to have already done something to use: it wants the output of a command
 * they ran on their own machine. Open by default it is a large empty textarea on every page view,
 * which is what the surface would look like if the feature were about collecting data rather than
 * about being checkable.
 *
 * **The rule that guards it, from the issue and unchanged: measurements accumulate as evidence, and
 * retuning stays a deliberate act.** Nothing here writes to the engine's constants, and nothing
 * about a submitted number is applied automatically. Submissions in bulk are how
 * `bandwidthEfficiency` × `CLASS_BANDWIDTH_UTILIZATION` finally becomes identifiable; a drip of
 * individually-absorbed measurements is how the next masked error gets fitted into the constants
 * silently.
 */

const REPO_URL = 'https://github.com/MrZoller/bench';

/**
 * The cache precision in llama.cpp's own spelling, which is what a paste will carry.
 *
 * The same mapping the launch emitter uses for `-ctk`/`-ctv`; duplicated rather than imported only
 * because that module is not on this branch, and the two must be merged when both land.
 */
const LLAMA_KV_TYPES: Record<string, string> = { fp16: 'f16', q8: 'q8_0', q4: 'q4_0' };

export function Calibrate({ evaluation }: { evaluation: Evaluation }) {
  const headingId = useId();
  const fieldId = useId();
  const regionId = useId();
  const config = useConfig();
  const [expanded, setExpanded] = useState(false);
  const [pasted, setPasted] = useState('');

  const model = getModel(config.modelId);
  const device = getDevice(config.deviceId);

  const comparisons = useMemo(() => {
    const measurements = parseLlamaBench(pasted);
    if (measurements.length === 0) return [];

    /**
     * The prediction, taken from the same evaluation every panel above reads.
     *
     * `effectivePromptTokens` rather than `config.promptTokens`, because that is the boundary
     * `estimatePrefill` actually times — a scenario that leaves the prompt unset is timed at 90% of
     * the window, and comparing a measurement against the unset field would be comparing it to a
     * figure the engine never used.
     */
    const usage = {
      contextTokens: config.contextTokens,
      concurrency: config.concurrency,
      promptTokens: config.promptTokens,
      kvPrecision: config.kvPrecision,
    };
    const promptTokens = effectivePromptTokens(usage);

    return compare(measurements, {
      prefillTokensPerSec: evaluation.prefill.prefillTokensPerSec,
      decodeTokensPerSec: evaluation.decode.perUserTokensPerSec,
      promptTokens,
      generationTokens: Math.max(1, config.contextTokens - promptTokens),
      /**
       * The whole window, because that is what `estimateDecode` charges every step's cache read at
       * — not the prompt, which was the first version's answer. In the default 8K-prompt/32K-context
       * scenario a run near 8K depth was being graded against the predicted 32K-cache rate, and the
       * run at the modelled 32K was marked as the mismatch.
       */
      residentContextTokens: config.contextTokens,
      concurrency: config.concurrency,
      /**
       * The four fields that let `describeMismatch` see the *job* rather than only the numbers.
       *
       * Without them a paste at a different quantization, a different cache precision, a different
       * layer split or a different runtime reads as a clean percentage — which is the shape of a
       * measurement of something else, presented as evidence about this one.
       *
       * `cachedPrefixTokens` is deliberately absent: it is a property of a verdict archetype and not
       * of the Bench's own sliders, whose three tiles describe a standalone prompt. Prefill's
       * expected depth is therefore zero here, and that is the right answer for this panel rather
       * than an omission.
       */
      runtimeId: config.runtimeId,
      quantLabel: getQuant(config.quantId).label,
      modelName: model.name,
      kvType: LLAMA_KV_TYPES[config.kvPrecision],
      /**
       * Stated only when the placement is fully resident, where "every layer" is unambiguous.
       * Where it spills there is no layer count to compare against without converting a rig-wide
       * fraction into one, which is the derivation #14 was about — so the field is omitted and the
       * check is skipped rather than made up.
       */
      /**
       * Zero on a machine with no GPU, and that distinction is what the EPYC anchor depends on.
       * `planPlacement` reports `offloadFraction === 0` for a `cpu-ram` rig because there is no
       * faster tier to spill *from* — so reading that as "every layer on the GPU" marked exactly
       * the CPU measurements this feature exists to collect as a different job.
       */
      ...(device.class === 'cpu-ram'
        ? { gpuLayers: 0 }
        : evaluation.placement.offloadFraction === 0
          ? { gpuLayers: model.layers }
          : {}),
    });
  }, [pasted, config, evaluation, model.layers, model.name, device.class]);

  const href = useMemo(
    () =>
      submissionUrl({
        repoUrl: REPO_URL,
        scenarioUrl: `${window.location.origin}${window.location.pathname}${configToShareSearch(config as Config)}`,
        deviceName: device.name,
        deviceCount: config.deviceCount,
        modelName: model.name,
        comparisons,
      }),
    [config, device.name, model.name, comparisons]
  );

  return (
    <section aria-labelledby={headingId} className="panel p-[min(1.25rem,5vw)]">
      <h2 id={headingId} className="text-sm font-medium text-[var(--color-text)]">
        Check these numbers
      </h2>
      <p className="mt-1 text-xs leading-relaxed text-[var(--color-text-muted)]">
        Every figure on this page is a model, calibrated on two measured machines and asserted
        within ±{Math.round(CALIBRATION_BAND * 100)}%. Paste your own <code>llama-bench</code>{' '}
        output and see both numbers side by side.
      </p>

      <DisclosureToggle
        expanded={expanded}
        onToggle={() => setExpanded((open) => !open)}
        controls={regionId}
      >
        {expanded ? 'Hide' : 'Paste a measurement'}
      </DisclosureToggle>

      {/* In the DOM in both states, `hidden` rather than unmounted — `aria-controls` renders
          unconditionally and pointing it at an id that does not exist is the reference-integrity
          violation #131 fixed at the other four call sites. */}
      <div id={regionId} hidden={!expanded} className="mt-3 flex flex-col gap-3">
        <label htmlFor={fieldId} className="text-xs text-[var(--color-text-muted)]">
          llama-bench output — the markdown it prints by default, or <code>-o json</code>, which
          also carries the build commit
        </label>
        {/* Before the field, not under the submission link where it started: a reader deciding
            whether to paste the output of a command they ran on their own machine needs this
            first, and under the link it only ever reassured people who had already pasted. */}
        <p className="text-[0.625rem] leading-relaxed text-[var(--color-text-muted)]">
          Parsed here in your browser. The text never leaves this page, and nothing is sent anywhere
          unless you choose to open the pre-filled issue below.
        </p>
        <textarea
          id={fieldId}
          value={pasted}
          onChange={(event) => setPasted(event.target.value)}
          rows={6}
          spellCheck={false}
          placeholder="| model | size | params | backend | ngl | test | t/s |"
          className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-bg)] p-2 font-mono text-xs text-[var(--color-text)]"
        />

        {pasted.trim() !== '' && comparisons.length === 0 && (
          /* A paste that parsed to nothing is a mistake to name, not an exception to throw. */
          <p className="text-xs text-[var(--color-text-muted)]">
            No benchmark rows in that. llama-bench prints a table with a <code>test</code> column
            reading <code>pp512</code> or <code>tg128</code>; paste the whole thing, headers and
            all.
          </p>
        )}

        {comparisons.length > 0 && (
          <>
            <table className="w-full text-xs">
              <caption className="sr-only">
                Each measured row against what this page predicted for it
              </caption>
              <thead className="text-[var(--color-text-muted)]">
                <tr>
                  <th scope="col" className="py-1 text-left font-normal">
                    Measure
                  </th>
                  <th scope="col" className="py-1 text-right font-normal">
                    Predicted
                  </th>
                  <th scope="col" className="py-1 text-right font-normal">
                    Measured
                  </th>
                  <th scope="col" className="py-1 text-right font-normal">
                    Error
                  </th>
                </tr>
              </thead>
              <tbody>
                {comparisons.map((c, index) => (
                  <Row
                    key={`${c.measurement.kind}:${c.measurement.tokens}:${index}`}
                    comparison={c}
                  />
                ))}
              </tbody>
            </table>

            {/* The link, and only once there is a *comparable* pair to submit — which is the whole
                reason the mismatches are computed. A row the panel has just called "not comparable"
                would otherwise reach the issue table as a bare percentage with its explanation
                stripped, which is a difference between two jobs entering the record as evidence.
                The scenario link is the other half: a measurement that cannot say which machine it
                was taken on is unusable, so the template makes that field the first line. */}
            {!hasSubmittablePair(comparisons) ? (
              <p className="text-xs leading-relaxed text-[var(--color-text-muted)]">
                Nothing here is comparable with the figures above, so there is nothing to submit.
                Fix the runs the rows describe — or change the scenario to match them — and the link
                comes back.
              </p>
            ) : (
              <a
                href={href}
                target="_blank"
                rel="noreferrer"
                className="inline-flex min-h-11 items-center self-start rounded-md border border-[var(--color-border)] px-3 py-1.5 text-xs text-[var(--color-accent)] hover:border-[var(--color-accent-dim)]"
              >
                Submit this pair as an issue
              </a>
            )}
            <p className="text-[0.625rem] leading-relaxed text-[var(--color-text-muted)]">
              It opens a pre-filled GitHub form carrying this scenario's link and the figures above,
              and you see the issue before you post it. Misses are published with the same weight as
              hits; a record that only ever shows wins is marketing rather than calibration.
            </p>
          </>
        )}
      </div>
    </section>
  );
}

function Row({ comparison }: { comparison: Comparison }) {
  const { measurement, predicted, error, withinBand, mismatch } = comparison;
  const percent = `${error > 0 ? '+' : ''}${Math.round(error * 100)}%`;

  return (
    <>
      <tr className="border-t border-[var(--color-border)]">
        <th scope="row" className="py-1 text-left font-normal text-[var(--color-text)]">
          {measurement.kind === 'prefill' ? 'Prefill' : 'Decode'}{' '}
          <span className="text-[var(--color-text-muted)]">
            at {measurement.tokens.toLocaleString('en-US')} tokens
            {measurement.depthTokens === undefined
              ? ''
              : `, depth ${measurement.depthTokens.toLocaleString('en-US')}`}
          </span>
        </th>
        <td className="tabular py-1 text-right">{predicted.toFixed(1)}</td>
        <td className="tabular py-1 text-right">{measurement.tokensPerSec.toFixed(1)}</td>
        {/* The word beside the colour, like every other graded figure on this page: a verdict
            carried by hue alone is not a verdict for everyone. `mismatch` overrides both, because
            a delta between two different jobs is not a hit or a miss — it is not a reading. */}
        <td
          className="tabular py-1 text-right"
          style={{
            color: mismatch
              ? 'var(--color-text-muted)'
              : withinBand
                ? 'var(--color-good)'
                : 'var(--color-warning)',
          }}
        >
          {mismatch ? '—' : percent}
          <span className="sr-only">
            {mismatch
              ? ', not comparable'
              : withinBand
                ? `, inside the ±${Math.round(CALIBRATION_BAND * 100)}% band`
                : `, outside the ±${Math.round(CALIBRATION_BAND * 100)}% band`}
          </span>
        </td>
      </tr>
      {mismatch !== undefined && (
        <tr>
          <td colSpan={4} className="pb-1 text-xs leading-relaxed text-[var(--color-warning)]">
            ◐ {mismatch}
          </td>
        </tr>
      )}
    </>
  );
}
