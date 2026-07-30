import type { Measure } from '@/engine/measure';
import type { DeviceSpec, DeviceStatus, KvPrecision, RuntimeSpec } from '@/engine/types';
// The curator's `note` is not an engine field — the engine has no use for it — so the picker's
// prose is typed against the catalog's row. Type-only, so it erases and this module still pulls in
// no data at runtime.
import type { CatalogDevice } from '@/data/catalog';
import { gibLabel, optionLabel, sentences } from './format';
// The scenario *shape*, not the store: `scenario.ts` deliberately depends on nothing but engine
// types so that everything needing the shape can have it without a cycle. Type-only, so it erases.
import type { Config } from '@/store/scenario';

/**
 * The values the controls can actually produce.
 *
 * One definition, because two surfaces read the same scenario and disagreed about its shape:
 * the Bench offered concurrency up to 128 and context up to a model's own ceiling, while the
 * Envelope drew a grid stopping at 64 users and 128K. The region was therefore answering "how
 * much room is left" over a smaller domain than the one you can steer into — the columns that
 * would have gone red were simply not drawn.
 *
 * Log-spaced rather than linear. The interesting jumps in context are 4K → 32K → 128K, and a
 * linear range would spend most of its travel in a region nobody is deciding between.
 */

export const CONTEXT_STOPS = [
  2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576,
] as const;
export const CONCURRENCY_STOPS = [1, 2, 4, 8, 16, 32, 64, 128] as const;
export const PROMPT_STOPS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072] as const;
export const DEVICE_COUNT_STOPS = [1, 2, 4, 8] as const;

/**
 * What each setting is called, wherever it is *labelled*.
 *
 * One entry per `Config` field, because every field is something a control sets and a picture can
 * then draw. The Envelope draws two of them as its axes — so its axis titles, its table's caption
 * and its row-header column read from here rather than restating them. They had already drifted
 * three ways for one setting: the slider said "Concurrent users", the table header said "Users",
 * and the field's own y axis named it nowhere at all (#81).
 *
 * Same reasoning as `kvLabel` below, one level up: two surfaces naming one setting differently is
 * the failure this repo keeps hitting, and it is cheaper to remove than to remember. The four setup
 * settings are here for the same reason and not because a second surface reads them yet — the
 * Matrix's row axis is `<span class="sr-only">Model</span>`, which agrees with the control by
 * coincidence today, exactly the coincidence recorded on `kvLabel`.
 *
 * `satisfies Record<keyof Config, string>` is what makes the keying a claim rather than a comment:
 * rename a `Config` field, or add one, and this fails to compile instead of silently keeping a
 * label for a setting that no longer exists.
 *
 * **Prose is not a label and does not read from here.** "Currently at 32K context and 1 user" is a
 * sentence about a cell, and the Envelope's subhead ("context against concurrent users") is a
 * sentence about the panel; forcing a control's name into either produces "Currently at 32K Context
 * per sequence", which is worse English than the drift it would prevent. The test of which side a
 * surface falls on is whether it *names a setting* — a control label, an axis title, a caption or a
 * column header — or whether it says something about the state in a sentence.
 */
export const SETTING_LABELS = {
  modelId: 'Model',
  deviceId: 'Hardware',
  quantId: 'Quantization',
  runtimeId: 'Runtime',
  contextTokens: 'Context per sequence',
  concurrency: 'Concurrent users',
  promptTokens: 'Prompt length',
  kvPrecision: 'KV precision',
  deviceCount: 'Device count',
} as const satisfies Record<keyof Config, string>;

/**
 * What each measure is called, wherever a grid offers to colour itself by one.
 *
 * Both grids offer the same three — the Matrix over model × device, the Envelope over context ×
 * concurrent users since #65 — and they are the same three questions, so they are named once.
 * `KV_PRECISIONS` above is here for exactly the same reason: the options a control offers are
 * vocabulary, and a second hand-written copy is how two surfaces come to call one thing two things.
 *
 * `hint` is derived from `paints` rather than written beside it, because the two are the same
 * sentence in two registers: the toggle's caption states it as a heading would, and the picture's
 * `aria-label` needs it as a clause ("Coloured by tokens per second for one user"). Written twice,
 * a reworded caption leaves the screen-reader description describing the old colouring.
 *
 * **`ends` names what the two extremes of a ramp *are*, and deliberately not whether they are good.**
 * A ramp whose domain is the grid's own span cannot label its dark end "worse": on gpt-oss-20b at
 * llama.cpp/MXFP4 against an EPYC 9755 every one of the Envelope's 56 cells runs, the `fit` domain is
 * headroom 0.726 to 0.991, and the darkest step — the one `tokens.ts` calls "the one that recedes
 * into the panel" — lands on the 128K x 128-user corner, which still has 72.6% of a 1,450 GiB ceiling
 * free. "Worse" there is a claim about the machine; "less room" is a claim about the ramp, which is
 * all a rank-relative scale is entitled to say. Comparatives rather than superlatives for the same
 * reason.
 *
 * Per measure rather than one generic pair, because one of the three is inverted. `measureOf`
 * returns `1 / ttftSeconds` so that larger is better throughout, which makes a generic "least → most"
 * label read backwards against the caption beside it ("time until the first token appears"): a reader
 * would take the dark end for the *quick* one. The direction has to be spelled in the measure's own
 * units.
 *
 * The Matrix's ramp still says worse/better, and that is not drift: its domain is floored at zero
 * over a grid spanning a desktop CPU to a B200, so a dark cell there really is near the bottom of
 * what any hardware on the page achieves and a verdict word is a claim it can make. See the `fill`
 * comment in `Matrix.tsx` and `magnitudeFill` in `tokens.ts` for the two domains and why they differ.
 *
 * **The entries keep their literal types, so exhaustiveness can be checked below.**
 *
 * `satisfies` rather than a type annotation, and that distinction is the whole point (found in
 * review). Annotating the array `readonly { value: Measure; … }[]` widens each `value` to `Measure`,
 * so `(typeof MEASURES)[number]['value']` *is* `Measure` however few arms are listed — the check the
 * docblock on `measureVocabulary` claimed to be relying on could not fail. `satisfies` validates the
 * same shape while leaving `'fit' | 'decode' | 'ttft'` intact for `UncoveredMeasure` to subtract.
 */
const MEASURE_ENTRIES = [
  {
    value: 'fit',
    label: 'Does it fit',
    paints: 'headroom left after weights, cache and overhead',
    ends: ['less room', 'more room'],
  },
  {
    value: 'decode',
    label: 'How fast',
    paints: 'tokens per second for one user',
    ends: ['slower', 'faster'],
  },
  {
    value: 'ttft',
    label: 'How responsive',
    paints: 'time until the first token appears',
    ends: ['slower to start', 'quicker to start'],
  },
] as const satisfies readonly {
  value: Measure;
  label: string;
  paints: string;
  ends: readonly [string, string];
}[];

/**
 * Any `Measure` with no entry above — and the assertion that there is none.
 *
 * `AssertNever` fails to compile when its argument is inhabited, so adding an arm to `Measure` in the
 * engine breaks *here*, naming the measure that has no control. Without it the omission was silent in
 * both directions a reader would check: these grids would render one fewer button, and
 * `measureVocabulary` would fall back to the fit vocabulary for the new arm — labelling a decode ramp
 * "less room / more room" rather than failing.
 *
 * This is the claim `SETTING_LABELS` makes with `satisfies Record<keyof Config, string>` one screen
 * up. It cannot be made that way here, because the order of these entries is the order of the control
 * and a `Record` does not carry one.
 */
type AssertNever<T extends never> = T;
type UncoveredMeasure = Exclude<Measure, (typeof MEASURE_ENTRIES)[number]['value']>;
export type _EveryMeasureHasAnEntry = AssertNever<UncoveredMeasure>;

export const MEASURES: readonly {
  value: Measure;
  label: string;
  paints: string;
  hint: string;
  ends: readonly [string, string];
}[] = MEASURE_ENTRIES.map((m) => ({
  ...m,
  hint: `${m.paints[0].toUpperCase()}${m.paints.slice(1)}.`,
}));

/**
 * Everything a surface says about the measure in force, in one lookup.
 *
 * The Envelope names the same colouring in four places — the caption under the toggle, the ramp key's
 * two ends, its trailing clause and the canvas `aria-label` — and a per-place `MEASURES.find(...)?.x`
 * makes each of them independently optional for a value that cannot be missing. Total by
 * construction: `Measure` is a closed union and `_EveryMeasureHasAnEntry` above fails to compile if
 * an arm has no entry, so the fallback is unreachable. Unreachable rather than asserted, because a
 * non-null assertion would go on surviving if that check were ever loosened.
 *
 * This docblock previously credited "the annotation on `MEASURES`" with that guarantee, which it
 * never had: annotating the array widened every `value` to `Measure`, so the coverage test was
 * comparing `Measure` against itself and could not fail. The check is real now; the sentence was not.
 */
export function measureVocabulary(measure: Measure): (typeof MEASURES)[number] {
  return MEASURES.find((m) => m.value === measure) ?? MEASURES[0];
}

/**
 * What each setting *means*, in one sentence, for the controls that have to say it themselves.
 *
 * The five Usage controls are the whole KV-cache argument — context times users times bits per
 * token is most of what the budget bar draws — and the panel's entire text content at the default
 * scenario was the labels and the values: "Context per sequence 32K Concurrent users 1 …". The
 * relationship was stated in `Envelope.tsx`'s docstring, in a source comment, and on no surface a
 * reader can see. So was the coupling between the prompt and the context, and so was the fact that
 * KV precision is the same memory-for-quality trade `Quantization` makes for weights (#80).
 *
 * **Persistent text under the control, not a tooltip.** A native `title` needs a mouse and about a
 * second of dwell, does not exist on touch at all, and is invisible to a sighted keyboard user —
 * which is the defect #71 is open against on the Matrix, and taking the hover route here would walk
 * straight into it. `BudgetBar`'s hover-and-focus readout is the right pattern for *many* series
 * sharing *one* reserved line, where the text changes with what you point at; this is five controls
 * with one fixed sentence each, and a fixed sentence should simply be on screen. A disclosure was
 * the fallback and fails for the same reason in reverse: someone who does not know what "KV
 * precision" means does not know to open a disclosure about it.
 *
 * Keyed by `Config` field and kept beside the labels, because a setting's name and its
 * one-sentence explanation are the same vocabulary and drift the same way — this is the level
 * `SETTING_LABELS` already exists at.
 *
 * **Partial, deliberately, and in two different ways.** The four setup selects explain themselves
 * *per option* — the Hardware note is about the machine you picked, not about hardware — so a fixed
 * control-level sentence there would either duplicate or fight with the note that changes
 * underneath it. That is also why `Select`'s `hint` prop is gone rather than wired up here: it
 * rendered only when the selected option had no note of its own, so the explanation would appear
 * and vanish for a reason the reader cannot see.
 *
 * `deviceCount` is absent for the opposite reason: what an extra device *buys* is not a property of
 * the setting, it is a property of the setting and the runtime together, so it cannot be a constant
 * keyed by setting. It is `deviceCountNote` below. Every note in this table is true at every
 * scenario, which is the invariant that makes a table the right shape for them.
 *
 * `satisfies` makes the keying a claim the compiler checks; that every control in the Usage panel
 * actually *has* a sentence is a claim `App.test.tsx` sweeps for, since a control added later
 * cannot fail a type.
 */
export const SETTING_NOTES = {
  contextTokens:
    'The window each user gets: prompt plus everything generated so far. Multiplied by the number of users, this is what sizes the KV cache.',
  concurrency:
    'How many sequences are in flight at once. Each one holds its own cache, so this multiplies memory directly.',
  promptTokens:
    'How much of that context is already filled when generation starts. Part of the context, not extra — it sets how long you wait for the first token.',
  kvPrecision:
    'How many bits each cached token costs. Narrower shrinks the cache and can cost quality, the same trade quantization makes for weights.',
} as const satisfies Partial<Record<keyof Config, string>>;

/**
 * What a second device buys you, which depends entirely on how the runtime splits the model.
 *
 * The draft copy for this control said "shard the model across, tensor-parallel. Adds memory and
 * bandwidth, minus what the interconnect costs." That is true of vLLM and of nothing else the app
 * offers. llama.cpp — the default runtime — and MLX both declare `parallelism: 'layer'`, and
 * `achievedBandwidth` and the FLOPS closure in `speed.ts` both return the *per-device* figure and
 * short-circuit before `effectiveDeviceCount` for a layer split. So at the default scenario the
 * reader drags Device count from 1x to 4x, every speed figure on the page holds still to the last
 * decimal — 35.57 tok/s decode, 1438 tok/s prefill on a DGX Spark — and a sentence directly beneath
 * the slider credits an interconnect penalty against a bandwidth gain that was never evaluated.
 *
 * `docs/ROADMAP.md` records that derivation as one that was wrong first and is silent when it
 * breaks: "A layer split is not a speedup ... a single stream sees one card's bandwidth and one
 * card's FLOPS however many cards there are — that rig buys capacity, not speed." Putting the claim
 * back on screen is the same mistake one layer up, so the sentence reads `parallelism` for the same
 * reason the arithmetic does.
 *
 * A layer split can still make the page faster, and the note deliberately does not deny it: on a
 * 4090 at Q4_K_M this model decodes 14.25 tok/s spilling to host memory on one card and 190.11 on
 * four, because the fourth card is what stops it spilling. That is the capacity, arriving as speed.
 * "Buys capacity, not speed" is a claim about the mechanism, which is why it is the clause here.
 */
/**
 * `drives` is a parameter rather than a `runtimeDrives(runtime, device)` call inside, because this
 * module takes nothing from the engine but its *types* — see the import block, where every engine
 * and catalog import is type-only — and the caller already has the answer for the warning it prints
 * on the Runtime control. `devicePickerNote` below takes its ceiling for the same reason.
 *
 * It exists because `canShard` asks only about the hardware. A DGX Spark has an interconnect, so the
 * slider renders under MLX, which cannot drive that machine at all — and this note then described a
 * layer split buying capacity, three controls below "Does not run on NVIDIA DGX Spark". Two
 * sentences on one screen, one of them describing an evaluation that never happens. The unsupported
 * branch is deliberately not silent: the control is still there and still stores a value, so it
 * needs to say why nothing it does moves a figure.
 */
export function deviceCountNote(runtime: RuntimeSpec, drives: boolean): string {
  const opening = 'How many of this machine to shard the model across.';
  if (!drives) {
    return `${opening} ${runtime.label} does not run on this machine, so nothing here is evaluated — what a second device buys is decided by the runtime that ends up driving it.`;
  }
  return runtime.parallelism === 'layer'
    ? `${opening} ${runtime.label} runs whole layers on each device in turn, so this buys capacity, not speed — one device’s bandwidth is the ceiling however many you add.`
    : `${opening} ${runtime.label} shards every layer across every device, so this adds bandwidth as well as memory, minus what the interconnect costs.`;
}

/**
 * `Record<Exclude<DeviceStatus, 'shipping'>, string>` rather than a ternary, and that is the whole
 * reason it is a table: a fourth status added to the engine's union fails to compile *here*, naming
 * the status that has no word for it. A ternary keeps compiling and quietly calls it "announced" —
 * the same claim `DEVICE_STATUSES` makes about the loader in `catalog.ts`, at the other end of the
 * same field.
 *
 * The catalog spells the field `rumored` and the app says "rumoured": the data follows its schema,
 * the UI follows its own register, and this is the one place the two are reconciled.
 */
const PRE_RELEASE_WORDS: Record<Exclude<DeviceStatus, 'shipping'>, string> = {
  rumored: 'rumoured',
  announced: 'announced',
};

/**
 * What a status other than `shipping` is called where a reader sees it, or `undefined` for hardware
 * that exists.
 *
 * One resolution, read by both halves of what the Hardware picker says — the marker in the
 * `<option>` and the warning in the note — so the two cannot come to disagree about which rows are
 * pre-release. It was one expression in each of two places before, and only one of them ran at the
 * point of choice (#69).
 */
function preReleaseWord(status: DeviceStatus): string | undefined {
  return status === 'shipping' ? undefined : PRE_RELEASE_WORDS[status];
}

/**
 * What the Hardware picker calls a machine in the open list, which is the only place a reader
 * compares two of them.
 *
 * The pre-release marker is here rather than in the note because of where the browser renders each:
 * `Select` draws every option's *label* and only the selected option's *note*, so the caveat that
 * decides whether a row is hardware at all was reachable only after the row had been chosen. In the
 * list "Mac Studio M5 Ultra (512 GB) — 512 GiB" sat one line above the 512 GB M3 Ultra — a real
 * machine with measured bandwidth — presented as its equal (#69). CLAUDE.md states the rule as a
 * requirement rather than a preference: pre-release specs must stay visibly labelled in the UI, and
 * a label that waits for the selection is not visible where the comparison happens.
 *
 * **Not an `<optgroup>`, which the issue floats as the stronger version.** A group is contiguous, so
 * grouping the non-shipping rows imposes an order on the picker — and the order of the hardware list
 * is a separate question with an issue of its own (#79). A marker holds for any list order and for
 * any number of pre-release rows, including the zero of them this catalog may have after the M5
 * ships; a group of one row is a heading with nothing to group.
 *
 * The note keeps the fuller sentence. This is a tag on a row you are scanning; that is a clause for
 * the row you chose, and `devicePickerNote` below is where it is composed.
 */
export function deviceOptionLabel(device: DeviceSpec): string {
  return optionLabel(
    `${device.name} — ${gibLabel(device.capacityBytes)}`,
    preReleaseWord(device.status)
  );
}

/**
 * The same rule one control over, for the fact that decides whether any figure on the page means
 * anything.
 *
 * `runtimeOptions` has said "Does not run on <machine>." since the Runtime picker existed, in the
 * note — so on a Mac Studio the open list offered llama.cpp, vLLM and MLX as three equals, and the
 * refusal arrived only after vLLM had been picked and every tile on the page had been replaced by
 * "Unsupported". That is the pre-release defect exactly: a caveat a reader needs in order to choose,
 * living in the one string a `<select>` will not show them until they have. Swept rather than waited
 * for, because these two pickers share a component and therefore share the failure.
 *
 * **"on this hardware", not "here", and the difference is the whole point of moving the fact.** An
 * option's own text is *all* that is announced for a row nobody has selected — a screen-reader user
 * arrowing this listbox hears "vLLM · …" and no surrounding context — so a marker whose referent
 * lives outside itself has reintroduced, one word smaller, the dependency it was written to remove:
 * "here" resolves to nothing, and the string that names the machine is still the selected option's
 * note. "this hardware" points at the control one row up, which is labelled Hardware (found in
 * review). Not the machine's name, which is the note's job: this is a tag on a row being scanned, and
 * `Does not run on Mac Studio M3 Ultra (256 GB).` inside three of them is a paragraph in a picker.
 *
 * `drives` is a parameter for the reason `deviceCountNote` above takes one: this module reads engine
 * types and no engine values, and the caller already has the answer.
 *
 * Marked rather than `disabled`. An unsupported pairing is a legitimate thing to select here — the
 * page's answer to it is a full explanation of the refusal, which is more use than a row that cannot
 * be clicked and says nothing about why.
 */
export function runtimeOptionLabel(runtime: RuntimeSpec, drives: boolean): string {
  return optionLabel(runtime.label, !drives && 'does not run on this hardware');
}

/**
 * What the Hardware picker says about the selected machine, split into the two different kinds of
 * text that used to be one string.
 *
 * **`claim` is picker copy**: the short, derived facts a reader needs *while choosing* — that these
 * specs are not final, and that the allocation ceiling is a default rather than a wall. Both are
 * generated here, both are one clause, and both end in a full stop of their own so that nothing
 * downstream has to guess where they finish.
 *
 * **`detail` is reference prose**: `devices.json`'s `note`, which is provenance for a reader who
 * has already chosen — which GPU bin the figures describe, why a bandwidth rating is not the
 * measured figure, that a 3090 pair is modelled at PCIe rates. It is 40 to 180 words of it, with
 * backticked sysctl names and a derivation, and it was being concatenated onto the claim and handed
 * to the control as its `aria-describedby` (#68). So a screen-reader user heard the whole
 * derivation before they could choose anything, and a sighted reader got five lines of 12px prose
 * under a `<select>` in a two-column grid — which pushed the row beneath it down and left a void
 * under Model. `Select` puts this behind a disclosure and keeps it out of the description.
 *
 * This is the trimming `quantOptions` in `Bench.tsx` already describes and the device note never
 * got: "a short claim, not the whole derivation — the panel below carries that, and printing the
 * same forty words twice on one screen taught people to skip both."
 *
 * **The curated note is still on screen**, which is the constraint this cannot trade away: it was
 * dropped entirely once before, taking with it the 3090's warning that the estimates assume PCIe
 * and do not model its optional NVLink bridge — precisely the caveat an owner of a bridged pair
 * needs. A disclosure is one click, not a deletion.
 *
 * `ceilingBytes` is a parameter rather than a `maxAllocatablePerDevice(device)` call inside, for the
 * same reason `deviceCountNote` takes `drives`: this module reads engine *types* and no engine
 * values, and the caller already holds the figure. Passing it also keeps the "you could raise this"
 * arithmetic in the one place that owns it — the Bench and the Envelope each had their own copy of
 * that once, which is how one of them came to be wrong.
 */
export function devicePickerNote(
  device: CatalogDevice,
  ceilingBytes: number
): { claim?: string; detail?: string } {
  const preRelease = preReleaseWord(device.status);
  return {
    claim: sentences(
      // Pre-release specs must stay visibly labelled, not silently mixed in with shipping ones. The
      // marker in the option label is the half of that a reader sees while choosing; this is the
      // sentence, which says what being pre-release costs them. Sentence case from the shared word
      // rather than a second ternary over `status`: the two surfaces have to name the same rows, and
      // `PRE_RELEASE_WORDS` is where they agree.
      preRelease && `${preRelease[0].toUpperCase()}${preRelease.slice(1)} — specs may change.`,
      // The tunable ceiling matters for the same reason in reverse: it is a default, and treating
      // it as a hardware limit turns a raiseable setting into a flat "will not run".
      device.allocatableTunable === true &&
        ceilingBytes > device.allocatableBytes &&
        `${gibLabel(device.allocatableBytes)} allocatable by default, raiseable to ${gibLabel(
          ceilingBytes
        )}.`
    ),
    detail: device.note,
  };
}

/**
 * The cache precisions a control can offer, with the name to use when a runtime has none of
 * its own.
 */
export const KV_PRECISIONS: readonly { value: KvPrecision; label: string }[] = [
  { value: 'fp16', label: 'FP16' },
  { value: 'q8', label: 'Q8' },
  { value: 'q4', label: 'Q4' },
];

const KV_FALLBACK_LABELS = new Map(KV_PRECISIONS.map((k) => [k.value, k.label]));

/**
 * What a runtime calls a cache precision.
 *
 * `KvPrecision` is an internal width, not a name anyone types: vLLM's one-byte cache is FP8 with
 * no integer-Q8 option at all, so the catalog gives it a `kvLabels` entry and the control names
 * something the user could actually pass on a command line.
 *
 * One function because there were two resolutions and they disagreed about the fallback — the
 * Bench control read this table while the Matrix heading upper-cased the raw value. They agree on
 * all three current precisions by coincidence, and the first one whose display name is not its id
 * in capitals (`fp8_e5m2`, `q4_0`) would have had the two surfaces printing different names for
 * one setting. That is the failure this repo keeps hitting, and it is cheaper to remove than to
 * remember.
 */
export function kvLabel(runtime: RuntimeSpec, precision: KvPrecision): string {
  return (
    runtime.kvLabels?.[precision] ?? KV_FALLBACK_LABELS.get(precision) ?? precision.toUpperCase()
  );
}

/**
 * The fixed stops plus whatever is currently stored.
 *
 * `coerce` accepts any integer in range, so a value arriving from a URL — `?u=3` — has to be
 * offered too, or the control displays 2 while the engine evaluates 3.
 */
export function withStored(values: readonly number[], stored: number): number[] {
  return [...new Set([...values, stored])].sort((a, b) => a - b);
}

/**
 * Contexts this model can reach, plus its exact ceiling, plus whatever is selected.
 *
 * The ceiling is included as its own stop because it is rarely a power of two — Qwen3 stops at
 * 40,960 — and `coerce` clamps to it. Without it the largest offered value was 32,768 while the
 * model would hold a quarter more, and the Envelope's rightmost column understated the room.
 */
export function contextStopsFor(maxContext: number, stored: number): number[] {
  const within = CONTEXT_STOPS.filter((t) => t < maxContext);
  const stops = new Set([...within, maxContext, Math.min(stored, maxContext)]);
  return [...stops].filter((t) => t <= maxContext).sort((a, b) => a - b);
}
