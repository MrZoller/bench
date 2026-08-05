import type { DeviceSpec, ModelSpec } from '@/engine/types';
import { GB, GIB, TFLOP } from '@/engine/types';
import devicesJson from './devices.json';
import modelsJson from './models.generated.json';

/**
 * Loads both catalogs into engine types.
 *
 * The two files are stored in the units their sources use — GiB for capacity, GB/s for
 * bandwidth, dense TFLOPS for compute — so a reviewer can check a row against a datasheet
 * without doing arithmetic. Conversion to the engine's bytes-and-bytes-per-second happens
 * here, once, rather than being duplicated at every call site.
 */

export interface DeviceRow {
  id: string;
  name: string;
  vendor: string;
  class: string;
  status: string;
  capacityGiB: number;
  allocatableGiB: number;
  allocatableTunable?: boolean;
  maxAllocatableGiB?: number;
  bandwidthGBs: number;
  // Rows list only the dtypes their datasheet publishes, so TypeScript widens the absent keys
  // to `undefined` across the union of row shapes. Modelled honestly rather than cast away.
  tflops: Record<string, number | undefined>;
  interconnect?: string;
  hostLinkGBs?: number;
  tdpWatts?: number;
  msrpUsd?: number;
  releasedAt?: string;
  source: string;
  note?: string;
}

/** A device plus the curator's note, which the UI shows but the engine has no use for. */
export interface CatalogDevice extends DeviceSpec {
  note?: string;
}

/**
 * The string fields whose values the engine narrows to a union, and the members it accepts.
 *
 * Written as records keyed by the union rather than as arrays, so adding a member to `DeviceClass`
 * or `DeviceStatus` without listing it here is a compile error. A list of accepted values that can
 * silently fall behind the type it guards is the same class of defect as no list at all.
 */
const DEVICE_CLASSES: Record<DeviceSpec['class'], true> = {
  'discrete-gpu': true,
  'unified-soc': true,
  'cpu-ram': true,
};

const DEVICE_STATUSES: Record<DeviceSpec['status'], true> = {
  shipping: true,
  announced: true,
  rumored: true,
};

const FLOPS_DTYPES: Record<keyof DeviceSpec['flops'], true> = {
  fp16: true,
  bf16: true,
  fp8: true,
  fp4: true,
  int8: true,
};

/**
 * Narrow a hand-typed string to the union the engine expects, or fail at load saying so.
 *
 * The JSON import types every one of these as `string`, so the cast is the only thing standing
 * between a typo and the engine — and a cast checks nothing. `toModel` makes this argument for the
 * *generated* catalog, which is machine-written from one script that always emits the same shape.
 * `devices.json` is edited by hand, a row at a time, against datasheets. It needs it more.
 *
 * What a typo buys without this, in each field:
 *
 *   - **`class`** — every runtime's `supports` check misses, so the device reports as driven by
 *     nothing at all, and `CLASS_BANDWIDTH_UTILIZATION[class]` is `undefined` underneath, which
 *     takes decode to zero. A confident wrong sentence on every surface, for every runtime.
 *   - **`status`** — the Matrix filters to `shipping`, so the device silently vanishes from the
 *     comparison grid, and the Bench picker labels it "Announced — specs may change".
 *   - **a `tflops` key** — the loudest of the three and the one the issue did not name: a
 *     misspelled `fp16` leaves `peakFlops` with nothing to fall back to, and prefill divides by a
 *     zero rate. The catalogued RTX 5090 reports a time to first token of `Infinity`.
 */
function narrow<T extends string>(
  value: string,
  allowed: Record<T, true>,
  field: string,
  rowId: string
): T {
  if (!Object.hasOwn(allowed, value)) {
    throw new Error(
      `Catalog device ${rowId || '<unknown>'} has unsupported ${field} "${value}". ` +
        `Expected one of: ${Object.keys(allowed).join(', ')}.`
    );
  }
  return value as T;
}

/**
 * Exported alongside `DeviceRow` so the guard above is reachable from a test.
 *
 * `DEVICES` cannot exercise it: every committed row is correct today, which is exactly why a
 * defensive check here would otherwise ship untested. Proving it rejects anything means building a
 * row that should be rejected.
 */
export function toDevice(row: DeviceRow): CatalogDevice {
  const flops: DeviceSpec['flops'] = {};
  for (const [dtype, value] of Object.entries(row.tflops)) {
    if (value === undefined) continue;
    flops[narrow(dtype, FLOPS_DTYPES, 'compute dtype', row.id)] = value * TFLOP;
  }

  /**
   * A raiseable ceiling has to say how far, and it can never be all of physical memory.
   *
   * Two fields that only mean something together, and the pairing was unenforced: every Apple row
   * declared `allocatableTunable` and stated no maximum, so `maxAllocatablePerDevice` fell back to
   * physical capacity and the app offered the owner of a 96 GiB Mac Studio a 95.5 GiB
   * configuration — a machine with nothing left for the OS, the window server, or the inference
   * process's own unwired allocations. Enforced here rather than left to the curator, because the
   * failure is silent on every surface and reads as generosity.
   */
  if (row.allocatableTunable) {
    if (row.maxAllocatableGiB === undefined) {
      throw new Error(
        `Catalog device ${row.id || '<unknown>'} is allocatableTunable with no maxAllocatableGiB. ` +
          'State how far the ceiling actually raises; it is never all of physical memory.'
      );
    }
    if (row.maxAllocatableGiB >= row.capacityGiB) {
      throw new Error(
        `Catalog device ${row.id || '<unknown>'} raises its allocation ceiling to ` +
          `${row.maxAllocatableGiB} of ${row.capacityGiB} GiB. The platform accepts that value; ` +
          'the machine does not — reserve room for the OS.'
      );
    }
    if (row.maxAllocatableGiB < row.allocatableGiB) {
      throw new Error(
        `Catalog device ${row.id || '<unknown>'} states a maximum (${row.maxAllocatableGiB} GiB) ` +
          `below its own default (${row.allocatableGiB} GiB).`
      );
    }
  } else if (row.maxAllocatableGiB !== undefined) {
    // The same pairing, enforced from the other side. A row stating a ceiling without the flag
    // that gives it meaning has its figure dropped by `maxAllocatablePerDevice` and shows up
    // nowhere — silently, and as the failure of a curator who did the work rather than one who
    // skipped it. That is the worse of the two to swallow.
    throw new Error(
      `Catalog device ${row.id || '<unknown>'} states maxAllocatableGiB without ` +
        'allocatableTunable. The two only mean anything together: without the flag nothing reads ' +
        'the ceiling.'
    );
  }

  return {
    id: row.id,
    name: row.name,
    vendor: row.vendor,
    class: narrow(row.class, DEVICE_CLASSES, 'class', row.id),
    status: narrow(row.status, DEVICE_STATUSES, 'status', row.id),
    capacityBytes: row.capacityGiB * GIB,
    allocatableBytes: row.allocatableGiB * GIB,
    ...(row.allocatableTunable ? { allocatableTunable: true } : {}),
    ...(row.maxAllocatableGiB === undefined
      ? {}
      : { maxAllocatableBytes: row.maxAllocatableGiB * GIB }),
    bandwidthBytesPerSec: row.bandwidthGBs * GB,
    flops,
    ...(row.interconnect ? { interconnect: row.interconnect } : {}),
    ...(row.hostLinkGBs === undefined ? {} : { hostLinkBytesPerSec: row.hostLinkGBs * 1e9 }),
    ...(row.tdpWatts ? { tdpWatts: row.tdpWatts } : {}),
    ...(row.msrpUsd ? { msrpUsd: row.msrpUsd } : {}),
    ...(row.releasedAt ? { releasedAt: row.releasedAt } : {}),
    source: row.source,
    ...(row.note ? { note: row.note } : {}),
  };
}

export const DEVICES: readonly CatalogDevice[] = (devicesJson.devices as DeviceRow[]).map(toDevice);

/**
 * Ids that used to name a row, and the row they name now.
 *
 * A device id is not an internal detail: `url.ts` writes it into every shared scenario link as `d`,
 * and those links are meant to survive in forum threads for months. Renaming a row without an alias
 * does not break such a link loudly — `coerce` cannot resolve the id, falls back to the default
 * device, and the reader is shown a different machine's numbers under the sender's URL. That is the
 * worst of the three possible outcomes, and the cheapest to avoid.
 *
 * Kept as data rather than as a rename-and-forget, because the pair (old id, new id) is the only
 * record that the old one was ever real.
 */
export const DEVICE_ID_ALIASES: Readonly<Record<string, string>> = {
  // `rtx-a6000-ada` fused two products that both exist: the Ampere card is the RTX A6000 and the
  // Ada one is the RTX 6000 Ada Generation. Every spec on the row was the Ada card's and only the
  // id was wrong, so this is a rename rather than a correction — but it is a rename of the thing
  // that ends up in other people's links.
  'rtx-a6000-ada': 'rtx-6000-ada',
};

/**
 * The current id for a device, following an alias if the caller has an old one.
 *
 * Exported because the *store* has to canonicalise before it keeps the value: resolving only inside
 * `getDevice` would return the right device while leaving the stale id in the config, which then
 * re-encodes into the URL and matches no `<option>` in the hardware picker — a control showing
 * nothing selected beside figures for a device that is genuinely loaded.
 *
 * `Object.hasOwn` rather than a lookup with `??`, for the same reason `narrow` above uses it: the
 * ids arrive from a querystring, and `DEVICE_ID_ALIASES['toString']` resolves up the prototype chain
 * to a function. That is not nullish, so `??` would not fire and this would return something that is
 * not a string at all from a signature promising one.
 */
export function canonicalDeviceId(id: string): string {
  return Object.hasOwn(DEVICE_ID_ALIASES, id) ? DEVICE_ID_ALIASES[id] : id;
}

/**
 * A model plus the note explaining any hand-entered correction the generator applied.
 *
 * Mirrors {@link CatalogDevice}: the seed list requires a reason whenever a figure is typed by
 * a human rather than derived, and that reason has to be reachable from the UI or the
 * requirement is decorative. Six catalogued models carry an overridden `totalParams`.
 */
export interface CatalogModel extends ModelSpec {
  overrideNote?: string;
}

/**
 * Validates the generated catalog at load rather than trusting it.
 *
 * The file is machine-written from network data, so a shape the engine can't handle — an
 * attention kind its switch doesn't cover, most obviously — would otherwise surface as NaN
 * deep in a throughput readout instead of as an error at startup.
 */
function toModel(raw: unknown): CatalogModel {
  const model = raw as CatalogModel;
  const kind = model.attention?.core?.kind;
  if (kind !== 'gqa' && kind !== 'mla') {
    throw new Error(
      `Catalog model ${model.id ?? '<unknown>'} has unsupported attention kind "${kind}". ` +
        'Regenerate with `npm run catalog`.'
    );
  }
  // A catalog generated before these fields existed leaves them undefined, and every throughput
  // figure downstream would quietly become NaN rather than failing here.
  for (const [field, value] of [
    ['activeDenseParams', model.activeDenseParams],
    ['attention.projectionWidth', model.attention?.projectionWidth],
  ] as const) {
    if (!Number.isFinite(value) || (value as number) <= 0) {
      throw new Error(
        `Catalog model ${model.id ?? '<unknown>'} has no usable ${field}. ` +
          'Regenerate with `npm run catalog`.'
      );
    }
  }
  return model;
}

export const MODELS: readonly CatalogModel[] = modelsJson.models.map(toModel);

/** When the model catalog was last regenerated — surfaced in the UI so staleness is visible. */
export const CATALOG_GENERATED_AT: string = modelsJson.generatedAt;

const DEVICES_BY_ID = new Map(DEVICES.map((d) => [d.id, d]));
const MODELS_BY_ID = new Map(MODELS.map((m) => [m.id, m]));

export function getDevice(id: string): CatalogDevice {
  const device = DEVICES_BY_ID.get(canonicalDeviceId(id));
  if (!device) throw new Error(`Unknown device: ${id}`);
  return device;
}

export function getModel(id: string): CatalogModel {
  const model = MODELS_BY_ID.get(id);
  if (!model) throw new Error(`Unknown model: ${id}`);
  return model;
}

/**
 * What the model lists are ordered by, in the reader's own words — rendered verbatim beside the
 * Bench's Model picker.
 *
 * **It lives here, three lines from the comparator, so that the sentence and the sort cannot
 * drift** (#179). A caption written in a component is a claim about code in another file with
 * nothing holding the two together: the order is decided in exactly one place, and this is that
 * place. The argument for saying it at all is `Recommend.tsx`'s, unchanged — an unstated tie-break
 * is the difference between a measurement and an opinion, and a list of 35 models in an order
 * nothing explains reads as arbitrary however carefully it was chosen.
 *
 * **No model is named, deliberately.** Between the committed catalog and the pending weekly
 * refresh — five days apart — 13 of 35 rows change rank and the second and third swap, so "led by
 * X" is a sentence that goes stale between releases while "most-downloaded first" stays true.
 *
 * The date is `generatedAt`, because the figure is a fetch rather than a feed: every run reads
 * `expand[]=downloads` for every seed, and a run whose figures move is a run that gets committed
 * (`scripts/catalog-diff.ts` counts popularity as substance), so the counts and the stamp are
 * always from the same fetch. The window is Hugging Face's own: `huggingface_hub` documents
 * `ModelInfo.downloads` as "Number of downloads of the model over the last 30 days", against
 * `downloadsAllTime` for the cumulative one.
 */
export const MODEL_ORDER_RULE = `Most-downloaded first, by Hugging Face downloads over the 30 days before the catalog was generated on ${new Date(CATALOG_GENERATED_AT).toISOString().slice(0, 10)} — a snapshot, not a live count.`;

/** Models a user is most likely to be looking for, most-downloaded first. See {@link MODEL_ORDER_RULE}. */
export function modelsByPopularity(): readonly CatalogModel[] {
  return [...MODELS].sort(
    (a, b) => (b.popularity?.downloads ?? 0) - (a.popularity?.downloads ?? 0)
  );
}

/**
 * What the comparison grid covers: every model against every _shipping_ device.
 *
 * Both halves were inline in `Matrix.tsx` and both are facts about the catalog rather than about
 * rendering, which is the reason to name them here. `status` is a catalog field and "a pre-release
 * spec must stay visibly labelled" is a catalog rule; leaving the filter in a component put it
 * somewhere `catalog.test.ts` could not reach, and the rumoured row would have arrived in the
 * shortlist the day someone tidied that `useMemo`. Row order is `devices.json`'s own — nothing
 * sorts it (#79) — and column order is popularity, from the one helper that defines it.
 *
 * **It is also the grid's one seam, and that is the second reason it is a function**
 * ([#101](https://github.com/MrZoller/headroom/issues/101)). The Matrix is models × devices and
 * `App.test.tsx` renders the whole page per test, so both catalog axes are load-bearing inputs to
 * the unit suite's wall clock: #78 and #77 together took the grid from 408 cells to 1,470 and that
 * file from 42s to about fourteen minutes on CI, and two pull requests that touched no component
 * failed on a per-test timeout for it. One function is what lets the integration suite bound the
 * grid it renders while the tests that are genuinely _about_ the grid keep the real one. Narrowing
 * it is a test's decision and never a caller's: there is no parameter here, because a grid extent
 * the app could pass is a grid extent the app could get wrong.
 */
export function comparisonGrid(): {
  models: readonly CatalogModel[];
  devices: readonly CatalogDevice[];
} {
  return {
    models: modelsByPopularity(),
    // Shipping hardware only: a rumoured row would put speculative specs into a comparison people
    // read as a shortlist, and `status` exists precisely so that never happens silently.
    devices: DEVICES.filter((d) => d.status === 'shipping'),
  };
}
