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
 * A model plus the note explaining any hand-entered correction the generator applied.
 *
 * Mirrors {@link CatalogDevice}: the seed list requires a reason whenever a figure is typed by
 * a human rather than derived, and that reason has to be reachable from the UI or the
 * requirement is decorative. Three catalogued models carry an overridden `totalParams`.
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
  const device = DEVICES_BY_ID.get(id);
  if (!device) throw new Error(`Unknown device: ${id}`);
  return device;
}

export function getModel(id: string): CatalogModel {
  const model = MODELS_BY_ID.get(id);
  if (!model) throw new Error(`Unknown model: ${id}`);
  return model;
}

/** Models a user is most likely to be looking for, most-downloaded first. */
export function modelsByPopularity(): readonly CatalogModel[] {
  return [...MODELS].sort(
    (a, b) => (b.popularity?.downloads ?? 0) - (a.popularity?.downloads ?? 0)
  );
}
