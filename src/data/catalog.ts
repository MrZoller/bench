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

interface DeviceRow {
  id: string;
  name: string;
  vendor: string;
  class: string;
  status: string;
  capacityGiB: number;
  allocatableGiB: number;
  allocatableTunable?: boolean;
  bandwidthGBs: number;
  measuredBandwidthGBs?: number;
  // Rows list only the dtypes their datasheet publishes, so TypeScript widens the absent keys
  // to `undefined` across the union of row shapes. Modelled honestly rather than cast away.
  tflops: Record<string, number | undefined>;
  interconnect?: string;
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

function toDevice(row: DeviceRow): CatalogDevice {
  const flops: DeviceSpec['flops'] = {};
  for (const [dtype, value] of Object.entries(row.tflops)) {
    if (value === undefined) continue;
    flops[dtype as keyof DeviceSpec['flops']] = value * TFLOP;
  }

  return {
    id: row.id,
    name: row.name,
    vendor: row.vendor,
    class: row.class as DeviceSpec['class'],
    status: row.status as DeviceSpec['status'],
    capacityBytes: row.capacityGiB * GIB,
    allocatableBytes: row.allocatableGiB * GIB,
    ...(row.allocatableTunable ? { allocatableTunable: true } : {}),
    bandwidthBytesPerSec: row.bandwidthGBs * GB,
    ...(row.measuredBandwidthGBs
      ? { measuredBandwidthBytesPerSec: row.measuredBandwidthGBs * GB }
      : {}),
    flops,
    ...(row.interconnect ? { interconnect: row.interconnect } : {}),
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
