import type { DeviceSpec, ModelSpec, QuantSpec, Rig, RuntimeSpec, UsageSpec } from './types';
import { activationBytes } from './activations';
import { kvBytesTotal, layerKvBytes } from './kv';
import { weightBytes } from './weights';

/**
 * Where the bytes actually land.
 *
 * Two things here are routinely wrong elsewhere:
 *
 *   1. **Allocatable is not capacity.** A 128 GB Mac cannot hand 128 GB to a model — macOS
 *      caps GPU-wired memory near 75% of RAM by default. Strix Halo exposes 96 of its 128 GB.
 *      vLLM reserves a fixed fraction up front. Sizing against the number on the box is how
 *      you get a configuration that reports "fits" and then OOMs on load.
 *   2. **Offload is a cliff, not a slope.** When weights spill to host RAM over PCIe, the
 *      spilled fraction reads at a small fraction of device bandwidth. It is the most common
 *      reason a working setup is inexplicably slow, and it should be visible in the output
 *      rather than buried in an efficiency constant.
 */

/** Typical dual-channel DDR5 desktop bandwidth — the speed host RAM itself reads at. */
export const DEFAULT_HOST_BANDWIDTH = 80e9;

/**
 * The rate spilled weights actually stream at: the slower of host memory and the link to it.
 *
 * Host RAM bandwidth alone was the whole model, so every offloaded configuration read at 80 GB/s
 * regardless of what the card is plugged into. An RTX 4090 is PCIe 4.0 x16 — 31.5 GB/s one
 * direction — so a heavily offloaded DeepSeek V3 had both decode and TTFT understated by 2.5x,
 * well outside the band this engine claims.
 *
 * Derived here from the rig rather than passed in, because the previous shape let a caller omit
 * it and silently get the optimistic default — which is exactly what the whole app did.
 */
export function offloadBandwidth(rig: Rig, hostBandwidth: number, runtime?: RuntimeSpec): number {
  const link = rig.device.hostLinkBytesPerSec;
  if (link === undefined) return hostBandwidth;

  // Links add only where the devices work at the same time. Under tensor parallelism every card
  // streams its own shard concurrently, so the rig's links sum — up to host memory itself. Under
  // a layer split the cards run one after another, so a single token's transfer is limited by
  // whichever card is currently working: one link, however many are installed. Aggregating there
  // was the same mistake as aggregating its bandwidth, one function over.
  const links = runtime?.parallelism === 'layer' ? 1 : effectiveDeviceCount(rig);
  return Math.min(hostBandwidth, link * links);
}

export interface Placement {
  fits: boolean;

  /** Per-device figures, after tensor-parallel sharding across the rig. */
  weightBytesPerDevice: number;
  kvBytesPerDevice: number;
  activationBytesPerDevice: number;
  usedBytesPerDevice: number;
  allocatableBytesPerDevice: number;

  /** Totals across the whole rig, for the headline readout. */
  totalWeightBytes: number;
  totalKvBytes: number;

  /** Spare room per device; negative when over. */
  headroomBytes: number;
  utilization: number;

  /**
   * The most any one device must hold that offload cannot move — its cache plus its activations.
   *
   * This is the quantity `impossible` tests, exposed so the panels explaining an impossible
   * placement quote the device that caused it. Under a layer split that is not always the device
   * the rest of this readout describes: `weightBytesPerDevice` and `kvBytesPerDevice` come from the
   * busiest card by *combined* load, and on a hybrid model the card holding the most cache is the
   * one a balanced scheduler gives fewer layers. Gemma 3 12B on three 4090s at 128K and 8 users is
   * impossible because two cards need 24.6 GiB of cache *and activations* against a 23 GiB ceiling,
   * while the card this readout describes needs 19.1 — so a sentence built from `kvBytesPerDevice`
   * printed "the cache and overhead alone need 19.1 GiB" beside a 23.0 GiB ceiling and disproved
   * itself.
   */
  floorBytesPerDevice: number;

  /**
   * Fraction of the model's weights that must live in host RAM. Zero when everything fits. Only
   * meaningful for discrete GPUs — a unified-memory machine has no faster tier to fall
   * back from, so an over-budget config there simply does not run.
   *
   * **Rig-wide, not per-device**, and the distinction only shows up under a layer split, where the
   * devices hold different amounts and one can be over its ceiling while the others are not. Both
   * speed estimators multiply this by the whole model's active weights, so a per-device figure
   * charged every serial stage for a single stage's overflow.
   */
  offloadFraction: number;
  /** True when the configuration is over budget and offload cannot rescue it. */
  impossible: boolean;

  /** Set when the runtime cannot drive this class of device at all. */
  unsupported?: string;
}

/**
 * Whether a model can be sharded across several of these at all.
 *
 * Keyed on having a transport, not on device class. A DGX Spark is `unified-soc` and has a real
 * ConnectX link that `tpEfficiency` already models; a Mac Studio is the same class with nothing
 * between chassis. Exported so the UI and the store cannot hold divergent copies of this rule —
 * they did, and the slider offered counts the store immediately reset.
 *
 * Deliberately distinct from offload, which really is a discrete-GPU property: spilling needs a
 * slower *tier*, sharding needs a *link*, and the Spark has one without the other.
 */
export function canShard(device: DeviceSpec): boolean {
  return device.interconnect !== undefined;
}

/**
 * Devices that actually cooperate on one request — one, unless there is a link between them.
 *
 * Every divisor and every multiplier keyed off `rig.count` directly, so eight Mac Studios divided
 * a model eight ways and summed eight cards' bandwidth over an interconnect the catalog says they
 * do not have. `planPlacement` now refuses that rig outright, but a refusal that still returns
 * arithmetic for the impossible split is half a fix: the figures have to describe the machine that
 * exists, which is one of them.
 *
 * Shared by placement and speed so the memory panel and the throughput panel cannot disagree about
 * how many devices are in play — the failure mode this file has hit repeatedly.
 */
export function effectiveDeviceCount(rig: Rig): number {
  return canShard(rig.device) ? Math.max(1, rig.count) : 1;
}

/** Why this format cannot run on this device, or undefined when it can. */
function unmetRequirement(quant: QuantSpec, device: DeviceSpec): string | undefined {
  const { vendor, dtype } = quant.requires ?? {};
  if (vendor !== undefined && device.vendor !== vendor) {
    return `${quant.label} needs ${vendor} hardware.`;
  }
  if (dtype !== undefined && device.flops[dtype] === undefined) {
    return `${quant.label} needs ${dtype.toUpperCase()} tensor cores, which ${device.name} does not have.`;
  }
  return undefined;
}

/**
 * The most memory this device could ever hand the model, after any tuning its platform allows.
 *
 * `allocatableBytes` is the *default*; this is the ceiling on raising it. Apple's
 * `iogpu.wired_limit_mb` goes as far as physical memory, so those are capped by capacity — but
 * AMD's Variable Graphics Memory tops out at 96 of the Ryzen AI Max+'s 128 GB, which is already
 * its default. Treating physical capacity as everyone's maximum told a Ryzen owner that a
 * 117 GiB configuration would fit once they raised a setting the platform will not raise.
 *
 * Shared so the "you could raise this" claim is made in one place. The Bench and the Envelope
 * each had their own version of it, which is how one of them came to be wrong.
 */
export function maxAllocatablePerDevice(device: DeviceSpec): number {
  if (device.allocatableTunable !== true) return device.allocatableBytes;
  return Math.min(device.maxAllocatableBytes ?? device.capacityBytes, device.capacityBytes);
}

/** Whether raising the ceiling would actually buy this configuration anything. */
export function raisingCeilingWouldHelp(device: DeviceSpec, usedBytesPerDevice: number): boolean {
  return (
    device.allocatableTunable === true &&
    maxAllocatablePerDevice(device) > device.allocatableBytes &&
    usedBytesPerDevice <= maxAllocatablePerDevice(device)
  );
}

/** Memory a single device can actually give the model, after the runtime takes its cut. */
export function allocatablePerDevice(rig: Rig, runtime: RuntimeSpec): number {
  const { device } = rig;
  const ceiling =
    runtime.preallocFraction === undefined
      ? device.allocatableBytes
      : Math.min(device.allocatableBytes, device.capacityBytes * runtime.preallocFraction);
  return Math.max(0, ceiling);
}

/**
 * Clamp a scenario to values every module agrees on.
 *
 * Without this the modules disagree at the boundaries: `kvBytesTotal` would drop the entire
 * KV term at concurrency 0 and report that anything fits, while `estimateDecode` clamps to a
 * single sequence and reports a real throughput for it. `maxContextThatFits` inherits the
 * former and would claim full context on hardware that cannot hold it.
 */
/**
 * `Math.max(1, Math.floor(NaN))` is NaN, so a plain clamp would pass garbage straight through
 * and paint NaN across every field of the result. Scenarios arrive from sliders and from
 * hand-editable querystrings, where `Number(params.get('ctx'))` on nonsense yields NaN.
 */
function positiveInt(value: number, fallback = 1): number {
  return Number.isFinite(value) ? Math.max(1, Math.floor(value)) : fallback;
}

/**
 * The same guard for a count where zero is a real answer rather than a degenerate one.
 *
 * `positiveInt` floors at 1 because a scenario with no context or no devices is nonsense. An empty
 * cached prefix is not nonsense — it is the standalone-prompt case, and every archetype but one.
 */
function nonNegativeInt(value: number, fallback = 0): number {
  return Number.isFinite(value) ? Math.max(0, Math.floor(value)) : fallback;
}

export function normalizeUsage(usage: UsageSpec): UsageSpec {
  return {
    ...usage,
    contextTokens: positiveInt(usage.contextTokens),
    concurrency: positiveInt(usage.concurrency),
    ...(usage.promptTokens === undefined ? {} : { promptTokens: positiveInt(usage.promptTokens) }),
    ...(usage.cachedPrefixTokens === undefined
      ? {}
      : { cachedPrefixTokens: nonNegativeInt(usage.cachedPrefixTokens) }),
  };
}

export function normalizeRig(rig: Rig): Rig {
  return { ...rig, count: positiveInt(rig.count) };
}

/**
 * How many ways the KV cache actually divides across a tensor-parallel rig.
 *
 * Weights shard cleanly to any degree; KV does not, and assuming it does is optimistic in the
 * one direction that matters — it reports a configuration fitting when the real layout does not.
 *
 *   - **GQA** shards by attention head, so the degree is capped by the number of KV heads. A
 *     model with 4 KV heads on 8 cards gives every rank at least one whole head, and the cache
 *     is replicated across each pair: per-card KV is a quarter of the total, not an eighth.
 *     Qwen3 30B-A3B on 8x RTX 5080 at 32K and 16 users read 11.1 GiB against a 14.4 GiB ceiling
 *     and "fits"; the layout it would really produce needs 17.1 GiB.
 *   - **MLA** caches a single latent per token per layer with nothing to split along, so vLLM
 *     replicates it on every rank. The divisor is 1 however many cards there are — the case the
 *     old code was off by the full device count on.
 *
 * What each rank holds is `ceil(kvHeads / shards)` heads, so the effective divisor is
 * `kvHeads / ceil(kvHeads / shards)`. At 4 heads on 8 cards that is 4/1 = 4, and at 4 heads on
 * 3 cards it is 4/2 = 2 — replication is not always a clean fraction of the rig.
 *
 * All of the above describes **tensor parallelism**, which is vLLM's layout and not everyone's.
 * llama.cpp and Ollama split by *layer* by default: each card owns whole layers and their whole
 * KV buffers, so the cache divides by the device count with no head axis involved and no MLA
 * exception. Imposing the tensor-parallel cap on them rejected rigs that work — Qwen3 30B-A3B on
 * eight 5090s was charged 48 GiB of KV per card and told it would not run, where a layer split
 * needs about 24 and fits.
 *
 * Exported because `estimateDecode` has to price the cache the same way this sizes it. Holding
 * separate opinions is how the memory panel came to say every card holds the whole MLA latent
 * while the speed panel charged one eighth of it.
 */
/**
 * How many ways the *weights* divide.
 *
 * Tensor parallelism splits every tensor, so any degree works. A layer split hands out whole
 * layers, so the busiest card holds `ceil(layers / shards)` of them — the same ceiling `kvShards`
 * applies, because under a layer split the two quantities travel together and rounding only one
 * of them up describes a machine that does not exist.
 */
export function weightShards(model: ModelSpec, shards: number, runtime?: RuntimeSpec): number {
  if (shards <= 1) return 1;
  if (runtime?.parallelism !== 'layer') return shards;
  return model.layers / Math.ceil(model.layers / shards);
}

export function kvShards(model: ModelSpec, shards: number, runtime?: RuntimeSpec): number {
  if (shards <= 1) return 1;
  // A layer split replicates nothing — a card that holds layer 7 holds all of layer 7's cache
  // and none of anyone else's — but layers are whole and do not always divide evenly. A 36-layer
  // model on 8 cards puts 5 layers on some and 4 on others, so the busiest card holds a ninth
  // more than an even split would suggest. Same ceiling as the head axis below, on a different
  // quantity: assuming it divides cleanly is optimistic in the direction that reports a fit.
  if (runtime?.parallelism === 'layer') return weightShards(model, shards, runtime);

  const core = model.attention.core;
  if (core.kind === 'mla') return 1;
  const headsPerRank = Math.ceil(core.kvHeads / shards);
  return core.kvHeads / headsPerRank;
}

/** What one device of a rig ends up holding, weights and cache together. */
interface DeviceLoad {
  weightBytes: number;
  kvBytes: number;
}

const loadOf = (d: DeviceLoad) => d.weightBytes + d.kvBytes;

/**
 * What each device holds under a layer split, weights and cache together.
 *
 * One assignment, not two. Taking the heaviest-KV card from one packing and the heaviest-weight
 * card from another and adding them describes a device that does not exist: on a hybrid model the
 * card carrying the big full-attention caches is precisely the one a balanced scheduler gives
 * *fewer* layers, so the two maxima land on different cards. Summing them reported Gemma 3 27B on
 * four 5090s at 31.7 GiB and spilling, where a joint assignment fits it in 30.8 under a 31 GiB
 * ceiling — the first error tonight in the pessimistic direction, and wrong for the same reason
 * as all the optimistic ones: two numbers combined that were never measured on the same thing.
 *
 * Longest-processing-time again, now on the combined per-layer cost, so the balance being struck
 * is the one that actually matters to whether the rig fits.
 *
 * Every bin is returned rather than only the heaviest. The per-device readout needs the busiest and
 * nothing else, but the spill fraction is a property of the *rig*: under a layer split the devices
 * hold different amounts, so one card can be over its ceiling while the rest of the serial stages
 * stay resident. Returning a single load left `offloadFraction` a per-device number that both speed
 * estimators then charged against the whole model's active weights — every stage billed for host-bus
 * time on an overflow that happened at one of them.
 */
function layerSplitBins(
  model: ModelSpec,
  totalWeightBytes: number,
  usage: UsageSpec,
  shards: number,
  runtime: RuntimeSpec
): DeviceLoad[] {
  const perLayerWeight = totalWeightBytes / model.layers;
  const sequences = Math.max(1, usage.concurrency);

  const layers = Array.from({ length: model.layers }, (_, i) => ({
    weightBytes: perLayerWeight,
    kvBytes: layerKvBytes(model, i, usage.contextTokens, usage.kvPrecision, runtime) * sequences,
  })).sort((a, b) => loadOf(b) - loadOf(a));

  // Capped at the layer count, because a card with no layers on it holds nothing — and the device
  // count arrives from a hand-editable querystring, where `?n=99999999` would otherwise allocate a
  // bin per phantom card. The cap is not merely a guard: `weightShards` already divides by
  // `layers / ceil(layers / shards)`, which saturates at `layers` for the same reason, so capping
  // is what keeps the packing and the divisor describing one machine.
  const devices = Math.max(1, Math.min(Math.floor(shards), model.layers));
  const bins: DeviceLoad[] = Array.from({ length: devices }, () => ({
    weightBytes: 0,
    kvBytes: 0,
  }));

  for (const layer of layers) {
    let lightest = 0;
    for (let d = 1; d < bins.length; d++) {
      if (loadOf(bins[d]) < loadOf(bins[lightest])) lightest = d;
    }
    bins[lightest].weightBytes += layer.weightBytes;
    bins[lightest].kvBytes += layer.kvBytes;
  }

  return bins;
}

export function planPlacement(
  model: ModelSpec,
  quant: QuantSpec,
  rawUsage: UsageSpec,
  rawRig: Rig,
  runtime: RuntimeSpec
): Placement {
  const usage = normalizeUsage(rawUsage);
  const rig = normalizeRig(rawRig);

  const totalWeightBytes = weightBytes(model, quant);
  const totalKvBytes = kvBytesTotal(
    model,
    usage.contextTokens,
    usage.concurrency,
    usage.kvPrecision,
    // llama.cpp's q8_0/q4_0 KV carries a block scale, so the cache costs more than its nominal
    // width. Passed rather than assumed, since it is exact for a float format.
    runtime
  );
  const activations = activationBytes(model, usage, runtime);

  // Activations are per-device rather than shared. Weights and KV each get a divisor, and under
  // a layer split it is the *same* divisor: a card that owns a layer owns both its parameters
  // and its cache, so an indivisible layer count rounds them up together. Dividing weights
  // evenly while rounding KV up was the half-fix — 61 DeepSeek layers over two B200s is 31/30,
  // and the even split reported 175.4 GiB under a 178 GiB ceiling for a card really holding
  // 178.3.
  const shards = effectiveDeviceCount(rig);
  /**
   * Layer splits are packed rather than divided, because layers are not interchangeable: a
   * full-attention layer of a hybrid model holds up to 128x the cache a sliding one does, so the
   * layer *count* says nothing useful about what any card ends up holding. Weights and cache come
   * from one assignment so that both figures describe the same device.
   *
   * Tensor parallelism and the single-device case hand every rank the same load, so those are one
   * entry standing for `binsPerEntry` devices rather than `n` copies of one object. The spill
   * fraction below has to sum over the devices that actually exist, and under a layer split those
   * hold different amounts — but materialising a bin per rank would make the allocation, and the
   * `Math.max` over it, proportional to a device count that arrives from a hand-editable
   * querystring.
   */
  const layerSplit = runtime.parallelism === 'layer' && shards > 1;
  const bins: DeviceLoad[] = layerSplit
    ? layerSplitBins(model, totalWeightBytes, usage, shards, runtime)
    : [
        {
          weightBytes: totalWeightBytes / weightShards(model, shards, runtime),
          kvBytes: totalKvBytes / kvShards(model, shards, runtime),
        },
      ];
  /** How many real devices each entry of `bins` describes. */
  const binsPerEntry = layerSplit ? 1 : shards;

  const busiest = bins.reduce((a, b) => (loadOf(b) > loadOf(a) ? b : a));
  const weightBytesPerDevice = busiest.weightBytes;
  const kvBytesPerDevice = busiest.kvBytes;
  const usedBytesPerDevice = weightBytesPerDevice + kvBytesPerDevice + activations;

  const allocatableBytesPerDevice = allocatablePerDevice(rig, runtime);
  const headroomBytes = allocatableBytesPerDevice - usedBytesPerDevice;
  const fits = headroomBytes >= 0;

  // Only a discrete GPU has somewhere slower to spill to. On unified memory or CPU RAM the
  // pool in question *is* system memory, so over budget means it does not run.
  const canOffload = rig.device.class === 'discrete-gpu';

  /**
   * The bytes that actually leave the devices, summed over the rig — not one device's overflow
   * expressed as a fraction of one device's weights.
   *
   * Both speed estimators multiply `offloadFraction` by the *whole model's* active weights, so the
   * figure they need is rig-wide. Taking it from the busiest device charged every serial stage of a
   * layer split for an overflow at one of them: an uneven split where only the heaviest card is over
   * its ceiling had its resident stages billed host-bus time all the same, overstating decode and
   * TTFT together.
   *
   * A device can only spill what it holds, hence the per-bin clamp: past that point the KV and the
   * activations are what is over budget, and no amount of weight movement rescues it — which is the
   * `impossible` test below rather than a bigger fraction here.
   *
   * The uniform case is unchanged by construction. Every rank holds the same load, so summing `n`
   * identical overflows over `n` identical shards gives back exactly the per-device ratio this used
   * to compute.
   */
  const spilledBytes = canOffload
    ? binsPerEntry *
      bins.reduce((sum, bin) => {
        const over = loadOf(bin) + activations - allocatableBytesPerDevice;
        return sum + Math.min(Math.max(0, over), bin.weightBytes);
      }, 0)
    : 0;
  const offloadFraction = Math.min(1, spilledBytes / Math.max(totalWeightBytes, 1));

  // Even offloading every weight leaves KV and activations, which must sit on the device. Taken
  // over every device rather than the busiest one: `busiest` is heaviest by *combined* load, and on
  // a hybrid model the card holding the most cache is the one given fewer layers — so a rig can be
  // impossible at a device that is not the one this readout describes. Carried on the result rather
  // than recomputed by the callers, so the panels explaining the refusal quote the device that
  // caused it; a sentence built from `kvBytesPerDevice` instead contradicted its own ceiling.
  // `reduce`, not `Math.max(...spread)`: the spread throws `RangeError` past ~65k arguments, and a
  // bin list is only bounded by the model's layer count.
  const floorBytesPerDevice = bins.reduce(
    (max, bin) => Math.max(max, bin.kvBytes + activations),
    0
  );
  const impossible = !fits && (!canOffload || floorBytesPerDevice > allocatableBytesPerDevice);

  const drives = runtime.supports.some(
    (s) =>
      s.class === rig.device.class && (s.vendor === undefined || s.vendor === rig.device.vendor)
  );

  const unsupported = !drives
    ? `${runtime.label} does not run on ${rig.device.name}.`
    : // Sharding needs a link, and `canShard` is the one place that knows which devices have one.
      // Without this the split above ran anyway: eight Mac Studios came back as a supported
      // placement holding an eighth of the model each, over an interconnect that does not exist.
      // The Bench hides its device-count control for these rows, but that is the store protecting
      // one surface — the Matrix, the Envelope and any direct `evaluate` caller went straight past
      // it, which is the same UI-enforced-rule gap the weight-format check below was added for.
      rig.count > 1 && !canShard(rig.device)
      ? `${rig.device.name} has no interconnect, so a model cannot be split across ${rig.count} of them.`
      : !runtime.kvPrecisions.includes(usage.kvPrecision)
        ? `${runtime.label} cannot store a ${usage.kvPrecision.toUpperCase()} KV cache.`
        : // The same guarantee `kvPrecisions` gives the cache, for the weights. llama.cpp loads
          // GGUF and not AWQ; vLLM reads neither GGUF K-quant. `quantApplies` enforced this for
          // the picker and the store, but every caller that reaches the engine directly — the
          // Matrix, the Envelope, anything importing `evaluate` — bypassed it and got capacity and
          // throughput figures for a checkpoint the runtime cannot open. A rule that only the UI
          // applies is not a rule the engine has.
          !runtime.weightFormats.includes(quant.id)
          ? `${runtime.label} cannot load ${quant.label} weights.`
          : // A format tied to particular silicon is as unrunnable as a runtime that cannot drive
            // the device, and was previously waved through — leaving `peakFlops` to read a rate
            // published for a different format, or for hardware that has no such units at all.
            unmetRequirement(quant, rig.device);

  return {
    fits,
    weightBytesPerDevice,
    kvBytesPerDevice,
    activationBytesPerDevice: activations,
    usedBytesPerDevice,
    allocatableBytesPerDevice,
    totalWeightBytes,
    totalKvBytes,
    headroomBytes,
    utilization: usedBytesPerDevice / allocatableBytesPerDevice,
    floorBytesPerDevice,
    offloadFraction,
    impossible,
    unsupported,
  };
}

/**
 * Largest context that still fits, at the given concurrency. Binary search rather than an
 * inverted formula, because hybrid sliding-window models make KV a non-linear function of
 * context — the closed form would be wrong for exactly the models it matters most for.
 */
export interface ContextLimitOptions {
  /**
   * Count a context as reachable when the weights spill to host RAM, rather than requiring a
   * fully resident placement.
   *
   * The distinction is not cosmetic. `fits` is false for *any* offloaded configuration, even at
   * a one-token context, so the resident limit collapses to zero the moment a model is too big
   * for the card — reporting "caps out at 0" for a rig that would happily hold 128K of KV once
   * the weights are offloaded. Ask for the resident figure when saying what fits comfortably,
   * and the offload-aware one when saying what can actually be run.
   */
  allowOffload?: boolean;
}

export function maxContextThatFits(
  model: ModelSpec,
  quant: QuantSpec,
  usage: UsageSpec,
  rig: Rig,
  runtime: RuntimeSpec,
  { allowOffload = false }: ContextLimitOptions = {}
): number {
  const attempt = (contextTokens: number) => {
    const placement = planPlacement(model, quant, { ...usage, contextTokens }, rig, runtime);
    if (placement.unsupported) return false;
    return allowOffload ? !placement.impossible : placement.fits;
  };

  if (!attempt(1)) return 0;

  let low = 1;
  let high = model.maxContext;
  if (attempt(high)) return high;

  while (high - low > 1) {
    const mid = Math.floor((low + high) / 2);
    if (attempt(mid)) low = mid;
    else high = mid;
  }
  return low;
}
