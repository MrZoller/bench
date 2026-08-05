import type { DeviceSpec, ModelSpec, QuantSpec, Rig, RuntimeSpec, UsageSpec } from './types';
import { activationBytes } from './activations';
import { kvBytesTotal, layerKvBytes } from './kv';
import { weightBreakdown, type WeightBreakdown } from './weights';

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

  /** How the model was actually laid out over the rig. See {@link Assignment}. */
  assignment: Assignment;

  /** Set when the runtime cannot drive this class of device at all. */
  unsupported?: string;
}

/**
 * One device's share of the model — or several devices', where they all hold the same thing.
 *
 * `layers` and `weightBytes` answer different questions and only `Assignment.parallelism` says
 * which: under a layer split a device owns `layers` whole layers and `weightBytes` is their total
 * weight, while under tensor parallelism every device holds a *slice* of every layer, so `layers`
 * is the model's own count and `weightBytes` is the slice. Reading one as the other is how a
 * launch command comes to name a split that does not exist.
 */
export interface DeviceShare {
  /** How many of the rig's devices carry this exact load. */
  deviceCount: number;
  layers: number;
  /**
   * *Which* layers, by the model's own index, ascending — not merely how many (#166).
   *
   * `layers` is `layerIndices.length` and stays because it is the quantity a flag reads; this is
   * the quantity a *description* needs. The two are not interchangeable, and the gap between them
   * is the whole finding: `layerSplitBins` packs individual layers by combined weight-plus-cache
   * load, so a bin is a non-contiguous mixture and **many different assignments share one set of
   * counts while holding radically different amounts**. On Gemma 3 12B at 128K over 8 users a card
   * that lands one full-attention layer is full, and a card holding only sliding layers takes
   * twenty to reach the same load — the counts land 19 apart on five cards, and nothing in them
   * says why. A consumer reading `layers` alone cannot reproduce the layout that the `weightBytes`
   * and `kvBytes` beside it describe; reading these it can.
   *
   * Recorded during the packing, never recovered afterwards — the same rule the rest of this
   * interface follows, and for a sharper reason here: the packing walks the layers in *load* order,
   * so there is no arithmetic over `layers` and `deviceCount` that reconstructs the sets.
   *
   * **Under tensor parallelism this is every layer**, exactly as `layers` is, because a rank holds
   * a slice of each. `Assignment.parallelism` is still what says which reading applies: the same
   * list means "these layers, whole" on one branch and "a slice of each of these" on the other.
   */
  layerIndices: readonly number[];
  /**
   * Of `layers`, how many keep their weights on the device rather than streaming from host RAM.
   *
   * **Derived from this share's own bytes and this share's own layer count**, never from the
   * rig-wide `offloadFraction`. That distinction is the whole reason this field exists: under a
   * layer split the devices hold different amounts, so a rig-wide ratio applied to a per-device
   * layer count is the #14 defect one level up — and here it would be shipped as a number the
   * reader pastes into a shell.
   *
   * **And the bytes it is a share of are the *repeating* ones**, not `weightBytes` (#165). The
   * embedding table, the output projection and any vision tower are on the rig and are in no layer,
   * so a count taken against the whole share calls a fraction of them layers — which was the same
   * defect one level further down, with the operands swapped again.
   *
   * Floored, because the rounding has a safe direction: too few layers on the device is slow, and
   * one too many is an out-of-memory error on load.
   */
  residentLayers: number;
  weightBytes: number;
  kvBytes: number;
}

/**
 * The layout `planPlacement` packed, kept rather than discarded.
 *
 * The packing has always existed — `layerSplitBins` sizes a real assignment, because a hybrid
 * model's layers are not interchangeable — but only the busiest bin's byte totals survived onto
 * `Placement`, so nothing downstream could say *how many layers go where*. That is the one
 * question a launch command has to answer (#136), and recovering it from `offloadFraction`
 * afterwards is exactly the derivation this engine has already been wrong about once.
 *
 * **And a count was not the whole of what the packing knew** (#166): the first version kept how
 * many layers each device holds and discarded which, so the object described a *family* of
 * assignments rather than the one it had sized. `DeviceShare.layerIndices` closes that; nothing
 * else here moved, because nothing else was wrong.
 *
 * Recording, not modelling: every figure here comes from the same bins the byte totals do.
 */
export interface Assignment {
  /**
   * The runtime's layout, restated so a reader of this object can interpret `layers` without
   * fetching the runtime back.
   */
  parallelism: RuntimeSpec['parallelism'];
  /** One entry per distinct load, compressed exactly as the packing is — see `binsPerEntry`. */
  shares: readonly DeviceShare[];
  /**
   * Layers whose weights sit on a device, counted across the whole rig.
   *
   * Under a layer split the devices hold *different* layers, so the rig's total is the sum; under
   * tensor parallelism they all hold slices of the *same* layers, so the rig's count is one
   * device's and summing would multiply the model by the rig.
   *
   * **This is a layout, not yet a flag.** It is the quantity llama.cpp's `-ngl` is, and a caller
   * emitting that flag still owes it two gates. A `cpu-ram` rig has no GPU to offload *to*, so
   * every layer reports resident here — truthfully, since nothing spilled — while the honest
   * `-ngl` for that machine is 0. And llama.cpp counts the output tensor as one position past the
   * repeating blocks, so "all of them" is `layers + 1` rather than `layers`. Neither belongs in the
   * engine, which has no notion of a flag; both belong to whatever prints one.
   */
  residentLayers: number;
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
 * `allocatableBytes` is the *default*; this is the ceiling on raising it. AMD's Variable Graphics
 * Memory tops out at 96 of the Ryzen AI Max+'s 128 GB, which is already its default. Treating
 * physical capacity as everyone's maximum told a Ryzen owner that a 117 GiB configuration would
 * fit once they raised a setting the platform will not raise.
 *
 * **A device that states no maximum is treated as not raiseable at all**, rather than as raiseable
 * to physical capacity, which is what this used to assume. That assumption was wrong in the
 * direction that says "this will work": every Apple row was tunable with no stated maximum, so all
 * six resolved to 100% of RAM, and the app told the owner of a 96 GiB Mac Studio that a 95.5 GiB
 * configuration would fit once they raised the ceiling. `iogpu.wired_limit_mb` will *accept* that
 * value — it is bounded by what macOS needs to keep running, not by what the sysctl parses. The
 * rows now state their own ceilings and `catalog.ts` refuses a tunable row that does not, so the
 * absent case is a defensive floor rather than a live path.
 *
 * Shared so the "you could raise this" claim is made in one place. The Bench and the Envelope
 * each had their own version of it, which is how one of them came to be wrong.
 */
export function maxAllocatablePerDevice(device: DeviceSpec): number {
  if (device.allocatableTunable !== true) return device.allocatableBytes;
  return Math.min(device.maxAllocatableBytes ?? device.allocatableBytes, device.capacityBytes);
}

/** Whether raising the ceiling would actually buy this configuration anything. */
export function raisingCeilingWouldHelp(device: DeviceSpec, usedBytesPerDevice: number): boolean {
  return (
    device.allocatableTunable === true &&
    maxAllocatablePerDevice(device) > device.allocatableBytes &&
    usedBytesPerDevice <= maxAllocatablePerDevice(device)
  );
}

/**
 * Whether a placement was judged on its bytes, as against turned away on a categorical ground.
 *
 * `unsupported` collects the refusals that never consult the arithmetic — wrong device class, no
 * interconnect to shard over, a KV precision or weight format the runtime does not offer, an unmet
 * requirement — where `impossible` is nothing but arithmetic. Anything that answers "did these
 * numbers come from somewhere" has to split the two, and two surfaces now do: the Matrix's stand-in
 * legend and the Bench's stand-in banner.
 *
 * One function rather than the expression written twice, for the reason `substitutionFor`'s own
 * docblock gives about "is this a substitution" — the copies are fine until they disagree, and the
 * failure mode here is silent. Note it is *not* `fits`: a configuration that was measured and came
 * up short still got its verdict from the format it was scored at, and dropping the mark there is
 * the polarity error #32 fixed on the Matrix. Raised by Codex on PR #32.
 */
export function wasEvaluated(placement: Placement): boolean {
  return placement.unsupported === undefined;
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
 * The prompt a scenario is actually timed at, including the default when none is stated.
 *
 * Stated here rather than inline in `estimatePrefill` because it is no longer only prefill's
 * business: `clampUsageToContext` has to know how much of the window the prompt will occupy in
 * order to work out what is left for a cached prefix. A boundary written down twice is a boundary
 * that will disagree with itself — the mistake `cachedPrefix` in `verdict.ts` already carries a
 * comment about.
 */
export function effectivePromptTokens(usage: UsageSpec): number {
  return Math.max(1, usage.promptTokens ?? Math.floor(Math.max(1, usage.contextTokens) * 0.9));
}

/**
 * Narrow a scenario to one cell's own context.
 *
 * The Envelope and the Matrix both evaluate a grid of contexts against usage chosen for a single
 * point, so both have to answer the same question: what does this scenario mean in a column
 * narrower than the one it was written for. The answer is that a cell's figures must describe a
 * request that cell can actually hold, which is a property of the whole working set and not of any
 * one field.
 *
 * `promptTokens` is capped at the context because the prompt is *part* of the context — a 32K
 * prompt in a 2K column describes a request that cannot be made. `cachedPrefixTokens` is part of
 * it in exactly the same way, and is capped at whatever room the prompt leaves rather than at the
 * context itself: a prefix is tokens already resident when the prompt arrives, so a prompt and a
 * prefix that each fit alone but not together still overflow the window. This is the same
 * arithmetic `verdict.ts` does from the other side, deriving a prefix as the window minus what the
 * turn needs.
 *
 * The prefix clamp is the one that was missing, and it is the more expensive omission of the two:
 * decode ignores the prefix, but prefill charges every new token for attending over it, so an
 * unclamped prefix inflates time-to-first-token without bound — a 2K column was timed against a
 * 49K resident session, and a prefix past the model's own limit took one Matrix cell from 16 s to
 * 273 s.
 */
export function clampUsageToContext(usage: UsageSpec, contextTokens: number): UsageSpec {
  const promptTokens =
    usage.promptTokens === undefined ? undefined : Math.min(usage.promptTokens, contextTokens);
  // Against the effective prompt, not the stated one, so a scenario that leaves `promptTokens`
  // unset is held to the same bound as one that spells it out. The Envelope leaves it unset.
  const room = Math.max(
    0,
    contextTokens - effectivePromptTokens({ ...usage, contextTokens, promptTokens })
  );

  return {
    ...usage,
    contextTokens,
    ...(promptTokens === undefined ? {} : { promptTokens }),
    ...(usage.cachedPrefixTokens === undefined
      ? {}
      : { cachedPrefixTokens: Math.min(usage.cachedPrefixTokens, room) }),
  };
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
 * How many ways the *repeating stack* divides.
 *
 * Tensor parallelism splits every tensor, so any degree works. A layer split hands out whole
 * layers, so the busiest card holds `ceil(layers / shards)` of them — the same ceiling `kvShards`
 * applies, because under a layer split the two quantities travel together and rounding only one
 * of them up describes a machine that does not exist.
 *
 * **`planPlacement` reaches this only on the branch `layerSplitBins` does not cover** — tensor
 * parallelism, where every tensor really is sliced, and the single-device case, where the answer is
 * 1 either way. The layer arm survives because `kvShards` delegates to it, and a cache genuinely
 * does divide by whole layers. So the tensors that divide by nothing (#165) never reach it.
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
  /**
   * Of `weightBytes`, the part belonging to the repeating layers — so the remainder is this bin's
   * share of the embedding table, the output projection and any vision tower.
   *
   * The only basis a layer *count* may be taken against, and it is not `weightBytes`: the two
   * diverge by the fixed tensors' share of the file, which is 12.3% on Llama 3.2 3B and 25% on
   * Gemma 3 4B. Tracked during the packing rather than derived afterwards, for the same reason
   * `layers` is.
   */
  layerWeightBytes: number;
  /**
   * Whole layers in this bin.
   *
   * Counted during the packing rather than recovered afterwards as `weightBytes / perLayerWeight`.
   * The two agree today only because weights are modelled uniform per layer, while the packing
   * balances on the *combined* per-layer cost and therefore hands out layers by cache size — so
   * the derived version is a coincidence of the weight model, and it would go silently wrong the
   * day per-layer weights stop being uniform.
   */
  layers: number;
  /** Which layers, by the model's own index — see {@link DeviceShare.layerIndices}. */
  layerIndices: number[];
}

/**
 * Structural rather than `DeviceLoad`, because it is also asked of a single *layer* during the
 * packing, and a layer has no layer count of its own. The two byte fields are all it reads.
 */
const loadOf = (d: { weightBytes: number; kvBytes: number }) => d.weightBytes + d.kvBytes;

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
 *
 * **The weights are still divided by the layer count, and `layerWeightBytes` is what stops that
 * being read as a layer count** (#165). `weightBytes` here is `totalWeightBytes / layers` per layer,
 * which charges every bin a share of tensors no layer holds — the embedding table, the output
 * projection where it is untied, any vision tower. As a byte total that is right in aggregate and
 * approximate per bin; as a *count* it is neither, so the count is taken against the repeating
 * bytes that `weightBreakdown` separates out.
 *
 * **Assigning the fixed block to one bin instead was measured and deferred**, and the measurement is
 * the reason. Those tensors really do sit whole on one device, so seeding a bin with them is the
 * physically faithful packing — but it moves `usedBytesPerDevice` by more than 5% on a tenth of the
 * catalog's multi-card configurations, by up to 28%, and flips `fits` on 0.6% of them. That is a
 * change to what the product answers, not to a layer count, and it turns on which device llama.cpp
 * puts `token_embd` and a *tied* output tensor on — a question this file would have to read from
 * upstream rather than reason about. Filed rather than folded in.
 */
function layerSplitBins(
  model: ModelSpec,
  weights: WeightBreakdown,
  usage: UsageSpec,
  shards: number,
  runtime: RuntimeSpec
): DeviceLoad[] {
  const perLayerWeight = weights.totalBytes / model.layers;
  const perLayerRepeating = weights.layerBytes / model.layers;
  const sequences = Math.max(1, usage.concurrency);

  // The index travels with the load, because the sort below reorders them and the assignment is
  // recorded in the order the walk visits — a bin's membership is not recoverable from its count
  // afterwards, which is the whole of #166.
  const layers = Array.from({ length: model.layers }, (_, i) => ({
    index: i,
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
    layerWeightBytes: 0,
    kvBytes: 0,
    layers: 0,
    layerIndices: [],
  }));

  for (const layer of layers) {
    let lightest = 0;
    for (let d = 1; d < bins.length; d++) {
      if (loadOf(bins[d]) < loadOf(bins[lightest])) lightest = d;
    }
    bins[lightest].weightBytes += layer.weightBytes;
    bins[lightest].layerWeightBytes += perLayerRepeating;
    bins[lightest].kvBytes += layer.kvBytes;
    bins[lightest].layers += 1;
    bins[lightest].layerIndices.push(layer.index);
  }

  // The walk visits layers heaviest-first, so a bin collects its indices in load order. Sorted
  // back into the model's own order because that is the order every reader of the list means —
  // a card's share is "layers 5, 11 and 17", not "the three most expensive it was handed".
  for (const bin of bins) bin.layerIndices.sort((a, b) => a - b);

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

  const weights = weightBreakdown(model, quant);
  const totalWeightBytes = weights.totalBytes;
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
  // One divisor for the whole file and for the repeating part of it, because this branch is the one
  // case where every tensor really is sliced the same way — a tensor-parallel rank holds its
  // fraction of the embedding table exactly as it holds its fraction of a layer — and the
  // single-device case, where the divisor is 1 and there is nothing to say.
  const uniformShards = weightShards(model, shards, runtime);
  const bins: DeviceLoad[] = layerSplit
    ? layerSplitBins(model, weights, usage, shards, runtime)
    : [
        {
          weightBytes: totalWeightBytes / uniformShards,
          layerWeightBytes: weights.layerBytes / uniformShards,
          kvBytes: totalKvBytes / kvShards(model, shards, runtime),
          // Every layer, in both cases this branch covers, and for two different reasons: one
          // device holds all of them whole, and a tensor-parallel rank holds a slice of each. The
          // bytes beside it are what distinguishes the two, which is why `Assignment.parallelism`
          // has to travel with the shares.
          layers: model.layers,
          layerIndices: Array.from({ length: model.layers }, (_, i) => i),
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
  const spilledOf = (bin: DeviceLoad) =>
    canOffload
      ? Math.min(
          Math.max(0, loadOf(bin) + activations - allocatableBytesPerDevice),
          bin.weightBytes
        )
      : 0;
  const spilledBytes = binsPerEntry * bins.reduce((sum, bin) => sum + spilledOf(bin), 0);
  const offloadFraction = Math.min(1, spilledBytes / Math.max(totalWeightBytes, 1));

  /**
   * The same overflow read as a layer count, which is what a launch command needs.
   *
   * **Every quantity in the expression belongs to the same bin**, and that is the point rather
   * than an implementation detail: the spill is this device's own share of its own weights, and
   * multiplying it by this device's own layer count keeps the whole derivation inside one device.
   * Reaching for `offloadFraction` here — a rig-wide ratio — and applying it to a per-device layer
   * count is the #14 defect with the operands swapped, and it would ship as a number the reader
   * pastes into a shell rather than as a figure on a panel.
   *
   * **And the denominator is `layerWeightBytes`, not `weightBytes`** (#165). A bin's `weightBytes`
   * is `layers x totalWeightBytes / layers`, which is a layer plus a share of the embedding table,
   * the output projection and any vision tower — so the ratio reserves for those tensors in
   * proportion to how much of the stack stays resident rather than in full, and over-counts `-ngl`
   * on precisely the large-vocabulary models where they are worth counting. Llama 3.2 3B at Q4_K_M
   * against a 2 GiB ceiling reported 7 of 28 layers where 4 is what the card can hold. The two
   * readings agree exactly whenever nothing spills, which is most of the catalog.
   *
   * **The whole spill is charged against the layers, and that is llama.cpp's own eviction order
   * rather than a cautious approximation of it** (#202). An earlier version of this comment argued
   * the opposite — that llama.cpp sheds the output tensor before any layer, so this under-reports —
   * and it was backwards. `i_gpu_start = max(n_layer_all + 1 - ngl, 0)` shifts the resident window
   * off the **front** of the stack, and the output tensor is slot `n_layer_all`, where
   * `il - i_gpu_start = ngl - 1 < act_gpu_layers = ngl` for every `ngl >= 1`: the output tensor is
   * resident whenever anything is, and what an `-ngl` short of `layers + 1` sheds is layer 0. Read
   * 5 August 2026 from `src/llama-model.cpp:1318-1343` and `common/fit.cpp:581` at
   * ggml-org/llama.cpp commit `360e134`, and measured: an 18-layer model at `-ngl 1` has the output
   * table on the GPU and nothing else.
   *
   * **The count is unchanged, because the true rule is what the count already assumed.** A device
   * over budget keeps its fixed tensors and gives up repeating layers from the front, so every
   * overflowing byte really does come out of `layerWeightBytes`, and the generous reading is not the
   * other defensible option — it is a placement no `-ngl` expresses. It would let a spilling device
   * report every layer resident, which `launch.ts` turns into `layers + 1`, and any `-ngl` holding
   * all `layers` repeating blocks is at least `layers + 1` and therefore holds the output too. There
   * is no flag for "every layer, output evicted"; asking for one is an OOM on load.
   *
   * Floored for the same reason, and under the true rule the floor is exact rather than merely
   * safe: llama.cpp sheds *whole* layers, so the floored ratio is the count it lands on rather than
   * a cautious rounding-down of one.
   *
   * What the corrected rule *does* touch is which tensors sit on which device — the fixed block is
   * smeared across bins here, and upstream puts `token_embd` and the output projection at opposite
   * ends of the rig. That is bytes, not this count, and it belongs to #182.
   */
  const residentLayersOf = (bin: DeviceLoad) => {
    if (bin.layers <= 0 || bin.layerWeightBytes <= 0) return bin.layers;
    const resident = bin.layerWeightBytes - spilledOf(bin);
    return Math.max(
      0,
      Math.min(bin.layers, Math.floor((resident / bin.layerWeightBytes) * bin.layers))
    );
  };

  const shares: DeviceShare[] = bins.map((bin) => ({
    deviceCount: binsPerEntry,
    layers: bin.layers,
    layerIndices: bin.layerIndices,
    residentLayers: residentLayersOf(bin),
    weightBytes: bin.weightBytes,
    kvBytes: bin.kvBytes,
  }));

  const assignment: Assignment = {
    parallelism: runtime.parallelism,
    shares,
    // A sum under a layer split, because the cards hold different layers; one card's count under
    // tensor parallelism, because they hold slices of the same ones. Summing there would report a
    // rig holding several times the model it is running.
    residentLayers:
      runtime.parallelism === 'layer'
        ? shares.reduce((sum, s) => sum + s.deviceCount * s.residentLayers, 0)
        : shares.reduce((min, s) => Math.min(min, s.residentLayers), model.layers),
  };

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
    assignment,
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
