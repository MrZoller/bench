import type { DeviceSpec } from '@/engine/types';
import { GIB } from '@/engine/types';

/**
 * One-click "what can my machine run" (#137).
 *
 * The current entry point assumes the visitor knows their GPU or SoC and can find it in a 43-row
 * picker. Most people who would benefit from bench know "a MacBook" or "a gaming PC". This reads
 * what the browser exposes and narrows the picker — **to a candidate shortlist and a confirmation
 * step, never to a silent selection.**
 *
 * ## The contract: a guess is visibly a guess
 *
 * The same polarity rule the substituted-format marker (#18) established, applied to the other end
 * of the pipeline. A wrong guess silently applied is invented data wearing the chassis of a
 * measurement; a shortlist with "which of these is yours?" is honest and still removes almost all
 * of the picker's friction.
 *
 * **The shortlist's size is conditional on what the browser admits, and no size is promised.** On a
 * candid browser it is a handful; on a redacting one, vendor alone maps to 17 shipping NVIDIA rows
 * or 10 Apple ones. A follow-up question is a first-class path here, not a failure branch.
 *
 * ## A filter that would leave nothing is a filter that is wrong about this machine
 *
 * The rule that keeps the shortlist non-empty, and it is not a guard — it is the issue's "fall back
 * when the signals conflict", made general. Signals really do contradict each other on real
 * hardware: an **Intel Mac** reports an Intel or AMD adapter on a macOS platform, and the platform
 * prune below would take the vendor's four rows to none. An iPhone with WebGPU does the same from
 * the other side.
 *
 * So every prune is applied only if something survives it, and a prune that would empty the list is
 * recorded as a conflict instead. That keeps the reader's real machine in the list wherever it is in
 * the catalog at all, and it means `candidates` is never empty — which is what stops the surface
 * rendering "which of these is yours?" over nothing.
 *
 * ## What the signals are actually worth, which is less than it looks
 *
 * Every mapping below is derived from a cited source rather than recalled, and three of them turned
 * out to be weaker than the issue assumed:
 *
 *   - **`GPUAdapterInfo.architecture` does not identify a Mac.** Apple GPUs report no DeviceID
 *     through Metal, so Dawn reports the highest supported *common feature family* instead —
 *     `common-1`, `common-2`, `common-3`. Every Apple silicon Mac from the M1 to the newest reports
 *     the same string, so on the one platform where a unified-memory row is the headline case, the
 *     architecture narrows nothing at all. It is `deviceMemory` and a question that do the work.
 *   - **`navigator.deviceMemory` is capped at 8 in Chrome and absent in Safari**, so it separates
 *     small machines from large ones and nothing above 8 GiB from anything else. Read as a *floor*
 *     here, never as a capacity.
 *   - **The adapter limits are allocation ceilings, not memory.** `maxBufferSize` is the largest
 *     single buffer, which a driver caps well below VRAM — so it is a lower bound on capacity and
 *     nothing more. That is still real narrowing: a card reporting a 4 GiB maximum buffer cannot be
 *     an 8 GiB card's smaller sibling.
 *
 * ## Everything degrades to the picker, quietly
 *
 * `navigator.gpu` is undefined in Safari behind a flag and in any hardened browser, and detection
 * is a fingerprinting surface browsers keep sanding down. Nothing here throws, nothing logs, and an
 * empty result means "use the picker" rather than "something went wrong" — the preflight-not-red
 * pattern `deploy.yml` already follows.
 */

/**
 * Architecture strings, as Chrome actually reports them.
 *
 * **Generated from Dawn's own data file rather than transcribed**, and the transformation matters:
 * `gpu_info.json` stores names like `RDNA 3` and `Gen 12 LP`, and `dawn_gpu_info_generator.py`'s
 * `js_enum_case()` lowercases and joins with a hyphen — *except* after a digit, where it joins with
 * nothing. So `RDNA 3` is `rdna-3` and `Gen 12 LP` is `gen-12lp`, not `gen-12-lp`. A hand-written
 * table would have got the Intel rows wrong, and a web search returned both `rdna-3` and `rdna4`
 * for the same field.
 *
 * This maps architecture to **vendor only**, which is all it can honestly do. Dawn's architecture is
 * a silicon generation and `devices.json` has no generation column — adding one would mean asserting
 * a generation per row from memory, which is the one thing this project does not do. What the
 * generation buys is a *sentence* for the confirmation step ("your browser reports a Blackwell
 * NVIDIA GPU"), and corroboration for a vendor string that browsers sometimes redact.
 *
 * Source: https://dawn.googlesource.com/dawn/+/refs/heads/main/src/dawn/gpu_info.json and
 * generator/dawn_gpu_info_generator.py (`Name.js_enum_case`), read 2026-08-01.
 */
export const ARCHITECTURE_VENDORS: Readonly<Record<string, DeviceSpec['vendor']>> = {
  // NVIDIA
  ampere: 'NVIDIA',
  blackwell: 'NVIDIA',
  fermi: 'NVIDIA',
  kepler: 'NVIDIA',
  lovelace: 'NVIDIA',
  maxwell: 'NVIDIA',
  pascal: 'NVIDIA',
  turing: 'NVIDIA',
  volta: 'NVIDIA',
  // AMD
  'cdna-1': 'AMD',
  'gcn-1': 'AMD',
  'gcn-2': 'AMD',
  'gcn-3': 'AMD',
  'gcn-4': 'AMD',
  'gcn-5': 'AMD',
  'rdna-1': 'AMD',
  'rdna-2': 'AMD',
  'rdna-3': 'AMD',
  'rdna-4': 'AMD',
  'terascale-2': 'AMD',
  // Intel
  'gen-7': 'Intel',
  'gen-8': 'Intel',
  'gen-9': 'Intel',
  'gen-11': 'Intel',
  'gen-12hp': 'Intel',
  'gen-12lp': 'Intel',
  'xe-lpg': 'Intel',
  'xe-2hpg': 'Intel',
  'xe-2lpg': 'Intel',
  'xe-3lpg': 'Intel',
  'xe-3lpg-xs': 'Intel',
};

/**
 * Apple's architecture strings, which are feature families rather than chips.
 *
 * Listed separately and deliberately *not* in the table above, because they behave differently in
 * kind: the entries above identify a vendor and a generation, and these identify only that the
 * machine is an Apple GPU. `gpu_info.json`'s own comment on the Apple vendor says why — Apple GPUs
 * report no DeviceID through the Metal API, so the recommended approach is to report the highest
 * supported common family. Every Apple silicon Mac reports one of these.
 */
export const APPLE_FAMILIES: readonly string[] = ['common-1', 'common-2', 'common-3'];

/** What a browser was willing to say. Every field optional, because every field can be withheld. */
export interface DetectionSignals {
  /** `GPUAdapterInfo.vendor`, lowercase per the WebGPU spec. */
  adapterVendor?: string;
  /** `GPUAdapterInfo.architecture` — see the two tables above for what it is worth. */
  adapterArchitecture?: string;
  /** `GPUSupportedLimits.maxBufferSize`, in bytes. A lower bound on device memory, never a capacity. */
  maxBufferBytes?: number;
  /** `navigator.deviceMemory`, in GiB. Capped at 8 in Chrome, absent in Safari. */
  deviceMemoryGiB?: number;
  /** `navigator.userAgentData.platform` or `navigator.platform`, however it spells itself. */
  platform?: string;
}

export interface Detection {
  /** Rows the signals do not rule out, in catalog order. Never a selection. */
  candidates: readonly DeviceSpec[];
  /** What was read and what each reading did, in words, for the confirmation step to print. */
  evidence: readonly string[];
  /**
   * Why the reader still has to choose, when the signals could not get close.
   *
   * Present whenever the shortlist is too long to be a shortlist. The issue's own scoping: a
   * follow-up question is a first-class path, so the surface needs to know when to ask one and
   * what to ask about.
   */
  askAbout?: 'memory' | 'machine';
  /**
   * Set when a signal was dropped because applying it would have left no machine at all.
   *
   * The reader's browser said two things that cannot both be true of a catalogued row — an Intel
   * Mac is the reachable case — so the surface should say the narrowing is partial rather than
   * present the survivors as though every signal agreed.
   */
  conflicted?: true;
}

/** Above this the list is not a shortlist and the reader is better served by a question. */
const SHORTLIST_LIMIT = 6;

/**
 * Narrow the catalog by what the browser admitted.
 *
 * Pure and synchronous — reading the adapter is the caller's job, so this is testable against
 * recorded signals, which is the half jsdom can answer. Every filter is a *ruling out*: a row
 * survives unless a signal contradicts it, so an absent signal narrows nothing rather than
 * excluding everything.
 */
export function detect(signals: DetectionSignals, devices: readonly DeviceSpec[]): Detection {
  const evidence: string[] = [];
  // Shipping rows only. A `rumored` row is a spec nobody can own yet, and offering one as "your
  // machine" is the pre-release-labelling rule broken in the most direct way available.
  let candidates = devices.filter((d) => d.status === 'shipping');
  let conflicted = false;

  /**
   * Apply a prune, unless it would leave nothing.
   *
   * One helper rather than a guard repeated at each filter, because the failure is the same shape
   * every time and a hand-written copy per filter is how one of them comes to be missing — which is
   * exactly what happened: the platform prune had no guard and emptied the list on an Intel Mac.
   *
   * Returns whether it narrowed, so a caller only pushes its sentence when there is something to
   * say about it.
   */
  const prune = (keep: (device: DeviceSpec) => boolean): 'narrowed' | 'unchanged' | 'conflict' => {
    const next = candidates.filter(keep);
    if (next.length === 0) {
      conflicted = true;
      return 'conflict';
    }
    if (next.length === candidates.length) return 'unchanged';
    candidates = next;
    return 'narrowed';
  };

  const architecture = signals.adapterArchitecture?.toLowerCase();
  const isApple = architecture !== undefined && APPLE_FAMILIES.includes(architecture);
  const vendor =
    (architecture === undefined
      ? undefined
      : isApple
        ? 'Apple'
        : ARCHITECTURE_VENDORS[architecture]) ?? vendorFromString(signals.adapterVendor);

  /**
   * **A GPU adapter is evidence about a GPU, so the CPU rows are ruled out rather than matched**
   * (raised by Codex on #168). Matching on vendor alone offered an Intel reader the `xeon-6980p`
   * CPU-RAM row beside the Arc GPUs, on a signal that says nothing about the host CPU at all — and
   * on a machine whose CPU and GPU vendors differ it would have excluded the CPU row they own. A
   * reader running on CPU has no adapter worth detecting; the picker is their path.
   */
  if (
    vendor !== undefined &&
    prune((d) => d.vendor === vendor && d.class !== 'cpu-ram') !== 'conflict'
  ) {
    evidence.push(
      isApple
        ? `Your browser reports an Apple GPU. It reports the Metal feature family (${architecture}) ` +
            `rather than the chip, and every Apple silicon Mac reports one of three — so this says ` +
            `which vendor and nothing about which Mac.`
        : architecture !== undefined && ARCHITECTURE_VENDORS[architecture] !== undefined
          ? `Your browser reports a ${architecture} GPU, which is ${vendor} silicon.`
          : `Your browser reports a ${vendor} GPU.`
    );
  }

  /**
   * The platform prune, and it only runs in one direction.
   *
   * macOS implies Apple silicon among the shipping rows, which is a real narrowing. The converse is
   * not true in the way it looks: a Windows or Linux machine rules out the Apple rows, and says
   * nothing about which of the remaining thirty-odd it is.
   */
  /**
   * **The platform corroborates a vendor; it never establishes one** (raised by Codex on #168).
   *
   * The conflict rule below only fires once a *vendor* prune has happened, so on an Intel Mac whose
   * adapter info is withheld — a state `readSignals` explicitly supports — `MacIntel` alone kept
   * only the Apple rows and reported Apple silicon confidently. There was nothing to conflict with.
   * So the macOS arm requires the adapter to have said Apple already; without a vendor it says
   * nothing, which is the honest reading of a platform string that Intel Macs also produce.
   *
   * The non-macOS arm is unaffected and stays unconditional: no Apple silicon Mac reports a
   * non-macOS platform, so ruling the Apple rows out there needs no corroboration.
   */
  const platform = signals.platform?.toLowerCase();
  if (platform !== undefined && platform !== '') {
    const mac = platform.includes('mac') || platform.includes('darwin');
    if (mac && vendor === undefined) {
      evidence.push(
        'The platform is macOS, but the adapter was withheld — and Intel Macs run macOS too, so ' +
          'this alone does not say the machine is Apple silicon.'
      );
    } else {
      const outcome = prune((d) => (mac ? d.vendor === 'Apple' : d.vendor !== 'Apple'));

      if (outcome === 'conflict') {
        /**
         * **An Intel Mac, and the comment this replaced said macOS "implies Apple silicon".** It does
         * not: Chrome ships WebGPU on Metal for Intel Macs, whose adapter is Intel or AMD. Taking the
         * platform at its word there leaves no rows at all, and tells a reader whose Mac is not Apple
         * silicon that it is. The adapter is the more specific witness, so it wins.
         */
        evidence.push(
          mac
            ? 'The platform is macOS but the adapter is not an Apple GPU — an Intel Mac. The adapter ' +
                'is the more specific signal, so the platform is ignored here.'
            : 'The adapter reports an Apple GPU on a platform that is not macOS, which cannot both ' +
                'be true of a catalogued machine. The platform is ignored here.'
        );
      } else if (outcome === 'narrowed') {
        evidence.push(
          mac
            ? 'The platform is macOS and the adapter is an Apple GPU, so this is Apple silicon.'
            : 'The platform is not macOS, which rules out the Apple rows and little else.'
        );
      }
    }
  }

  /**
   * **`maxBufferSize` is read but never pruned on, and dropping that prune was the second review's
   * correction.**
   *
   * It looked like a sound floor: the largest single allocation the driver will hand out, capped
   * well under total memory, so a device below it is impossible. But it is a *validation* ceiling
   * on a buffer descriptor rather than a promise that such a buffer can be allocated — WebGPU
   * checks a request against it and can still fail with an out-of-memory error. So a limit above a
   * device's real capacity is not a contradiction, and pruning on it removed the reader's actual
   * machine.
   *
   * That is the one failure this module cannot accept: a shortlist without the right answer in it
   * leaves the confirmation step with nothing to confirm. It is reported as evidence, where the
   * reader can weigh it, and narrows nothing.
   */
  if (signals.maxBufferBytes !== undefined && signals.maxBufferBytes > 0) {
    evidence.push(
      `The adapter accepts a single buffer of up to ${(signals.maxBufferBytes / GIB).toFixed(1)} ` +
        `GiB. That is a validation limit rather than a promise the memory exists, so it is not ` +
        `used to rule any machine out — but a very small figure is a hint that the machine is too.`
    );
  }

  /**
   * `deviceMemory` prunes upward only, and the cap is why.
   *
   * Chrome clamps it to 8, so a reading of 8 means "8 or more" and rules out nothing at the top.
   * A reading *below* 8 is a real ceiling on system RAM, which on a unified-memory machine is also
   * a ceiling on the GPU's memory. Applied only there, since a discrete card's VRAM is unrelated to
   * how much RAM the host has.
   *
   * **The factor of two is the spec's, not a fudge.** Device Memory reports actual RAM rounded
   * *down* to a power of two and then clamped to [0.25, 8], so a reading of `r` means the machine
   * has somewhere in `[r, 2r)`. `capacity <= 2r` is therefore the widest bound that is still sound,
   * and it is deliberately generous at the boundary: the direction that matters is never excluding
   * the reader's real machine.
   */
  if (signals.deviceMemoryGiB !== undefined && signals.deviceMemoryGiB < 8) {
    const reported = signals.deviceMemoryGiB;
    if (
      prune((d) => d.class !== 'unified-soc' || d.capacityBytes / GIB <= reported * 2) ===
      'narrowed'
    ) {
      evidence.push(
        `The browser reports about ${signals.deviceMemoryGiB} GiB of system memory, which bounds a ` +
          `unified-memory machine. Chrome caps this figure at 8, so it is only ever evidence of a ` +
          `small machine.`
      );
    }
  }

  return {
    candidates,
    evidence,
    ...(conflicted ? { conflicted: true as const } : {}),
    askAbout:
      candidates.length <= SHORTLIST_LIMIT
        ? undefined
        : candidates.every((d) => d.class === 'unified-soc')
          ? 'memory'
          : 'machine',
  };
}

/**
 * `GPUAdapterInfo.vendor`, matched loosely on purpose.
 *
 * The spec says the field is implementation-defined, and browsers have shipped `nvidia`, `intel`,
 * `apple` and — on some Linux stacks — the PCI vendor string with punctuation in it. Substring
 * matching against the four vendors the catalog has rows for is what survives that, and an
 * unrecognised string returns `undefined` rather than a nearest guess.
 */
function vendorFromString(raw: string | undefined): DeviceSpec['vendor'] | undefined {
  if (raw === undefined) return undefined;
  const value = raw.toLowerCase();
  if (value.includes('nvidia')) return 'NVIDIA';
  if (value.includes('apple')) return 'Apple';
  if (value.includes('intel')) return 'Intel';
  // Last, and the two spellings are why: AMD's adapter string is often `amd` and sometimes the
  // marketing name, and `radeon` appears in both. Neither collides with the three above.
  // `ati` on a word boundary, never as a substring: it sits inside "Imagination", so the loose form
  // classified a PowerVR adapter as AMD and removed every non-AMD row — the opposite of the stated
  // fallback for an unrecognised vendor. Raised by Codex on #168.
  if (value.includes('amd') || value.includes('radeon') || /\bati\b/.test(value)) return 'AMD';
  return undefined;
}

/**
 * Read what this browser will say, or nothing at all.
 *
 * The only impure function here, and it is deliberately tiny: everything it returns is a plain
 * record, so `detect` stays testable against recorded fixtures. Every access is guarded, because
 * each of these is absent somewhere real — `navigator.gpu` in Safari behind a flag, `deviceMemory`
 * in Safari and Firefox, `userAgentData` outside Chromium — and a detection affordance that throws
 * is worse than one that is not offered.
 */
export async function readSignals(): Promise<DetectionSignals | undefined> {
  const gpu = (navigator as Navigator & { gpu?: GPUFallback }).gpu;
  if (gpu?.requestAdapter === undefined) return undefined;

  try {
    /**
     * **Both power preferences, because the default is one GPU and a laptop has two** (raised by
     * Codex on #168). `requestAdapter()` with no preference returns the browser's own default,
     * which on a dual-GPU laptop is routinely the integrated Intel part — and the vendor prune then
     * removed the discrete card the reader actually cares about, with no way to get it back.
     *
     * The discrete one is the machine bench is about, so `high-performance` is asked first and its
     * answer wins. The default is the fallback, not the other way round.
     */
    const adapter =
      (await gpu.requestAdapter({ powerPreference: 'high-performance' })) ??
      (await gpu.requestAdapter());
    // A browser with `navigator.gpu` and no adapter is a real state — a blocklisted driver, a
    // headless run — and it means the picker, not an error.
    if (!adapter) return undefined;

    /**
     * `GPUAdapter.info` is newer than the adapter itself, and browsers that predate it expose the
     * same fields through the deprecated async `requestAdapterInfo()`. Without the fallback those
     * otherwise-capable browsers reported everything withheld and fell to the platform-only path,
     * which is the weakest reading available. Raised by Codex on #168.
     */
    const legacy = adapter as { requestAdapterInfo?: () => Promise<AdapterInfo | undefined> };
    const info = adapter.info ?? (await legacy.requestAdapterInfo?.());

    const nav = navigator as Navigator & {
      deviceMemory?: number;
      userAgentData?: { platform?: string };
    };

    return {
      adapterVendor: info?.vendor,
      adapterArchitecture: info?.architecture,
      maxBufferBytes: adapter.limits?.maxBufferSize,
      deviceMemoryGiB: nav.deviceMemory,
      platform: nav.userAgentData?.platform ?? navigator.platform,
    };
  } catch {
    // `requestAdapter` rejects rather than resolving to null in some embedded browsers. Same
    // meaning either way, and the same answer: fall back without a console error.
    return undefined;
  }
}

/**
 * The shape actually read, rather than the full WebGPU type surface.
 *
 * `@webgpu/types` is not a dependency and adding one for four optional fields would put a
 * build-time dependency behind a progressive enhancement. Every field is optional because every
 * one is: `GPUAdapter.info` is newer than `requestAdapter`, and older Chromiums expose neither.
 */
/** The two fields read off an adapter, however this browser exposes them. */
interface AdapterInfo {
  vendor?: string;
  architecture?: string;
}

interface GPUFallback {
  requestAdapter?: (options?: { powerPreference?: 'high-performance' | 'low-power' }) => Promise<{
    info?: AdapterInfo;
    /** The deprecated predecessor of `info`, still the only route on older Chromiums. */
    requestAdapterInfo?: () => Promise<AdapterInfo | undefined>;
    limits?: { maxBufferSize?: number };
  } | null>;
}
