import { describe, expect, it } from 'vitest';
import {
  APPLE_FAMILIES,
  ARCHITECTURE_VENDORS,
  detect,
  readSignals,
  type DetectionSignals,
} from './detect';
import { DEVICES } from '@/data/catalog';
import { GIB } from '@/engine/types';

/**
 * Machine detection (#137).
 *
 * The tests are about **what detection refuses to claim**, because that is the whole contract: a
 * wrong guess silently applied is invented data wearing the chassis of a measurement. So the
 * assertions below are mostly that a shortlist still contains the right answer, rather than that it
 * is short — a narrowing that excludes the reader's actual machine is worse than no narrowing.
 *
 * Recorded signals rather than a live adapter, which is the split the issue names: jsdom has no
 * `navigator.gpu` at all, so the mapping is unit-testable here and the wiring is e2e territory.
 */

const shipping = DEVICES.filter((d) => d.status === 'shipping');
const ids = (signals: DetectionSignals) => detect(signals, DEVICES).candidates.map((d) => d.id);

describe('the architecture table is Chrome’s own spelling', () => {
  it('hyphenates before a letter and not after a digit', () => {
    // `js_enum_case` joins chunks with a hyphen *except* after a digit. A hand-written table gets
    // the Intel rows wrong, and a web search returned both `rdna-3` and `rdna4` for the same field.
    expect(ARCHITECTURE_VENDORS['rdna-3']).toBe('AMD');
    expect(ARCHITECTURE_VENDORS['gen-12lp']).toBe('Intel');
    expect(ARCHITECTURE_VENDORS['xe-2hpg']).toBe('Intel');
    // The spellings a plausible transcription would have produced instead, absent.
    expect(ARCHITECTURE_VENDORS['rdna3']).toBeUndefined();
    expect(ARCHITECTURE_VENDORS['gen-12-lp']).toBeUndefined();
  });

  it('carries no Apple entry, because Apple reports a feature family', () => {
    // The finding that matters most here: Apple GPUs report no DeviceID through Metal, so Dawn
    // reports the highest supported common family. Every Apple silicon Mac reports one of three,
    // so on the platform where a unified-memory row is the headline case the architecture
    // identifies the vendor and nothing else.
    for (const family of APPLE_FAMILIES) {
      expect(ARCHITECTURE_VENDORS[family]).toBeUndefined();
    }
  });
});

describe('a signal narrows, and an absent signal narrows nothing', () => {
  it('returns every shipping row when the browser said nothing', () => {
    expect(ids({})).toEqual(shipping.map((d) => d.id));
  });

  it('never offers a pre-release row as a machine somebody owns', () => {
    // `devices.json` carries `rumored` rows on purpose, and offering one as "your machine" breaks
    // the pre-release labelling rule in the most direct way available.
    const rumoured = DEVICES.filter((d) => d.status !== 'shipping');
    expect(rumoured.length, 'no non-shipping rows, so this checks nothing').toBeGreaterThan(0);

    const everything = ids({});
    for (const row of rumoured) expect(everything).not.toContain(row.id);
  });

  it('takes the vendor from the architecture when the vendor string is withheld', () => {
    const detected = ids({ adapterArchitecture: 'blackwell' });

    expect(detected.length).toBeGreaterThan(0);
    for (const id of detected) {
      expect(DEVICES.find((d) => d.id === id)!.vendor).toBe('NVIDIA');
    }
  });

  it('reads a vendor string loosely, since browsers spell it several ways', () => {
    // The field is implementation-defined, and Linux stacks have shipped the PCI vendor string.
    for (const raw of ['nvidia', 'NVIDIA Corporation', 'nvidia corporation']) {
      expect(ids({ adapterVendor: raw }).length, raw).toBeGreaterThan(0);
      expect(
        new Set(ids({ adapterVendor: raw }).map((id) => DEVICES.find((d) => d.id === id)!.vendor))
      ).toEqual(new Set(['NVIDIA']));
    }
  });

  it('narrows nothing on a vendor string it does not recognise', () => {
    // A nearest guess is the failure this whole module is organised against.
    expect(ids({ adapterVendor: 'Some Future Vendor' })).toEqual(shipping.map((d) => d.id));
  });
});

describe('the shortlist keeps the right answer in it', () => {
  /**
   * The property that matters more than shortness. A filter that excludes the reader's actual
   * machine is worse than no filter, because the confirmation step then cannot recover — they are
   * asked to pick their machine from a list it is not in.
   */
  it('keeps a 5090 for every signal set a 5090 would really produce', () => {
    const rtx5090 = DEVICES.find((d) => d.id === 'rtx-5090')!;

    const plausible: DetectionSignals[] = [
      { adapterVendor: 'nvidia' },
      { adapterVendor: 'nvidia', adapterArchitecture: 'blackwell' },
      { adapterVendor: 'nvidia', adapterArchitecture: 'blackwell', platform: 'Windows' },
      {
        adapterVendor: 'nvidia',
        adapterArchitecture: 'blackwell',
        platform: 'Windows',
        // A driver-capped single-buffer maximum well under the card's 32 GiB.
        maxBufferBytes: 4 * GIB,
        deviceMemoryGiB: 8,
      },
    ];

    for (const signals of plausible) {
      expect(ids(signals), JSON.stringify(signals)).toContain(rtx5090.id);
    }
  });

  it('keeps a Mac Studio for the signals a Mac really produces', () => {
    const signals: DetectionSignals = {
      adapterVendor: 'apple',
      adapterArchitecture: 'common-3',
      platform: 'macOS',
    };
    const detected = ids(signals);

    expect(detected).toContain('mac-studio-m3-ultra-256');
    // And nothing but Apple survives, which is the one thing a Mac's signals do settle.
    for (const id of detected) expect(DEVICES.find((d) => d.id === id)!.vendor).toBe('Apple');
  });
});

describe('what each signal is worth is stated, and it is less than it looks', () => {
  it('says the Apple architecture identifies the vendor and not the Mac', () => {
    const { evidence, askAbout } = detect(
      { adapterVendor: 'apple', adapterArchitecture: 'common-3', platform: 'macOS' },
      DEVICES
    );

    expect(evidence.join(' ')).toMatch(/Metal feature family/i);
    expect(evidence.join(' ')).toMatch(/nothing about which Mac/i);
    // Ten shipping Apple rows is not a shortlist, so the surface has to ask rather than pretend.
    expect(askAbout).toBe('memory');
  });

  it('reports the buffer limit as evidence and rules nothing out on it', () => {
    /**
     * **This was a prune and should not have been**, which the second review round corrected.
     * `maxBufferSize` is a *validation* ceiling on a buffer descriptor rather than a promise the
     * memory exists — WebGPU checks a request against it and can still fail with an out-of-memory
     * error — so a limit above a device's real capacity is not a contradiction, and pruning on it
     * removed the reader's actual machine. That is the one failure this module cannot accept.
     */
    const withLimit = detect({ adapterVendor: 'nvidia', maxBufferBytes: 20 * GIB }, DEVICES);

    expect(withLimit.candidates.map((d) => d.id)).toEqual(ids({ adapterVendor: 'nvidia' }));
    expect(withLimit.evidence.join(' ')).toMatch(/validation limit rather than a promise/i);
  });

  it('reads a capped deviceMemory as evidence of a small machine and nothing else', () => {
    // Chrome clamps the figure at 8, so a reading *of* 8 means "8 or more" and must rule nothing
    // out at the top. Only a reading below it is a real ceiling.
    expect(ids({ adapterVendor: 'apple', deviceMemoryGiB: 8 })).toEqual(
      ids({ adapterVendor: 'apple' })
    );
  });

  it('drops a memory reading no catalogued machine could satisfy', () => {
    /**
     * The conflict rule doing its job on a signal I first expected to narrow. Device Memory rounds
     * *down* to a power of two, so a reading of 4 means RAM in [4, 8) — and the smallest Apple row
     * in the catalog is 16 GiB. Applying it leaves nothing, so it is dropped and recorded rather
     * than producing an empty shortlist.
     */
    const bounded = detect({ adapterVendor: 'apple', deviceMemoryGiB: 4 }, DEVICES);

    expect(bounded.candidates.map((d) => d.id)).toEqual(ids({ adapterVendor: 'apple' }));
    expect(bounded.conflicted).toBe(true);
  });

  it('asks about the machine when a vendor alone leaves too many rows', () => {
    // The issue's own scoping: on a redacting browser, vendor-only maps to seventeen shipping
    // NVIDIA rows, and a follow-up question is a first-class path rather than a failure branch.
    const { candidates, askAbout } = detect({ adapterVendor: 'nvidia' }, DEVICES);

    expect(candidates.length).toBeGreaterThan(6);
    expect(askAbout).toBe('machine');
  });

  it('asks nothing once the list is short enough to be a shortlist', () => {
    const { candidates, askAbout } = detect(
      { adapterVendor: 'intel', platform: 'Windows' },
      DEVICES
    );

    expect(candidates.length).toBeLessThanOrEqual(6);
    expect(askAbout).toBeUndefined();
  });
});

describe('what a GPU adapter is and is not evidence about', () => {
  it('does not infer Apple silicon from macOS alone', () => {
    /**
     * The conflict rule only fires once a *vendor* prune has happened — so on an Intel Mac whose
     * adapter info is withheld, `MacIntel` alone kept only the Apple rows and reported Apple
     * silicon confidently, with nothing to conflict against. The platform corroborates a vendor; it
     * never establishes one.
     */
    const withheld = detect({ platform: 'MacIntel' }, DEVICES);

    expect(withheld.candidates.map((d) => d.id)).toEqual(shipping.map((d) => d.id));
    expect(withheld.evidence.join(' ')).toMatch(/Intel Macs run macOS too/i);
  });

  it('still narrows on macOS once the adapter has said Apple', () => {
    // The corroborating case, which is what the platform prune is for.
    const confirmed = detect({ adapterVendor: 'apple', platform: 'macOS' }, DEVICES);
    for (const device of confirmed.candidates) expect(device.vendor).toBe('Apple');
  });

  it('offers no CPU row on the strength of a GPU adapter', () => {
    // An Intel adapter says nothing about the host CPU, and matching on vendor alone offered the
    // Xeon row beside the Arc GPUs. On a machine whose CPU and GPU vendors differ it would also
    // have excluded the CPU row the reader owns.
    for (const device of detect({ adapterVendor: 'intel' }, DEVICES).candidates) {
      expect(device.class, device.id).not.toBe('cpu-ram');
    }
  });

  it('does not read "ati" out of the middle of another word', () => {
    // `ati` sits inside "Imagination", so the loose form classified a PowerVR adapter as AMD and
    // removed every non-AMD row — the opposite of the stated fallback for an unrecognised vendor.
    expect(ids({ adapterVendor: 'Imagination Technologies' })).toEqual(shipping.map((d) => d.id));
    // And the legacy name on its own still resolves, so the token match is not simply a deletion.
    const legacy = detect({ adapterVendor: 'ATI Technologies Inc.' }, DEVICES);
    for (const device of legacy.candidates) expect(device.vendor).toBe('AMD');
  });
});

describe('signals that contradict each other keep the machine in the list', () => {
  /**
   * **An Intel Mac, which is reachable hardware and broke the first version outright.**
   *
   * Chrome ships WebGPU on Metal for Intel Macs, whose adapter is Intel or AMD — so the vendor
   * narrows to the Intel rows and the platform prune then keeps only Apple rows, leaving none. The
   * surface rendered "Which of these is yours?" over an empty list, and the evidence told a reader
   * whose Mac is not Apple silicon that it is.
   */
  it('keeps the adapter’s rows when the platform contradicts it', () => {
    const intelMac = detect({ adapterVendor: 'intel', platform: 'MacIntel' }, DEVICES);

    expect(intelMac.candidates.length).toBeGreaterThan(0);
    for (const device of intelMac.candidates) expect(device.vendor).toBe('Intel');
    expect(intelMac.conflicted).toBe(true);
    expect(intelMac.evidence.join(' ')).toMatch(/an Intel Mac/i);
    // And it must not have kept the sentence that was false for this reader.
    expect(intelMac.evidence.join(' ')).not.toMatch(/so this is Apple silicon/i);
  });

  it('tells a phone there is nothing here rather than asking which Mac it is', () => {
    /**
     * The mirror case, and it needed a different answer from the Intel Mac's. An iPhone exposing
     * WebGPU has its Apple adapter narrow the list *to Macs*, and the conflict guard then put them
     * back when the platform emptied it — so the panel asked an iPhone which Mac it was. The
     * catalog has no phone row, so the honest answer is a terminal state.
     */
    const phone = detect({ adapterVendor: 'apple', platform: 'iPhone' }, DEVICES);

    expect(phone.unsupportedPlatform).toBe('phone');
    expect(phone.evidence.join(' ')).toMatch(/no rows for/i);
  });

  it('separates an iPad from a Mac by the one signal desktop Safari does not fake', () => {
    // iPadOS Safari's desktop-class mode reports `MacIntel` with a genuine Apple adapter, so every
    // other signal agrees with a Mac. `maxTouchPoints` is what does not.
    const ipad = detect(
      { adapterVendor: 'apple', platform: 'MacIntel', maxTouchPoints: 5 },
      DEVICES
    );
    expect(ipad.unsupportedPlatform).toBe('tablet');

    // And a real Mac, which reports zero, is unaffected.
    const mac = detect(
      { adapterVendor: 'apple', platform: 'MacIntel', maxTouchPoints: 0 },
      DEVICES
    );
    expect(mac.unsupportedPlatform).toBeUndefined();
    for (const device of mac.candidates) expect(device.vendor).toBe('Apple');
  });

  it('reads Android from the browser’s own flag, since Android does not say Android', () => {
    // With `userAgentData` absent — every non-Chromium Android browser — `navigator.platform`
    // reports a Linux value like `Linux armv8l`, which took the non-macOS arm and pruned the Apple
    // rows as though this were a desktop. `mobile` is the boolean designed for the question.
    const android = detect(
      { adapterVendor: 'arm', platform: 'Linux armv8l', mobile: true },
      DEVICES
    );
    expect(android.unsupportedPlatform).toBe('phone');
  });

  it('recognises AMD by its legal name as well as its brands', () => {
    // An implementation-defined vendor string is exactly where a legal name turns up, and
    // "Advanced Micro Devices, Inc." contains neither `amd` nor `radeon`.
    const legal = detect({ adapterVendor: 'Advanced Micro Devices, Inc.' }, DEVICES);
    expect(legal.candidates.length).toBeGreaterThan(0);
    for (const device of legal.candidates) expect(device.vendor).toBe('AMD');
  });

  it('ignores a platform string it does not recognise', () => {
    // A hardened browser returning `Unknown` was classified as definitively non-macOS and pruned
    // the Apple rows on the strength of a string nobody parsed.
    const hardened = detect({ platform: 'Unknown' }, DEVICES);

    expect(hardened.candidates.map((d) => d.id)).toEqual(shipping.map((d) => d.id));
    expect(hardened.evidence.join(' ')).toMatch(/not a name this recognises/i);
  });

  it('says so when nothing narrowed, rather than offering the whole catalog', () => {
    // Every safeguard correctly declining to narrow left the panel asking "which of these is
    // yours?" over forty-two rows, which is the picker with extra steps.
    expect(detect({}, DEVICES).narrowedNothing).toBe(true);
    expect(detect({ adapterVendor: 'nvidia' }, DEVICES).narrowedNothing).toBeUndefined();
  });

  it('never returns an empty shortlist, whatever the signals say', () => {
    /**
     * The property the whole conflict rule exists for: a surface that offers a confirmation needs
     * something to confirm. Swept rather than spot-checked, because the failure was an *interaction*
     * between two filters that each looked right alone.
     */
    const vendors = [undefined, 'nvidia', 'amd', 'intel', 'apple', 'nonsense'];
    const platforms = [undefined, 'macOS', 'MacIntel', 'Windows', 'Linux', 'iPhone', ''];
    const buffers = [undefined, GIB, 20 * GIB, 4096 * GIB];
    const memories = [undefined, 0.25, 4, 8];

    for (const adapterVendor of vendors)
      for (const platform of platforms)
        for (const maxBufferBytes of buffers)
          for (const deviceMemoryGiB of memories) {
            const signals = { adapterVendor, platform, maxBufferBytes, deviceMemoryGiB };
            expect(
              detect(signals, DEVICES).candidates.length,
              JSON.stringify(signals)
            ).toBeGreaterThan(0);
          }
  });
});

describe('it degrades to the picker rather than failing', () => {
  it('reads nothing when the browser has no WebGPU at all', async () => {
    // jsdom has no `navigator.gpu`, which is also Safari behind a flag and any hardened browser.
    // The answer is `undefined` — meaning "use the picker" — with no console error.
    expect('gpu' in navigator).toBe(false);
    await expect(readSignals()).resolves.toBeUndefined();
  });

  it('reads nothing when there is a WebGPU object but no adapter', async () => {
    // A blocklisted driver or a headless run. A real state, and it means the picker.
    const stub = { requestAdapter: () => Promise.resolve(null) };
    Object.defineProperty(navigator, 'gpu', { value: stub, configurable: true });
    try {
      await expect(readSignals()).resolves.toBeUndefined();
    } finally {
      Reflect.deleteProperty(navigator, 'gpu');
    }
  });

  it('reads nothing when requesting the adapter rejects', async () => {
    // `requestAdapter` rejects rather than resolving to null in some embedded browsers. Same
    // meaning, same answer, and no unhandled rejection.
    const stub = { requestAdapter: () => Promise.reject(new Error('blocked')) };
    Object.defineProperty(navigator, 'gpu', { value: stub, configurable: true });
    try {
      await expect(readSignals()).resolves.toBeUndefined();
    } finally {
      Reflect.deleteProperty(navigator, 'gpu');
    }
  });

  it('survives an adapter that reports no info, which is every older Chromium', async () => {
    // `GPUAdapter.info` is newer than `requestAdapter`, so the fields have to be optional all the
    // way down rather than optional at the top.
    const stub = { requestAdapter: () => Promise.resolve({}) };
    Object.defineProperty(navigator, 'gpu', { value: stub, configurable: true });
    try {
      const signals = await readSignals();
      expect(signals).toBeDefined();
      expect(signals!.adapterVendor).toBeUndefined();
      expect(signals!.adapterArchitecture).toBeUndefined();
      // And an empty reading narrows nothing, rather than excluding everything.
      expect(detect(signals!, DEVICES).candidates.length).toBeGreaterThan(0);
    } finally {
      Reflect.deleteProperty(navigator, 'gpu');
    }
  });
});
