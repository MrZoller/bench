import { describe, expect, it } from 'vitest';
import { getDevice } from '@/data/catalog';
import {
  DGX_SPARK,
  EPYC_9654,
  MAC_STUDIO_M3_ULTRA_256,
  MAC_STUDIO_M3_ULTRA_512,
  RTX_4090,
  RTX_5090,
  STRIX_HALO_395,
} from './fixtures';

/**
 * A fixture device that shares an id with a catalog row is a claim about the same machine, so
 * the physics must agree (#116).
 *
 * The fixtures are deliberately frozen, but that rationale — stability across *refreshes* —
 * only ever applied to the generated model catalog. Device physics is hand-verified in both
 * places, and the divergence this guards against has happened twice: the 4090 fixture carried
 * half the card's tensor compute (one halving too many off the sparse headline, the exact
 * curator error devices.json's $comment-compute warns about), and the EPYC fixture kept the
 * double-discounted 6 TFLOPS that #111 corrected to the 7.37 theoretical vector peak in the
 * catalog. Both were latent — every test passed — while every test touching those devices
 * measured hardware nobody sells.
 *
 * Physics fields only: name, status, price, TDP and provenance are curation, and the model
 * fixtures stay frozen by design.
 */
describe('fixture devices agree with the catalog rows they name', () => {
  const FIXTURES = [
    RTX_5090,
    RTX_4090,
    DGX_SPARK,
    MAC_STUDIO_M3_ULTRA_256,
    MAC_STUDIO_M3_ULTRA_512,
    STRIX_HALO_395,
    EPYC_9654,
  ];

  it.each(FIXTURES.map((f) => [f.id, f] as const))('%s', (_id, fixture) => {
    const row = getDevice(fixture.id);
    expect(row, `no catalog row carries the id ${fixture.id}`).toBeDefined();

    expect(fixture.class).toBe(row!.class);
    expect(fixture.capacityBytes).toBe(row!.capacityBytes);
    expect(fixture.allocatableBytes).toBe(row!.allocatableBytes);
    expect(fixture.bandwidthBytesPerSec).toBeCloseTo(row!.bandwidthBytesPerSec, -6);
    expect(fixture.interconnect).toBe(row!.interconnect);
    expect(fixture.hostLinkBytesPerSec).toBe(row!.hostLinkBytesPerSec);
    expect(fixture.allocatableTunable).toBe(row!.allocatableTunable);
    expect(fixture.maxAllocatableBytes).toBe(row!.maxAllocatableBytes);

    // Per dtype and in both directions: an entry the fixture lacks falls through peakFlops'
    // fallback chain to a different rate — the 4090's absent int8 read 330 where the card
    // publishes 661 — so absence diverges as surely as a wrong number.
    const dtypes = new Set([...Object.keys(fixture.flops ?? {}), ...Object.keys(row!.flops ?? {})]);
    for (const dtype of dtypes) {
      const key = dtype as keyof NonNullable<typeof fixture.flops>;
      expect(
        fixture.flops?.[key],
        `${fixture.id} ${dtype}: fixture ${fixture.flops?.[key]} vs catalog ${row!.flops?.[key]}`
      ).toBeCloseTo(row!.flops?.[key] ?? Number.NaN, -9);
    }
  });
});
